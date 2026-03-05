use std::sync::Arc;

use axum::{
    body::Body,
    extract::{ConnectInfo, Request, State},
    http::{StatusCode, Uri},
    response::Response,
};
use bytes::Bytes;
use chrono::Utc;
use http_body_util::BodyExt;
use tokio::sync::mpsc;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::{
    auth::AppName,
    connection_id::ConnectionId,
    langfuse::{LangfuseCollector, LangfuseEvent},
    ollama::is_streaming,
    state::AppState,
};

/// Endpoints that get Langfuse tracing.
const TRACED_PATHS: &[&str] = &["/api/chat", "/api/generate", "/api/embed", "/api/embeddings"];

pub async fn proxy_handler(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    let method = req.method().clone();
    let url = req.uri().path_and_query().map(|pq| pq.as_str()).unwrap_or("/").to_string();
    let remote = req
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|c| c.0.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    info!(method = %method, url = %url, remote = %remote, "→");

    let collector = state.langfuse_collector.read().await.clone();
    let should_trace = collector.is_some()
        && TRACED_PATHS.iter().any(|p| url == *p || url.starts_with(p));

    let result = if should_trace {
        proxy_with_tracing(state, collector, req).await
    } else {
        proxy_passthrough(state, req).await
    };

    match &result {
        Ok(resp) => info!(method = %method, url = %url, remote = %remote, status = %resp.status(), "←"),
        Err(status) => info!(method = %method, url = %url, remote = %remote, status = %status, "←"),
    }

    result
}

/// Zero-copy passthrough for non-traced endpoints.
async fn proxy_passthrough(
    state: Arc<AppState>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    let upstream_url = state.upstream_url.read().await.clone();
    let upstream_req = build_upstream_request(&upstream_url, req).map_err(|e| {
        error!("Failed to build upstream request: {}", e);
        StatusCode::BAD_GATEWAY
    })?;

    let resp = state.http_client.request(upstream_req).await.map_err(|e| {
        error!("Upstream request failed: {}", e);
        StatusCode::BAD_GATEWAY
    })?;

    Ok(convert_response(resp))
}

/// Proxy with body buffering for Langfuse tracing.
async fn proxy_with_tracing(
    state: Arc<AppState>,
    collector: Option<Arc<LangfuseCollector>>,
    req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    let app_name = req
        .extensions()
        .get::<AppName>()
        .map(|a| a.0.clone())
        .unwrap_or_default();

    let session_id = req
        .extensions()
        .get::<ConnectionId>()
        .map(|c| c.0.clone());

    let path = req.uri().path().to_string();
    let start_time = Utc::now();

    // Buffer request body
    let (parts, body) = req.into_parts();
    let req_bytes = body
        .collect()
        .await
        .map(|c| c.to_bytes())
        .unwrap_or_default();

    let req_json: Option<serde_json::Value> = serde_json::from_slice(&req_bytes).ok();
    let streaming = req_json.as_ref().map(is_streaming).unwrap_or(true);

    // Rebuild request with buffered body
    let upstream_url = state.upstream_url.read().await.clone();
    let rebuilt = Request::from_parts(parts, Body::from(req_bytes.clone()));
    let upstream_req = build_upstream_request(&upstream_url, rebuilt).map_err(|e| {
        error!("Failed to build upstream request: {}", e);
        StatusCode::BAD_GATEWAY
    })?;

    let resp = state.http_client.request(upstream_req).await.map_err(|e| {
        error!("Upstream request failed: {}", e);
        StatusCode::BAD_GATEWAY
    })?;

    if streaming {
        handle_streaming_response(collector, resp, req_json, app_name, path, start_time, session_id).await
    } else {
        handle_non_streaming_response(collector, resp, req_json, app_name, path, start_time, session_id).await
    }
}

async fn handle_non_streaming_response(
    collector: Option<Arc<LangfuseCollector>>,
    resp: hyper::Response<hyper::body::Incoming>,
    req_json: Option<serde_json::Value>,
    app_name: String,
    path: String,
    start_time: chrono::DateTime<chrono::Utc>,
    session_id: Option<String>,
) -> Result<Response<Body>, StatusCode> {
    let (resp_parts, resp_body) = resp.into_parts();
    let resp_bytes = resp_body
        .collect()
        .await
        .map(|c| c.to_bytes())
        .unwrap_or_default();

    let end_time = Utc::now();

    // Parse and emit trace event in background
    if let Some(collector) = &collector {
        if let Some(event) = build_trace_event(
            &req_json,
            &resp_bytes,
            &path,
            &app_name,
            start_time,
            end_time,
            session_id,
        ) {
            debug!(model = %event.model, endpoint = %event.endpoint, app_name = %event.app_name, tokens_per_sec = ?event.tokens_per_sec, "queuing trace event");
            collector.send(event);
        }
    }

    let body = Body::from(resp_bytes);
    let response = Response::from_parts(resp_parts, body);
    Ok(response)
}

async fn handle_streaming_response(
    collector: Option<Arc<LangfuseCollector>>,
    resp: hyper::Response<hyper::body::Incoming>,
    req_json: Option<serde_json::Value>,
    app_name: String,
    path: String,
    start_time: chrono::DateTime<chrono::Utc>,
    session_id: Option<String>,
) -> Result<Response<Body>, StatusCode> {
    let (resp_parts, resp_body) = resp.into_parts();

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    // Spawn task to forward chunks and accumulate for tracing
    tokio::spawn(async move {
        let mut accumulated = Vec::<u8>::new();
        let mut stream = resp_body;
        let mut first_chunk_time: Option<chrono::DateTime<chrono::Utc>> = None;

        loop {
            match http_body_util::BodyExt::frame(&mut stream).await {
                Some(Ok(frame)) => {
                    if let Ok(data) = frame.into_data() {
                        if first_chunk_time.is_none() {
                            first_chunk_time = Some(Utc::now());
                        }
                        accumulated.extend_from_slice(&data);
                        if tx.send(Ok(data)).await.is_err() {
                            // Client disconnected
                            break;
                        }
                    }
                }
                Some(Err(e)) => {
                    error!("Error reading upstream stream: {}", e);
                    let _ = tx
                        .send(Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))
                        .await;
                    break;
                }
                None => break,
            }
        }

        // Stream done — emit trace
        let end_time = Utc::now();
        let ttft_ms = first_chunk_time
            .map(|t| (t - start_time).num_microseconds().unwrap_or(0) as f64 / 1000.0);
        if let Some(collector) = collector {
            let accumulated_bytes = Bytes::from(accumulated);
            if let Some(event) = build_trace_event_from_stream(
                &req_json,
                &accumulated_bytes,
                &path,
                &app_name,
                start_time,
                end_time,
                session_id,
                ttft_ms,
            ) {
                debug!(model = %event.model, endpoint = %event.endpoint, app_name = %event.app_name, session_id = ?event.session_id, tokens_per_sec = ?event.tokens_per_sec, ttft_ms = ?event.ttft_ms, "queuing trace event");
                collector.send(event);
            }
        }
    });

    let stream_body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    let response = Response::from_parts(resp_parts, stream_body);
    Ok(response)
}

fn build_trace_event(
    req_json: &Option<serde_json::Value>,
    resp_bytes: &Bytes,
    path: &str,
    app_name: &str,
    start_time: chrono::DateTime<chrono::Utc>,
    end_time: chrono::DateTime<chrono::Utc>,
    session_id: Option<String>,
) -> Option<LangfuseEvent> {
    let req_body = req_json.as_ref()?;
    let resp_json: serde_json::Value = serde_json::from_slice(resp_bytes).ok()?;

    let model = req_body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();

    let input = extract_input(req_body, path);
    let (output, prompt_tokens, completion_tokens, tokens_per_sec) = extract_output(&resp_json, path);

    Some(LangfuseEvent {
        trace_id: Uuid::new_v4().to_string(),
        generation_id: Uuid::new_v4().to_string(),
        app_name: app_name.to_string(),
        model,
        endpoint: path.to_string(),
        input,
        output,
        start_time,
        end_time,
        prompt_tokens,
        completion_tokens,
        tokens_per_sec,
        ttft_ms: None,
        session_id,
    })
}

fn build_trace_event_from_stream(
    req_json: &Option<serde_json::Value>,
    accumulated: &Bytes,
    path: &str,
    app_name: &str,
    start_time: chrono::DateTime<chrono::Utc>,
    end_time: chrono::DateTime<chrono::Utc>,
    session_id: Option<String>,
    ttft_ms: Option<f64>,
) -> Option<LangfuseEvent> {
    let req_body = req_json.as_ref()?;

    let model = req_body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();

    let input = extract_input(req_body, path);

    // Parse accumulated NDJSON lines
    let text = std::str::from_utf8(accumulated).ok()?;
    let chunks: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    if chunks.is_empty() {
        return None;
    }

    // Find the done chunk
    let done_chunk = chunks.iter().find(|c| c.get("done").and_then(|d| d.as_bool()).unwrap_or(false));

    let (mut output_text, prompt_tokens, completion_tokens, tokens_per_sec) = if let Some(done) = done_chunk {
        let prompt_tokens = done.get("prompt_eval_count").and_then(|v| v.as_u64());
        let completion_tokens = done.get("eval_count").and_then(|v| v.as_u64());
        let eval_duration_ns = done.get("eval_duration").and_then(|v| v.as_u64());
        let tokens_per_sec = completion_tokens.zip(eval_duration_ns).and_then(|(tokens, ns)| {
            if ns == 0 { None } else { Some(tokens as f64 / (ns as f64 / 1_000_000_000.0)) }
        });
        (String::new(), prompt_tokens, completion_tokens, tokens_per_sec)
    } else {
        (String::new(), None, None, None)
    };

    // Concatenate all content from message chunks
    for chunk in &chunks {
        if path == "/api/chat" {
            if let Some(msg) = chunk.get("message") {
                if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                    output_text.push_str(content);
                }
            }
        } else if path == "/api/generate" {
            if let Some(response) = chunk.get("response").and_then(|r| r.as_str()) {
                output_text.push_str(response);
            }
        }
    }

    Some(LangfuseEvent {
        trace_id: Uuid::new_v4().to_string(),
        generation_id: Uuid::new_v4().to_string(),
        app_name: app_name.to_string(),
        model,
        endpoint: path.to_string(),
        input,
        output: serde_json::Value::String(output_text),
        start_time,
        end_time,
        prompt_tokens,
        completion_tokens,
        tokens_per_sec,
        ttft_ms,
        session_id,
    })
}

fn extract_input(req_body: &serde_json::Value, path: &str) -> serde_json::Value {
    match path {
        "/api/chat" => req_body
            .get("messages")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        "/api/generate" => req_body
            .get("prompt")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        _ => req_body
            .get("input")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    }
}

fn extract_output(
    resp_json: &serde_json::Value,
    path: &str,
) -> (serde_json::Value, Option<u64>, Option<u64>, Option<f64>) {
    let prompt_tokens = resp_json.get("prompt_eval_count").and_then(|v| v.as_u64());
    let completion_tokens = resp_json.get("eval_count").and_then(|v| v.as_u64());
    let eval_duration_ns = resp_json.get("eval_duration").and_then(|v| v.as_u64());
    let tokens_per_sec = completion_tokens.zip(eval_duration_ns).and_then(|(tokens, ns)| {
        if ns == 0 { None } else { Some(tokens as f64 / (ns as f64 / 1_000_000_000.0)) }
    });

    let output = match path {
        "/api/chat" => resp_json
            .get("message")
            .and_then(|m| m.get("content"))
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        "/api/generate" => resp_json
            .get("response")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        "/api/embed" | "/api/embeddings" => {
            serde_json::Value::String("[embedding vector]".to_string())
        }
        _ => serde_json::Value::Null,
    };

    (output, prompt_tokens, completion_tokens, tokens_per_sec)
}

fn build_upstream_request(
    upstream_url: &str,
    req: Request<Body>,
) -> Result<hyper::Request<Body>, Box<dyn std::error::Error + Send + Sync>> {
    let (parts, body) = req.into_parts();

    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");

    let upstream_uri = format!("{}{}", upstream_url.trim_end_matches('/'), path_and_query);
    let uri: Uri = upstream_uri.parse()?;

    let mut builder = hyper::Request::builder()
        .method(parts.method)
        .uri(uri);

    // Forward headers, skipping hop-by-hop headers
    // Strip hop-by-hop headers and browser-side headers that confuse Ollama's
    // CORS check (Origin/Referer cause 403 when the client is on a remote host).
    let skip = ["host", "connection", "transfer-encoding", "authorization", "origin", "referer"];
    for (name, value) in &parts.headers {
        if !skip.contains(&name.as_str()) {
            builder = builder.header(name, value);
        }
    }

    Ok(builder.body(body)?)
}

fn convert_response(resp: hyper::Response<hyper::body::Incoming>) -> Response<Body> {
    let (parts, body) = resp.into_parts();
    let body = Body::new(body.map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    }));
    Response::from_parts(parts, body)
}
