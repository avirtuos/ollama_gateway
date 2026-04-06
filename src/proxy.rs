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
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use axum::response::IntoResponse;

use crate::{
    auth::{AppName, BearerToken},
    config::{BackendConfig, BackendType, Config},
    connection_id::ConnectionId,
    langfuse::{LangfuseCollector, LangfuseEvent},
    metrics::{MetricsCollector, MetricsRecord},
    processors::ProcessorRegistry,
    registry::ModelRegistry,
    state::AppState,
};

/// Endpoints that get Langfuse tracing.
const TRACED_PATHS: &[&str] = &[
    "/api/chat", "/api/generate", "/api/embed", "/api/embeddings",
    "/v1/chat/completions", "/v1/completions", "/v1/embeddings",
];

/// How to route a request path.
enum RouteCategory {
    /// Extract model from body → route to single backend.
    Inference,
    /// `/api/show` — extract model or name from body → single backend.
    ModelInfo,
    /// `/api/tags`, `/v1/models` — fan out to all healthy backends, merge.
    ModelList,
    /// `/api/ps` — fan out to all Ollama backends, merge.
    RunningModels,
    /// `/api/pull` etc. — try model lookup; fall back to default.
    OtherWithModel,
    /// Everything else — route to default backend.
    Passthrough,
}

fn classify_path(path: &str) -> RouteCategory {
    // Strip query string for matching
    let p = path.split('?').next().unwrap_or(path);
    match p {
        "/api/chat" | "/api/generate" | "/api/embed" | "/api/embeddings"
        | "/v1/chat/completions" | "/v1/completions" | "/v1/embeddings" => RouteCategory::Inference,
        "/api/show" => RouteCategory::ModelInfo,
        "/api/tags" | "/v1/models" => RouteCategory::ModelList,
        "/api/ps" => RouteCategory::RunningModels,
        "/api/pull" | "/api/push" | "/api/copy" | "/api/delete" | "/api/create" => {
            RouteCategory::OtherWithModel
        }
        _ => RouteCategory::Passthrough,
    }
}

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
    let app_name = req
        .extensions()
        .get::<AppName>()
        .map(|a| a.0.clone())
        .unwrap_or_default();
    let session_id = req
        .extensions()
        .get::<ConnectionId>()
        .map(|c| c.0.clone())
        .unwrap_or_default();

    let privacy_mode = *state.privacy_mode.read().await;

    let token_field: Option<String> = if !privacy_mode {
        req.extensions().get::<BearerToken>().map(|t| t.0.clone())
    } else {
        None
    };

    match &token_field {
        Some(token) => info!(method = %method, url = %url, remote = %remote, session = %session_id, app = %app_name, token = %token, "→"),
        None => info!(method = %method, url = %url, remote = %remote, session = %session_id, app = %app_name, "→"),
    }

    let path = req.uri().path().to_string();
    let result = dispatch(state, req, &path, &app_name, &session_id, privacy_mode).await;

    match &result {
        Ok(resp) => match &token_field {
            Some(token) => info!(method = %method, url = %url, remote = %remote, session = %session_id, app = %app_name, token = %token, status = %resp.status(), "←"),
            None => info!(method = %method, url = %url, remote = %remote, session = %session_id, app = %app_name, status = %resp.status(), "←"),
        },
        Err(status) => match &token_field {
            Some(token) => info!(method = %method, url = %url, remote = %remote, session = %session_id, app = %app_name, token = %token, status = %status, "←"),
            None => info!(method = %method, url = %url, remote = %remote, session = %session_id, app = %app_name, status = %status, "←"),
        },
    }

    result
}

async fn dispatch(
    state: Arc<AppState>,
    req: Request<Body>,
    path: &str,
    app_name: &str,
    session_id: &str,
    privacy_mode: bool,
) -> Result<Response<Body>, StatusCode> {
    let registry = state.model_registry.read().await.clone();

    match classify_path(path) {
        RouteCategory::ModelList => proxy_aggregate_models(&state, &registry, path).await,
        RouteCategory::RunningModels => proxy_aggregate_ps(&state, &registry).await,

        category @ (RouteCategory::Inference | RouteCategory::ModelInfo | RouteCategory::OtherWithModel) => {
            // Buffer body to extract model name
            let (parts, body) = req.into_parts();
            let req_bytes = body.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
            let mut req_json: Option<serde_json::Value> = serde_json::from_slice(&req_bytes).ok();

            // Inject stream_options.include_usage for /v1/ streaming so the backend
            // includes a usage chunk in the final SSE frame, which extract_streaming_stats() can parse.
            let req_bytes = if path.starts_with("/v1/")
                && req_json.as_ref().and_then(|j| j.get("stream")).and_then(|v| v.as_bool()).unwrap_or(false)
            {
                if let Some(ref mut json) = req_json {
                    json["stream_options"] = serde_json::json!({"include_usage": true});
                    serde_json::to_vec(json).map(bytes::Bytes::from).unwrap_or(req_bytes)
                } else {
                    req_bytes
                }
            } else {
                req_bytes
            };

            // Try both "model" and "name" keys
            let model_name = req_json.as_ref().and_then(|j| {
                j.get("model").or_else(|| j.get("name")).and_then(|v| v.as_str()).map(|s| s.to_string())
            });

            let resolved = model_name.as_deref().and_then(|m| registry.resolve_backend(m));

            // Warn if inference request's model isn't in registry
            if matches!(category, RouteCategory::Inference) {
                if let Some(ref m) = model_name {
                    if resolved.is_none() {
                        warn!(model = %m, "Model not found in registry, routing to default backend");
                    }
                }
            }

            let backend = match resolved.or_else(|| registry.default_backend()) {
                Some(b) => b.clone(),
                None => {
                    let model_str = model_name.as_deref().unwrap_or("unknown");
                    warn!(model = model_str, "No healthy backend found");
                    return Err(StatusCode::BAD_GATEWAY);
                }
            };

            // Resolve pre/post processors for this model + backend pair
            let processor_rules = state.processor_rules.read().await.clone();
            let model_str = model_name.as_deref().unwrap_or("");
            let (pre_ids, post_ids) = Config::resolve_processors(
                &processor_rules, model_str, &backend.name,
            );

            // Apply preprocessors to the request body
            let req_bytes = if !pre_ids.is_empty() {
                if let Some(ref mut json) = req_json {
                    state.processor_registry.apply_preprocessors(&pre_ids, json);
                    debug!(model = model_str, processors = ?pre_ids, "Applied preprocessors");
                    serde_json::to_vec(json).map(bytes::Bytes::from).unwrap_or(req_bytes)
                } else {
                    req_bytes
                }
            } else {
                req_bytes
            };

            // Debug: log full request body including tools array
            if let Some(ref json) = req_json {
                let has_tools = json.get("tools").and_then(|t| t.as_array()).map(|a| a.len());
                debug!(
                    model = model_str,
                    backend = %backend.name,
                    has_tools = ?has_tools,
                    body = %json,
                    "Full request body"
                );
            }

            let collector = state.langfuse_collector.read().await.clone();
            let on_traced_path = TRACED_PATHS.iter().any(|p| path == *p || path.starts_with(p));
            let log_content = !privacy_mode && on_traced_path;

            let rebuilt = Request::from_parts(parts, Body::from(req_bytes));
            if on_traced_path {
                proxy_with_tracing_buffered(
                    state, collector, rebuilt, req_json,
                    &backend, log_content, app_name.to_string(),
                    Some(session_id.to_string()),
                    post_ids,
                ).await
            } else {
                proxy_single_with_metrics(state, rebuilt, &backend, path, post_ids).await
            }
        }

        RouteCategory::Passthrough => {
            let backend = match registry.default_backend() {
                Some(b) => b.clone(),
                None => {
                    error!("No healthy backend available for passthrough");
                    return Err(StatusCode::BAD_GATEWAY);
                }
            };
            proxy_single_with_metrics(state, req, &backend, path, vec![]).await
        }
    }
}

/// Route a single request and record latency/status metrics (no token data).
async fn proxy_single_with_metrics(
    state: Arc<AppState>,
    req: Request<Body>,
    backend: &BackendConfig,
    path: &str,
    post_ids: Vec<String>,
) -> Result<Response<Body>, StatusCode> {
    let start_time = Utc::now();
    let upstream_req = build_upstream_request(&backend.url, req).map_err(|e| {
        error!("Failed to build upstream request: {}", e);
        StatusCode::BAD_GATEWAY
    })?;

    let resp = state.http_client.request(upstream_req).await.map_err(|e| {
        error!("Upstream request failed: {}", e);
        StatusCode::BAD_GATEWAY
    })?;

    let end_time = Utc::now();
    let status_code = resp.status().as_u16();
    let latency_ms = (end_time - start_time).num_microseconds().unwrap_or(0) as f64 / 1000.0;

    state.metrics_collector.record(MetricsRecord {
        timestamp: start_time.to_rfc3339(),
        backend_name: backend.name.clone(),
        model: String::new(),
        endpoint: path.to_string(),
        prompt_tokens: None,
        completion_tokens: None,
        tokens_per_sec: None,
        ttft_ms: None,
        latency_ms,
        status_code,
    });

    if !post_ids.is_empty() {
        apply_postprocessors_to_response(resp, &state.processor_registry, &post_ids).await
    } else {
        Ok(convert_response(resp))
    }
}

/// Buffer a response, apply postprocessors, and return the modified response.
async fn apply_postprocessors_to_response(
    resp: hyper::Response<hyper::body::Incoming>,
    registry: &Arc<ProcessorRegistry>,
    post_ids: &[String],
) -> Result<Response<Body>, StatusCode> {
    let (parts, body) = resp.into_parts();
    let resp_bytes = body.collect().await.map(|c| c.to_bytes()).unwrap_or_default();

    let parse_result = serde_json::from_slice::<serde_json::Value>(&resp_bytes);

    // If JSON parsing fails, try raw-text repair (e.g. stripping Gemma 4
    // special tokens that break JSON structure) then re-parse.
    let mut json = match parse_result {
        Ok(v) => v,
        Err(_) => {
            let raw_str = String::from_utf8_lossy(&resp_bytes);
            if let Some(repaired) = registry.try_repair_raw(post_ids, &raw_str) {
                match serde_json::from_str::<serde_json::Value>(&repaired) {
                    Ok(v) => v,
                    Err(_) => {
                        warn!("Postprocessor raw repair produced non-JSON, passing through");
                        return Ok(Response::from_parts(parts, Body::from(resp_bytes)));
                    }
                }
            } else {
                return Ok(Response::from_parts(parts, Body::from(resp_bytes)));
            }
        }
    };

    registry.apply_postprocessors(post_ids, &mut json);
    let new_bytes = serde_json::to_vec(&json).unwrap_or_else(|_| resp_bytes.to_vec());
    Ok(Response::from_parts(parts, Body::from(new_bytes)))
}

/// Fan out to all healthy backends, aggregate their model lists.
async fn proxy_aggregate_models(
    state: &Arc<AppState>,
    registry: &Arc<ModelRegistry>,
    path: &str,
) -> Result<Response<Body>, StatusCode> {
    let healthy = registry.all_healthy_backends();
    if healthy.is_empty() {
        return Ok(axum::Json(json!({ "error": "no healthy backends available" })).into_response());
    }

    // Collect model metadata from each backend
    let mut tasks = tokio::task::JoinSet::new();
    for backend in healthy {
        let url = backend.config.url.clone();
        let bt = backend.config.backend_type.clone();
        let client = state.http_client.clone();
        tasks.spawn(async move {
            let target = match bt {
                BackendType::Ollama => format!("{}/api/tags", url.trim_end_matches('/')),
                BackendType::Llamacpp => format!("{}/v1/models", url.trim_end_matches('/')),
            };
            let uri: Uri = target.parse().ok()?;
            let req = hyper::Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .ok()?;
            let resp = client.request(req).await.ok()?;
            let (parts, body) = resp.into_parts();
            if !parts.status.is_success() { return None; }
            let bytes = body.collect().await.ok()?.to_bytes();
            let json: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
            Some((bt, json))
        });
    }

    // Merge results — deduplicate by name, first seen wins (backends are priority-sorted)
    let mut seen = std::collections::HashSet::new();
    let mut merged: Vec<serde_json::Value> = Vec::new();

    while let Some(join_result) = tasks.join_next().await {
        let Some(Some((bt, json))) = join_result.ok() else { continue };
        let models_arr = match bt {
            BackendType::Ollama => json["models"].as_array().cloned().unwrap_or_default(),
            BackendType::Llamacpp => {
                // Normalize {data:[{id}]} → [{name}]
                json["data"].as_array().map(|arr| {
                    arr.iter().map(|m| json!({ "name": m["id"] })).collect()
                }).unwrap_or_default()
            }
        };
        for model in models_arr {
            if let Some(name) = model.get("name").and_then(|n| n.as_str()) {
                if seen.insert(name.to_string()) {
                    merged.push(model);
                }
            }
        }
    }

    // For /v1/models return OpenAI format; for /api/tags return Ollama format
    let resp_json = if path == "/v1/models" {
        let data: Vec<serde_json::Value> = merged.iter().map(|m| {
            let name = m.get("name").and_then(|n| n.as_str()).unwrap_or("");
            json!({ "id": name, "object": "model" })
        }).collect();
        json!({ "object": "list", "data": data })
    } else {
        json!({ "models": merged })
    };

    Ok(axum::Json(resp_json).into_response())
}

/// Fan out `/api/ps` to all healthy Ollama backends, merge running model lists.
async fn proxy_aggregate_ps(
    state: &Arc<AppState>,
    registry: &Arc<ModelRegistry>,
) -> Result<Response<Body>, StatusCode> {
    let ollama_backends = registry.all_healthy_ollama_backends();
    if ollama_backends.is_empty() {
        return Ok(axum::Json(json!({ "models": [] })).into_response());
    }

    let mut tasks = tokio::task::JoinSet::new();
    for backend in ollama_backends {
        let url = format!("{}/api/ps", backend.config.url.trim_end_matches('/'));
        let client = state.http_client.clone();
        tasks.spawn(async move {
            let uri: Uri = url.parse().ok()?;
            let req = hyper::Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .ok()?;
            let resp = client.request(req).await.ok()?;
            let (parts, body) = resp.into_parts();
            if !parts.status.is_success() { return None; }
            let bytes = body.collect().await.ok()?.to_bytes();
            let json: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
            json["models"].as_array().cloned()
        });
    }

    let mut merged: Vec<serde_json::Value> = Vec::new();
    while let Some(join_result) = tasks.join_next().await {
        if let Ok(Some(models)) = join_result {
            merged.extend(models);
        }
    }

    Ok(axum::Json(json!({ "models": merged })).into_response())
}

/// Proxy with body already buffered; handles tracing and content logging.
#[allow(clippy::too_many_arguments)]
async fn proxy_with_tracing_buffered(
    state: Arc<AppState>,
    collector: Option<Arc<LangfuseCollector>>,
    req: Request<Body>,
    req_json: Option<serde_json::Value>,
    backend: &BackendConfig,
    log_content: bool,
    app_name: String,
    session_id: Option<String>,
    post_ids: Vec<String>,
) -> Result<Response<Body>, StatusCode> {
    let path = req.uri().path().to_string();
    let start_time = Utc::now();
    let backend_name = backend.name.clone();

    // OpenAI-compatible paths default stream=false; native Ollama paths default stream=true
    let stream_default = !path.starts_with("/v1/");
    let streaming = req_json.as_ref().map(|body| {
        body.get("stream").and_then(|v| v.as_bool()).unwrap_or(stream_default)
    }).unwrap_or(stream_default);

    let upstream_req = build_upstream_request(&backend.url, req).map_err(|e| {
        error!("Failed to build upstream request: {}", e);
        StatusCode::BAD_GATEWAY
    })?;

    let resp = state.http_client.request(upstream_req).await.map_err(|e| {
        error!("Upstream request failed: {}", e);
        StatusCode::BAD_GATEWAY
    })?;

    let metrics = state.metrics_collector.clone();
    let proc_registry = state.processor_registry.clone();
    if streaming {
        handle_streaming_response(collector, metrics, resp, req_json, app_name, path, start_time, session_id, log_content, Some(backend_name), proc_registry, post_ids).await
    } else {
        handle_non_streaming_response(collector, metrics, resp, req_json, app_name, path, start_time, session_id, log_content, Some(backend_name), proc_registry, post_ids).await
    }
}

async fn handle_non_streaming_response(
    collector: Option<Arc<LangfuseCollector>>,
    metrics: Arc<MetricsCollector>,
    resp: hyper::Response<hyper::body::Incoming>,
    req_json: Option<serde_json::Value>,
    app_name: String,
    path: String,
    start_time: chrono::DateTime<chrono::Utc>,
    session_id: Option<String>,
    log_content: bool,
    backend_name: Option<String>,
    proc_registry: Arc<ProcessorRegistry>,
    post_ids: Vec<String>,
) -> Result<Response<Body>, StatusCode> {
    let (resp_parts, resp_body) = resp.into_parts();
    let status_code = resp_parts.status.as_u16();
    let resp_bytes = resp_body
        .collect()
        .await
        .map(|c| c.to_bytes())
        .unwrap_or_default();

    // Apply postprocessors to the buffered response
    let resp_bytes = if !post_ids.is_empty() {
        let parse_result = serde_json::from_slice::<serde_json::Value>(&resp_bytes);
        match parse_result {
            Ok(mut json) => {
                proc_registry.apply_postprocessors(&post_ids, &mut json);
                serde_json::to_vec(&json).map(bytes::Bytes::from).unwrap_or(resp_bytes)
            }
            Err(_) => {
                // JSON parse failed — try raw-text repair then re-parse
                let raw_str = String::from_utf8_lossy(&resp_bytes);
                if let Some(repaired) = proc_registry.try_repair_raw(&post_ids, &raw_str) {
                    if let Ok(mut json) = serde_json::from_str::<serde_json::Value>(&repaired) {
                        proc_registry.apply_postprocessors(&post_ids, &mut json);
                        serde_json::to_vec(&json).map(bytes::Bytes::from).unwrap_or(resp_bytes)
                    } else {
                        warn!("Postprocessor raw repair produced non-JSON, passing through");
                        resp_bytes
                    }
                } else {
                    resp_bytes
                }
            }
        }
    } else {
        resp_bytes
    };

    let end_time = Utc::now();

    // Debug: log full response body including tool_calls
    if let Ok(resp_json) = serde_json::from_slice::<serde_json::Value>(&resp_bytes) {
        debug!(
            endpoint = %path,
            status = status_code,
            body = %resp_json,
            "Full response body"
        );
    }

    if log_content {
        if let Some(req_body) = &req_json {
            if let Ok(resp_json) = serde_json::from_slice::<serde_json::Value>(&resp_bytes) {
                let model = req_body.get("model").and_then(|m| m.as_str()).unwrap_or("unknown");
                let input = extract_input(req_body, &path);
                let (output, _, _, _) = extract_output(&resp_json, &path);
                info!(model, session = ?session_id, input = %input, output = %output, "chat");
            }
        }
    }

    // Save backend name before it is potentially moved into build_trace_event
    let backend_for_metrics = backend_name.as_deref().unwrap_or("").to_string();

    if let Some(collector) = &collector {
        if let Some(event) = build_trace_event(
            &req_json,
            &resp_bytes,
            &path,
            &app_name,
            start_time,
            end_time,
            session_id,
            backend_name,
            log_content,
        ) {
            debug!(model = %event.model, endpoint = %event.endpoint, app_name = %event.app_name, session_id = ?event.session_id, tokens_per_sec = ?event.tokens_per_sec, "queuing trace event");
            collector.send(event);
        }
    }

    // Always record metrics
    let latency_ms = (end_time - start_time).num_microseconds().unwrap_or(0) as f64 / 1000.0;
    let model = req_json.as_ref()
        .and_then(|j| j.get("model").and_then(|v| v.as_str()))
        .unwrap_or("unknown")
        .to_string();
    let (_, pt, ct, tps) = serde_json::from_slice::<serde_json::Value>(&resp_bytes)
        .ok()
        .map(|j| extract_output(&j, &path))
        .unwrap_or_default();
    metrics.record(MetricsRecord {
        timestamp: start_time.to_rfc3339(),
        backend_name: backend_for_metrics,
        model,
        endpoint: path,
        prompt_tokens: pt,
        completion_tokens: ct,
        tokens_per_sec: tps,
        ttft_ms: None,
        latency_ms,
        status_code,
    });

    let body = Body::from(resp_bytes);
    let response = Response::from_parts(resp_parts, body);
    Ok(response)
}

async fn handle_streaming_response(
    collector: Option<Arc<LangfuseCollector>>,
    metrics: Arc<MetricsCollector>,
    resp: hyper::Response<hyper::body::Incoming>,
    req_json: Option<serde_json::Value>,
    app_name: String,
    path: String,
    start_time: chrono::DateTime<chrono::Utc>,
    session_id: Option<String>,
    log_content: bool,
    backend_name: Option<String>,
    proc_registry: Arc<ProcessorRegistry>,
    post_ids: Vec<String>,
) -> Result<Response<Body>, StatusCode> {
    let (resp_parts, resp_body) = resp.into_parts();
    let status_code = resp_parts.status.as_u16();

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let has_postprocessors = !post_ids.is_empty();
    let is_sse = path.starts_with("/v1/");

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

                        // Apply postprocessors to each streaming chunk
                        let data = if has_postprocessors {
                            process_streaming_chunk(&data, &proc_registry, &post_ids, is_sse)
                        } else {
                            data
                        };

                        accumulated.extend_from_slice(&data);
                        if tx.send(Ok(data)).await.is_err() {
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

        let end_time = Utc::now();
        let ttft_ms = first_chunk_time
            .map(|t| (t - start_time).num_microseconds().unwrap_or(0) as f64 / 1000.0);
        let accumulated_bytes = Bytes::from(accumulated);

        // Debug: log full accumulated streaming response
        if let Ok(text) = std::str::from_utf8(&accumulated_bytes) {
            debug!(
                endpoint = %path,
                status = status_code,
                body = %text,
                "Full streaming response (accumulated)"
            );
        }

        if log_content {
            if let Some(req_body) = &req_json {
                let model = req_body.get("model").and_then(|m| m.as_str()).unwrap_or("unknown");
                let input = extract_input(req_body, &path);
                let output_text = extract_streaming_output_text(&accumulated_bytes, &path);
                info!(model, session = ?session_id, input = %input, output = %output_text, "chat");
            }
        }

        // Save for metrics before backend_name is potentially moved
        let backend_for_metrics = backend_name.as_deref().unwrap_or("").to_string();
        let model_for_metrics = req_json.as_ref()
            .and_then(|j| j.get("model").and_then(|v| v.as_str()))
            .unwrap_or("unknown")
            .to_string();

        if let Some(collector) = collector {
            if let Some(event) = build_trace_event_from_stream(
                &req_json,
                &accumulated_bytes,
                &path,
                &app_name,
                start_time,
                end_time,
                session_id,
                ttft_ms,
                backend_name,
                log_content,
            ) {
                debug!(model = %event.model, endpoint = %event.endpoint, app_name = %event.app_name, session_id = ?event.session_id, tokens_per_sec = ?event.tokens_per_sec, ttft_ms = ?event.ttft_ms, "queuing trace event");
                collector.send(event);
            }
        }

        // Always record metrics
        let latency_ms = (end_time - start_time).num_microseconds().unwrap_or(0) as f64 / 1000.0;
        let (pt, ct, tps) = extract_streaming_stats(&accumulated_bytes, &path);
        metrics.record(MetricsRecord {
            timestamp: start_time.to_rfc3339(),
            backend_name: backend_for_metrics,
            model: model_for_metrics,
            endpoint: path,
            prompt_tokens: pt,
            completion_tokens: ct,
            tokens_per_sec: tps,
            ttft_ms,
            latency_ms,
            status_code,
        });
    });

    let stream_body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    let response = Response::from_parts(resp_parts, stream_body);
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
fn build_trace_event(
    req_json: &Option<serde_json::Value>,
    resp_bytes: &Bytes,
    path: &str,
    app_name: &str,
    start_time: chrono::DateTime<chrono::Utc>,
    end_time: chrono::DateTime<chrono::Utc>,
    session_id: Option<String>,
    backend_name: Option<String>,
    log_content: bool,
) -> Option<LangfuseEvent> {
    let req_body = req_json.as_ref()?;
    let resp_json: serde_json::Value = serde_json::from_slice(resp_bytes).ok()?;

    let model = req_body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();

    let (input, output) = if log_content {
        let input = extract_input(req_body, path);
        let (output, ..) = extract_output(&resp_json, path);
        (input, output)
    } else {
        (serde_json::json!("[redacted]"), serde_json::json!("[redacted]"))
    };
    let (_, prompt_tokens, completion_tokens, tokens_per_sec) = extract_output(&resp_json, path);

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
        backend_name,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_trace_event_from_stream(
    req_json: &Option<serde_json::Value>,
    accumulated: &Bytes,
    path: &str,
    app_name: &str,
    start_time: chrono::DateTime<chrono::Utc>,
    end_time: chrono::DateTime<chrono::Utc>,
    session_id: Option<String>,
    ttft_ms: Option<f64>,
    backend_name: Option<String>,
    log_content: bool,
) -> Option<LangfuseEvent> {
    let req_body = req_json.as_ref()?;

    let model = req_body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();

    let input = if log_content { extract_input(req_body, path) } else { serde_json::json!("[redacted]") };

    let text = std::str::from_utf8(accumulated).ok()?;
    let chunks = parse_streaming_chunks(text, path);

    if chunks.is_empty() {
        return None;
    }

    let (prompt_tokens, completion_tokens, tokens_per_sec) = if path.starts_with("/v1/") {
        let usage = chunks.iter().find_map(|c| c.get("usage"));
        let prompt_tokens = usage.and_then(|u| u.get("prompt_tokens")).and_then(|v| v.as_u64());
        let completion_tokens = usage.and_then(|u| u.get("completion_tokens")).and_then(|v| v.as_u64());
        (prompt_tokens, completion_tokens, None)
    } else {
        let done = chunks.iter().find(|c| c.get("done").and_then(|d| d.as_bool()).unwrap_or(false));
        let prompt_tokens = done.and_then(|d| d.get("prompt_eval_count")).and_then(|v| v.as_u64());
        let completion_tokens = done.and_then(|d| d.get("eval_count")).and_then(|v| v.as_u64());
        let eval_duration_ns = done.and_then(|d| d.get("eval_duration")).and_then(|v| v.as_u64());
        let tokens_per_sec = completion_tokens.zip(eval_duration_ns).and_then(|(tokens, ns)| {
            if ns == 0 { None } else { Some(tokens as f64 / (ns as f64 / 1_000_000_000.0)) }
        });
        (prompt_tokens, completion_tokens, tokens_per_sec)
    };

    let mut output_text = String::new();
    for chunk in &chunks {
        match path {
            "/api/chat" => {
                if let Some(content) = chunk.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()) {
                    output_text.push_str(content);
                }
            }
            "/api/generate" => {
                if let Some(response) = chunk.get("response").and_then(|r| r.as_str()) {
                    output_text.push_str(response);
                }
            }
            "/v1/chat/completions" => {
                if let Some(content) = chunk.get("choices").and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    output_text.push_str(content);
                }
            }
            "/v1/completions" => {
                if let Some(text) = chunk.get("choices").and_then(|c| c.get(0))
                    .and_then(|c| c.get("text"))
                    .and_then(|t| t.as_str())
                {
                    output_text.push_str(text);
                }
            }
            _ => {}
        }
    }

    Some(LangfuseEvent {
        trace_id: Uuid::new_v4().to_string(),
        generation_id: Uuid::new_v4().to_string(),
        app_name: app_name.to_string(),
        model,
        endpoint: path.to_string(),
        input,
        output: if log_content { serde_json::Value::String(output_text) } else { serde_json::json!("[redacted]") },
        start_time,
        end_time,
        prompt_tokens,
        completion_tokens,
        tokens_per_sec,
        ttft_ms,
        session_id,
        backend_name,
    })
}

/// Parse streaming chunks from either NDJSON (native Ollama) or SSE (OpenAI /v1/ paths).
fn parse_streaming_chunks(text: &str, path: &str) -> Vec<serde_json::Value> {
    if path.starts_with("/v1/") {
        text.lines()
            .filter_map(|line| {
                let payload = line.strip_prefix("data: ")?;
                if payload == "[DONE]" { return None; }
                serde_json::from_str(payload).ok()
            })
            .collect()
    } else {
        text.lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }
}

fn extract_streaming_output_text(accumulated: &Bytes, path: &str) -> String {
    let text = match std::str::from_utf8(accumulated) {
        Ok(t) => t,
        Err(_) => return String::new(),
    };
    let mut output = String::new();
    for chunk in parse_streaming_chunks(text, path) {
        match path {
            "/api/chat" => {
                if let Some(content) = chunk.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()) {
                    output.push_str(content);
                }
            }
            "/api/generate" => {
                if let Some(response) = chunk.get("response").and_then(|r| r.as_str()) {
                    output.push_str(response);
                }
            }
            "/v1/chat/completions" => {
                if let Some(content) = chunk.get("choices").and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    output.push_str(content);
                }
            }
            "/v1/completions" => {
                if let Some(text) = chunk.get("choices").and_then(|c| c.get(0))
                    .and_then(|c| c.get("text"))
                    .and_then(|t| t.as_str())
                {
                    output.push_str(text);
                }
            }
            _ => {}
        }
    }
    output
}

/// Extract token counts and tok/s from accumulated streaming bytes.
fn extract_streaming_stats(accumulated: &Bytes, path: &str) -> (Option<u64>, Option<u64>, Option<f64>) {
    let text = match std::str::from_utf8(accumulated) {
        Ok(t) => t,
        Err(_) => return (None, None, None),
    };
    let chunks = parse_streaming_chunks(text, path);
    if path.starts_with("/v1/") {
        let usage = chunks.iter().find_map(|c| c.get("usage"));
        let pt = usage.and_then(|u| u.get("prompt_tokens")).and_then(|v| v.as_u64());
        let ct = usage.and_then(|u| u.get("completion_tokens")).and_then(|v| v.as_u64());
        (pt, ct, None)
    } else {
        let done = chunks.iter().find(|c| c.get("done").and_then(|d| d.as_bool()).unwrap_or(false));
        let pt = done.and_then(|d| d.get("prompt_eval_count")).and_then(|v| v.as_u64());
        let ct = done.and_then(|d| d.get("eval_count")).and_then(|v| v.as_u64());
        let ns = done.and_then(|d| d.get("eval_duration")).and_then(|v| v.as_u64());
        let tps = ct.zip(ns).and_then(|(t, n)| {
            if n == 0 { None } else { Some(t as f64 / (n as f64 / 1_000_000_000.0)) }
        });
        (pt, ct, tps)
    }
}

fn extract_input(req_body: &serde_json::Value, path: &str) -> serde_json::Value {
    match path {
        "/api/chat" | "/v1/chat/completions" => req_body
            .get("messages")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        "/api/generate" | "/v1/completions" => req_body
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
    match path {
        "/api/chat" | "/api/generate" => {
            let prompt_tokens = resp_json.get("prompt_eval_count").and_then(|v| v.as_u64());
            let completion_tokens = resp_json.get("eval_count").and_then(|v| v.as_u64());
            let eval_duration_ns = resp_json.get("eval_duration").and_then(|v| v.as_u64());
            let tokens_per_sec = completion_tokens.zip(eval_duration_ns).and_then(|(tokens, ns)| {
                if ns == 0 { None } else { Some(tokens as f64 / (ns as f64 / 1_000_000_000.0)) }
            });
            let output = if path == "/api/chat" {
                resp_json.get("message").and_then(|m| m.get("content")).cloned()
            } else {
                resp_json.get("response").cloned()
            }.unwrap_or(serde_json::Value::Null);
            (output, prompt_tokens, completion_tokens, tokens_per_sec)
        }
        "/api/embed" | "/api/embeddings" => {
            let prompt_tokens = resp_json.get("prompt_eval_count").and_then(|v| v.as_u64());
            (serde_json::Value::String("[embedding vector]".to_string()), prompt_tokens, None, None)
        }
        "/v1/embeddings" => {
            let prompt_tokens = resp_json.get("usage").and_then(|u| u.get("prompt_tokens")).and_then(|v| v.as_u64());
            (serde_json::Value::String("[embedding vector]".to_string()), prompt_tokens, None, None)
        }
        "/v1/chat/completions" => {
            let (prompt_tokens, completion_tokens) = openai_usage(resp_json);
            let output = resp_json
                .get("choices").and_then(|c| c.get(0))
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            (output, prompt_tokens, completion_tokens, None)
        }
        "/v1/completions" => {
            let (prompt_tokens, completion_tokens) = openai_usage(resp_json);
            let output = resp_json
                .get("choices").and_then(|c| c.get(0))
                .and_then(|c| c.get("text"))
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            (output, prompt_tokens, completion_tokens, None)
        }
        _ => (serde_json::Value::Null, None, None, None),
    }
}

fn openai_usage(resp_json: &serde_json::Value) -> (Option<u64>, Option<u64>) {
    let usage = resp_json.get("usage");
    let prompt_tokens = usage.and_then(|u| u.get("prompt_tokens")).and_then(|v| v.as_u64());
    let completion_tokens = usage.and_then(|u| u.get("completion_tokens")).and_then(|v| v.as_u64());
    (prompt_tokens, completion_tokens)
}

/// Apply postprocessors to a raw streaming data chunk.
/// Each chunk may contain one or more JSON lines (NDJSON) or SSE `data:` lines.
fn process_streaming_chunk(
    data: &Bytes,
    registry: &Arc<ProcessorRegistry>,
    post_ids: &[String],
    is_sse: bool,
) -> Bytes {
    let text = match std::str::from_utf8(data) {
        Ok(t) => t,
        Err(_) => return data.clone(),
    };

    let mut output = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        if is_sse {
            if let Some(payload) = line.trim_end().strip_prefix("data: ") {
                if payload == "[DONE]" {
                    output.push_str(line);
                    continue;
                }
                let parsed = serde_json::from_str::<serde_json::Value>(payload);
                let mut json = match parsed {
                    Ok(v) => v,
                    Err(_) => {
                        // Chunk JSON broken — try raw repair (e.g. Gemma 4 token stripping)
                        if let Some(repaired) = registry.try_repair_raw(post_ids, payload) {
                            match serde_json::from_str::<serde_json::Value>(&repaired) {
                                Ok(v) => v,
                                Err(_) => {
                                    output.push_str(line);
                                    continue;
                                }
                            }
                        } else {
                            output.push_str(line);
                            continue;
                        }
                    }
                };
                registry.apply_chunk_postprocessors(post_ids, &mut json);
                output.push_str("data: ");
                output.push_str(&serde_json::to_string(&json).unwrap_or_else(|_| payload.to_string()));
                output.push('\n');
                continue;
            }
            output.push_str(line);
        } else {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                output.push_str(line);
                continue;
            }
            let parsed = serde_json::from_str::<serde_json::Value>(trimmed);
            let json_opt = match parsed {
                Ok(v) => Some(v),
                Err(_) => {
                    // Chunk JSON broken — try raw repair
                    if let Some(repaired) = registry.try_repair_raw(post_ids, trimmed) {
                        serde_json::from_str::<serde_json::Value>(&repaired).ok()
                    } else {
                        None
                    }
                }
            };
            if let Some(mut json) = json_opt {
                registry.apply_chunk_postprocessors(post_ids, &mut json);
                output.push_str(&serde_json::to_string(&json).unwrap_or_else(|_| trimmed.to_string()));
                if line.ends_with('\n') {
                    output.push('\n');
                }
            } else {
                output.push_str(line);
            }
        }
    }

    Bytes::from(output)
}

pub fn build_upstream_request(
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

    let skip = ["host", "connection", "transfer-encoding", "authorization", "origin", "referer", "content-length"];
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
