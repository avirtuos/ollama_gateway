use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path, Request, State},
    http::{StatusCode, Uri, header},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use http_body_util::BodyExt;
use base64::Engine;
use serde::Deserialize;
use serde_json::json;
use tracing::{error, info};

use crate::auth::AppName;
use crate::config::{Config, LangfuseConfig, OllamaConfig, TokenEntry};
use crate::langfuse::LangfuseCollector;
use crate::proxy::proxy_handler;
use crate::state::AppState;

static ADMIN_HTML: &str = include_str!("../assets/admin.html");

pub fn admin_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(admin_ui))
        .route("/api/langfuse", get(get_langfuse).put(put_langfuse))
        .route("/api/tokens", get(get_tokens).post(add_token))
        .route("/api/tokens/{token}", delete(delete_token))
        .route("/api/config", get(get_config).put(put_config))
        .route("/api/models", get(get_models))
        .route("/api/models/running", get(get_models_running))
        .route("/api/chat", post(admin_chat))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            basic_auth_middleware,
        ))
        .with_state(state)
}

pub async fn basic_auth_middleware(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok());

    let is_authorized = match auth_header {
        Some(h) if h.starts_with("Basic ") => {
            let encoded = &h["Basic ".len()..];
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .ok()
                .and_then(|b| String::from_utf8(b).ok())
                .map(|credentials| {
                    let expected = format!("admin:{}", state.admin_password);
                    credentials == expected
                })
                .unwrap_or(false)
        }
        _ => false,
    };

    if is_authorized {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(
                header::WWW_AUTHENTICATE,
                "Basic realm=\"Ollama Gateway Admin\"",
            )],
            Json(json!({ "error": "Unauthorized" })),
        )
            .into_response()
    }
}

async fn admin_ui() -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        ADMIN_HTML,
    )
        .into_response()
}

async fn get_langfuse(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let config = state.langfuse_config.read().await;
    Json(serde_json::to_value(&*config).unwrap_or_default())
}

#[derive(Deserialize)]
struct LangfuseUpdate {
    enabled: bool,
    host: String,
    public_key: String,
    secret_key: String,
    batch_size: Option<usize>,
    flush_interval_ms: Option<u64>,
}

async fn put_langfuse(
    State(state): State<Arc<AppState>>,
    Json(body): Json<LangfuseUpdate>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let new_config = LangfuseConfig {
        enabled: body.enabled,
        host: body.host,
        public_key: body.public_key,
        secret_key: body.secret_key,
        batch_size: body.batch_size.unwrap_or(10),
        flush_interval_ms: body.flush_interval_ms.unwrap_or(5000),
    };

    // Shutdown old collector
    {
        let collector_guard = state.langfuse_collector.read().await;
        if let Some(ref c) = *collector_guard {
            c.shutdown().await;
        }
    }

    // Build new collector (if enabled)
    let new_collector = if new_config.enabled {
        info!(host = %new_config.host, "Rebuilding Langfuse collector");
        let collector = LangfuseCollector::new(&new_config).await;
        Some(Arc::new(collector))
    } else {
        info!("Langfuse disabled, clearing collector");
        None
    };

    // Swap config and collector
    {
        let mut config_guard = state.langfuse_config.write().await;
        *config_guard = new_config;
    }
    {
        let mut collector_guard = state.langfuse_collector.write().await;
        *collector_guard = new_collector;
    }

    if let Err(e) = save_config_to_disk(&state).await {
        error!(error = %e, "Failed to save config to disk");
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Failed to persist configuration" })),
        ));
    }

    Ok(Json(json!({ "status": "ok" })))
}

async fn get_tokens(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let token_map = state.token_map.read().await;
    let tokens: Vec<serde_json::Value> = token_map
        .iter()
        .map(|(token, app_name)| json!({ "token": token, "app_name": app_name }))
        .collect();
    Json(json!(tokens))
}

#[derive(Deserialize)]
struct AddTokenRequest {
    token: String,
    app_name: String,
}

async fn add_token(
    State(state): State<Arc<AppState>>,
    Json(body): Json<AddTokenRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    {
        let mut token_map = state.token_map.write().await;
        token_map.insert(body.token, body.app_name);
    }

    if let Err(e) = save_config_to_disk(&state).await {
        error!(error = %e, "Failed to save config to disk");
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Failed to persist configuration" })),
        ));
    }

    Ok(Json(json!({ "status": "ok" })))
}

async fn delete_token(
    State(state): State<Arc<AppState>>,
    Path(token): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    {
        let mut token_map = state.token_map.write().await;
        token_map.remove(&token);
    }

    if let Err(e) = save_config_to_disk(&state).await {
        error!(error = %e, "Failed to save config to disk");
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Failed to persist configuration" })),
        ));
    }

    Ok(Json(json!({ "status": "ok" })))
}

async fn get_config(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let ollama_url = state.upstream_url.read().await.clone();
    let privacy_mode = *state.privacy_mode.read().await;
    Json(json!({ "ollama_url": ollama_url, "privacy_mode": privacy_mode }))
}

#[derive(Deserialize)]
struct ConfigUpdate {
    ollama_url: String,
    privacy_mode: Option<bool>,
}

async fn put_config(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ConfigUpdate>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    {
        let mut url = state.upstream_url.write().await;
        *url = body.ollama_url;
    }
    {
        let mut pm = state.privacy_mode.write().await;
        *pm = body.privacy_mode.unwrap_or(false);
    }

    if let Err(e) = save_config_to_disk(&state).await {
        error!(error = %e, "Failed to save config to disk");
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Failed to persist configuration" })),
        ));
    }

    Ok(Json(json!({ "status": "ok" })))
}

async fn get_models(State(state): State<Arc<AppState>>) -> Result<Response, StatusCode> {
    let upstream_url = state.upstream_url.read().await.clone();
    let url = format!("{}/api/tags", upstream_url.trim_end_matches('/'));
    let uri: Uri = url.parse().map_err(|_| StatusCode::BAD_GATEWAY)?;
    let upstream_req = hyper::Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    let resp = state.http_client.request(upstream_req).await.map_err(|e| {
        error!("Upstream request failed: {}", e);
        StatusCode::BAD_GATEWAY
    })?;
    Ok(ollama_response(resp))
}

async fn get_models_running(State(state): State<Arc<AppState>>) -> Result<Response, StatusCode> {
    let upstream_url = state.upstream_url.read().await.clone();
    let url = format!("{}/api/ps", upstream_url.trim_end_matches('/'));
    let uri: Uri = url.parse().map_err(|_| StatusCode::BAD_GATEWAY)?;
    let upstream_req = hyper::Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    let resp = state.http_client.request(upstream_req).await.map_err(|e| {
        error!("Upstream request failed: {}", e);
        StatusCode::BAD_GATEWAY
    })?;
    Ok(ollama_response(resp))
}

async fn admin_chat(
    State(state): State<Arc<AppState>>,
    mut req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    req.extensions_mut().insert(AppName("admin-ui".to_string()));
    proxy_handler(State(state), req).await
}

fn ollama_response(resp: hyper::Response<hyper::body::Incoming>) -> Response {
    let (parts, body) = resp.into_parts();
    let body = Body::new(body.map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    }));
    Response::from_parts(parts, body)
}

async fn save_config_to_disk(state: &Arc<AppState>) -> anyhow::Result<()> {
    let langfuse = state.langfuse_config.read().await.clone();
    let tokens: Vec<TokenEntry> = {
        let token_map = state.token_map.read().await;
        token_map
            .iter()
            .map(|(token, app_name)| TokenEntry {
                token: token.clone(),
                app_name: app_name.clone(),
            })
            .collect()
    };

    let upstream_url = state.upstream_url.read().await.clone();
    let privacy_mode = *state.privacy_mode.read().await;
    let config = Config {
        ollama: OllamaConfig {
            upstream_url: upstream_url,
        },
        langfuse,
        tokens,
        server: crate::config::ServerConfig {
            privacy_mode,
            ..state.server_config.clone()
        },
    };

    let _write_guard = state.config_write_lock.lock().await;
    config.save(&state.config_path)?;
    Ok(())
}
