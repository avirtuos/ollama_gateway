use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path, Query, Request, State},
    http::{StatusCode, Uri, header},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use http_body_util::BodyExt;
use base64::Engine;
use serde::Deserialize;
use serde_json::json;
use tracing::{error, info};

use crate::auth::AppName;
use crate::config::{BackendConfig, BackendType, Config, LangfuseConfig, ProcessorRule, TokenEntry};
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
        .route("/api/backends", get(get_backends).post(add_backend))
        .route("/api/backends/refresh", post(refresh_backends))
        .route("/api/backends/{name}", put(update_backend).delete(delete_backend))
        .route("/api/models", get(get_models))
        .route("/api/models/running", get(get_models_running))
        .route("/api/metrics/backends", get(get_metrics_backends))
        .route("/api/metrics/summary", get(get_metrics_summary))
        .route("/api/metrics/timeseries", get(get_metrics_timeseries))
        .route("/api/processors", get(get_processors))
        .route("/api/processor-rules", get(get_processor_rules).post(add_processor_rule))
        .route("/api/processor-rules/{index}", put(update_processor_rule).delete(delete_processor_rule))
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

    {
        let collector_guard = state.langfuse_collector.read().await;
        if let Some(ref c) = *collector_guard {
            c.shutdown().await;
        }
    }

    let new_collector = if new_config.enabled {
        info!(host = %new_config.host, "Rebuilding Langfuse collector");
        let collector = LangfuseCollector::new(&new_config).await;
        Some(Arc::new(collector))
    } else {
        info!("Langfuse disabled, clearing collector");
        None
    };

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
    let privacy_mode = *state.privacy_mode.read().await;
    let model_refresh_interval_secs = state.server_config.model_refresh_interval_secs;
    Json(json!({
        "privacy_mode": privacy_mode,
        "model_refresh_interval_secs": model_refresh_interval_secs,
    }))
}

#[derive(Deserialize)]
struct ConfigUpdate {
    privacy_mode: Option<bool>,
}

async fn put_config(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ConfigUpdate>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
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

// --- Backend CRUD ---

async fn get_backends(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let backends = state.backends.read().await;
    let registry = state.model_registry.read().await.clone();

    let list: Vec<serde_json::Value> = backends.iter().map(|b| {
        let reg_backend = registry.backends.iter().find(|rb| rb.config.name == b.name);
        let healthy = reg_backend.map(|rb| rb.healthy).unwrap_or(false);
        let model_count = reg_backend.map(|rb| rb.models.len()).unwrap_or(0);
        json!({
            "name": b.name,
            "url": b.url,
            "backend_type": b.backend_type,
            "priority": b.priority,
            "healthy": healthy,
            "model_count": model_count,
        })
    }).collect();

    Json(json!(list))
}

#[derive(Deserialize)]
struct BackendRequest {
    name: String,
    url: String,
    backend_type: Option<String>,
    priority: Option<i32>,
}

async fn add_backend(
    State(state): State<Arc<AppState>>,
    Json(body): Json<BackendRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let bt = parse_backend_type(body.backend_type.as_deref());
    let new_backend = BackendConfig {
        name: body.name,
        url: body.url,
        backend_type: bt,
        priority: body.priority.unwrap_or(0),
    };

    {
        let mut backends = state.backends.write().await;
        if backends.iter().any(|b| b.name == new_backend.name) {
            return Err((
                StatusCode::CONFLICT,
                Json(json!({ "error": format!("Backend '{}' already exists", new_backend.name) })),
            ));
        }
        backends.push(new_backend);
    }

    if let Err(e) = save_config_to_disk(&state).await {
        error!(error = %e, "Failed to save config to disk");
        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "Failed to persist configuration" }))));
    }

    state.registry_refresh_notify.notify_one();
    Ok(Json(json!({ "status": "ok" })))
}

async fn update_backend(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(body): Json<BackendRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    {
        let mut backends = state.backends.write().await;
        let backend = backends.iter_mut().find(|b| b.name == name).ok_or_else(|| {
            (StatusCode::NOT_FOUND, Json(json!({ "error": format!("Backend '{}' not found", name) })))
        })?;
        backend.url = body.url;
        backend.backend_type = parse_backend_type(body.backend_type.as_deref());
        backend.priority = body.priority.unwrap_or(backend.priority);
        // Allow renaming
        backend.name = body.name;
    }

    if let Err(e) = save_config_to_disk(&state).await {
        error!(error = %e, "Failed to save config to disk");
        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "Failed to persist configuration" }))));
    }

    state.registry_refresh_notify.notify_one();
    Ok(Json(json!({ "status": "ok" })))
}

async fn delete_backend(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    {
        let mut backends = state.backends.write().await;
        let before = backends.len();
        backends.retain(|b| b.name != name);
        if backends.len() == before {
            return Err((StatusCode::NOT_FOUND, Json(json!({ "error": format!("Backend '{}' not found", name) }))));
        }
    }

    if let Err(e) = save_config_to_disk(&state).await {
        error!(error = %e, "Failed to save config to disk");
        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "Failed to persist configuration" }))));
    }

    state.registry_refresh_notify.notify_one();
    Ok(Json(json!({ "status": "ok" })))
}

async fn refresh_backends(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    state.registry_refresh_notify.notify_one();
    Json(json!({ "status": "ok", "message": "Registry refresh triggered" }))
}

fn parse_backend_type(s: Option<&str>) -> BackendType {
    match s {
        Some("llamacpp") => BackendType::Llamacpp,
        _ => BackendType::Ollama,
    }
}

// --- Model listing (uses registry data) ---

async fn get_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let registry = state.model_registry.read().await.clone();
    let mut seen = std::collections::HashSet::new();
    let mut models: Vec<serde_json::Value> = Vec::new();

    for backend in registry.all_healthy_backends() {
        for model_name in &backend.models {
            if seen.insert(model_name.clone()) {
                models.push(json!({ "name": model_name }));
            }
        }
    }

    Json(json!({ "models": models }))
}

async fn get_models_running(State(state): State<Arc<AppState>>) -> Result<Response, StatusCode> {
    let registry = state.model_registry.read().await.clone();
    let ollama_backends = registry.all_healthy_ollama_backends();
    if ollama_backends.is_empty() {
        return Ok(Json(json!({ "models": [] })).into_response());
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

    Ok(Json(json!({ "models": merged })).into_response())
}

async fn admin_chat(
    State(state): State<Arc<AppState>>,
    mut req: Request<Body>,
) -> Result<Response<Body>, StatusCode> {
    req.extensions_mut().insert(AppName("admin-ui".to_string()));

    // Peek at the body to determine target backend
    let (parts, body) = req.into_parts();
    let req_bytes = body.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
    let req_json: Option<serde_json::Value> = serde_json::from_slice(&req_bytes).ok();

    let model_name = req_json.as_ref().and_then(|j| {
        j.get("model").and_then(|v| v.as_str()).map(|s| s.to_string())
    });

    let registry = state.model_registry.read().await.clone();
    let backend = model_name.as_deref()
        .and_then(|m| registry.resolve_backend(m))
        .or_else(|| registry.default_backend());

    // For llamacpp backends, rewrite the URI
    let new_path = if let Some(b) = backend {
        if b.backend_type == BackendType::Llamacpp {
            "/v1/chat/completions"
        } else {
            "/api/chat"
        }
    } else {
        "/api/chat"
    };

    let new_uri: Uri = new_path.parse().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut rebuilt = Request::from_parts(parts, Body::from(req_bytes));
    *rebuilt.uri_mut() = new_uri;

    proxy_handler(State(state), rebuilt).await
}

// --- Metrics endpoints ---

async fn get_metrics_backends(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let data = state.metrics_collector.query_backend_summary().await;
    Json(serde_json::json!(data))
}

async fn get_metrics_summary(
    State(state): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let range = params.get("range").map(|s| s.as_str()).unwrap_or("24h");
    Json(state.metrics_collector.query_summary(range).await)
}

async fn get_metrics_timeseries(
    State(state): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let range = params.get("range").map(|s| s.as_str()).unwrap_or("24h");
    let backend = params.get("backend").filter(|s| !s.is_empty()).cloned();
    let data = state.metrics_collector.query_timeseries(range, backend).await;
    Json(serde_json::json!(data))
}

// --- Processors ---

async fn get_processors(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let available = state.processor_registry.list();
    Json(serde_json::to_value(&available).unwrap_or_default())
}

async fn get_processor_rules(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let rules = state.processor_rules.read().await;
    Json(serde_json::to_value(&*rules).unwrap_or_default())
}

#[derive(Deserialize)]
struct ProcessorRuleRequest {
    model_pattern: String,
    #[serde(default)]
    backend_name: String,
    #[serde(default)]
    preprocessors: Vec<String>,
    #[serde(default)]
    postprocessors: Vec<String>,
}

async fn add_processor_rule(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ProcessorRuleRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if body.model_pattern.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "model_pattern is required" })),
        ));
    }

    {
        let mut rules = state.processor_rules.write().await;
        rules.push(ProcessorRule {
            model_pattern: body.model_pattern,
            backend_name: body.backend_name,
            preprocessors: body.preprocessors,
            postprocessors: body.postprocessors,
        });
    }

    if let Err(e) = save_config_to_disk(&state).await {
        error!(error = %e, "Failed to save config to disk");
        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "Failed to persist configuration" }))));
    }

    Ok(Json(json!({ "status": "ok" })))
}

async fn update_processor_rule(
    State(state): State<Arc<AppState>>,
    Path(index): Path<usize>,
    Json(body): Json<ProcessorRuleRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    {
        let mut rules = state.processor_rules.write().await;
        if index >= rules.len() {
            return Err((StatusCode::NOT_FOUND, Json(json!({ "error": "Rule index out of range" }))));
        }
        rules[index] = ProcessorRule {
            model_pattern: body.model_pattern,
            backend_name: body.backend_name,
            preprocessors: body.preprocessors,
            postprocessors: body.postprocessors,
        };
    }

    if let Err(e) = save_config_to_disk(&state).await {
        error!(error = %e, "Failed to save config to disk");
        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "Failed to persist configuration" }))));
    }

    Ok(Json(json!({ "status": "ok" })))
}

async fn delete_processor_rule(
    State(state): State<Arc<AppState>>,
    Path(index): Path<usize>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    {
        let mut rules = state.processor_rules.write().await;
        if index >= rules.len() {
            return Err((StatusCode::NOT_FOUND, Json(json!({ "error": "Rule index out of range" }))));
        }
        rules.remove(index);
    }

    if let Err(e) = save_config_to_disk(&state).await {
        error!(error = %e, "Failed to save config to disk");
        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "Failed to persist configuration" }))));
    }

    Ok(Json(json!({ "status": "ok" })))
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

    let backends = state.backends.read().await.clone();
    let privacy_mode = *state.privacy_mode.read().await;
    let processor_rules = state.processor_rules.read().await.clone();
    let config = Config {
        ollama: None,
        backends,
        langfuse,
        tokens,
        server: crate::config::ServerConfig {
            privacy_mode,
            ..state.server_config.clone()
        },
        processor_rules,
    };

    let _write_guard = state.config_write_lock.lock().await;
    config.save(&state.config_path)?;
    Ok(())
}
