use std::sync::Arc;

use axum::{
    body::Body,
    extract::{ConnectInfo, Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
    Json,
};
use serde_json::json;
use tracing::warn;

use crate::state::AppState;

/// Extension inserted into requests after successful auth, carrying the app name.
#[derive(Clone, Debug)]
pub struct AppName(pub String);

pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    mut req: Request<Body>,
    next: Next,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok());

    let app_name = match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header["Bearer ".len()..];
            let token_map = state.token_map.read().await;
            token_map.get(token).cloned()
        }
        _ => None,
    };

    match app_name {
        Some(name) => {
            // Remove auth header before forwarding
            req.headers_mut().remove("Authorization");
            req.extensions_mut().insert(AppName(name));
            Ok(next.run(req).await)
        }
        None => {
            let remote = req
                .extensions()
                .get::<ConnectInfo<std::net::SocketAddr>>()
                .map(|c| c.0.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let url = req
                .uri()
                .path_and_query()
                .map(|pq| pq.as_str())
                .unwrap_or("/")
                .to_string();
            let headers: Vec<(&str, &str)> = req
                .headers()
                .iter()
                .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.as_str(), v)))
                .collect();
            warn!(remote = %remote, url = %url, ?headers, "authorization failure");
            Err((
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": {
                        "message": "Invalid or missing Bearer token",
                        "type": "authentication_error"
                    }
                })),
            ))
        }
    }
}
