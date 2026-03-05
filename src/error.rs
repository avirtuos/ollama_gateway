#![allow(dead_code)]
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

pub enum GatewayError {
    Unauthorized(String),
    BadGateway(String),
    InternalError(String),
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            GatewayError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg),
            GatewayError::BadGateway(msg) => (StatusCode::BAD_GATEWAY, msg),
            GatewayError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };
        let body = Json(json!({ "error": { "message": message, "type": "gateway_error" } }));
        (status, body).into_response()
    }
}
