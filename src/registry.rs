use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use http_body_util::BodyExt;
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use tracing::warn;

use crate::config::{BackendConfig, BackendType};

pub struct Backend {
    pub config: BackendConfig,
    pub models: Vec<String>,
    pub healthy: bool,
}

pub struct ModelRegistry {
    pub backends: Vec<Backend>,
    /// model name → backend indices sorted by priority (ascending = preferred)
    model_index: HashMap<String, Vec<usize>>,
}

impl ModelRegistry {
    pub async fn refresh(
        http_client: &Client<HttpConnector, Body>,
        configs: &[BackendConfig],
    ) -> Arc<Self> {
        let mut tasks = tokio::task::JoinSet::new();
        for config in configs.iter().cloned() {
            let client = http_client.clone();
            tasks.spawn(async move {
                let result = fetch_models(&client, &config).await;
                (config, result)
            });
        }

        let mut backends: Vec<Backend> = Vec::new();
        while let Some(join_result) = tasks.join_next().await {
            match join_result {
                Ok((config, Ok(models))) => {
                    backends.push(Backend { config, models, healthy: true });
                }
                Ok((config, Err(e))) => {
                    warn!(name = %config.name, url = %config.url, error = %e, "Backend unreachable, marking unhealthy");
                    backends.push(Backend { config, models: vec![], healthy: false });
                }
                Err(e) => {
                    warn!(error = %e, "Registry refresh task panicked");
                }
            }
        }

        // Sort backends by priority so iteration order is deterministic
        backends.sort_by_key(|b| b.config.priority);

        // Build model → backend index
        let mut model_index: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, backend) in backends.iter().enumerate() {
            for model in &backend.models {
                model_index.entry(model.clone()).or_default().push(i);
            }
        }
        // Each list is already priority-sorted because backends are sorted
        // (indices are added in priority order).

        Arc::new(Self { backends, model_index })
    }

    /// Best healthy backend that has this model (lowest priority number).
    pub fn resolve_backend(&self, model: &str) -> Option<&BackendConfig> {
        let indices = self.model_index.get(model)?;
        for &i in indices {
            if self.backends[i].healthy {
                return Some(&self.backends[i].config);
            }
        }
        None
    }

    /// Lowest priority-number healthy backend (fallback for unknown models).
    pub fn default_backend(&self) -> Option<&BackendConfig> {
        self.backends
            .iter()
            .filter(|b| b.healthy)
            .min_by_key(|b| b.config.priority)
            .map(|b| &b.config)
    }

    /// All healthy backends (for fan-out aggregation).
    pub fn all_healthy_backends(&self) -> Vec<&Backend> {
        self.backends.iter().filter(|b| b.healthy).collect()
    }

    /// All healthy Ollama backends (for `/api/ps` aggregation).
    pub fn all_healthy_ollama_backends(&self) -> Vec<&Backend> {
        self.backends
            .iter()
            .filter(|b| b.healthy && b.config.backend_type == BackendType::Ollama)
            .collect()
    }
}

async fn fetch_models(
    client: &Client<HttpConnector, Body>,
    backend: &BackendConfig,
) -> anyhow::Result<Vec<String>> {
    let base = backend.url.trim_end_matches('/');
    let url = match backend.backend_type {
        BackendType::Ollama => format!("{}/api/tags", base),
        BackendType::Llamacpp => format!("{}/v1/models", base),
    };

    let uri: hyper::Uri = url.parse()?;
    let req = hyper::Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())?;

    let resp = client.request(req).await?;
    let (parts, body) = resp.into_parts();
    if !parts.status.is_success() {
        anyhow::bail!("HTTP {} from {}", parts.status, backend.url);
    }

    let bytes = body.collect().await?.to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes)?;

    let models = match backend.backend_type {
        BackendType::Ollama => json["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["name"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
        BackendType::Llamacpp => json["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
    };

    Ok(models)
}
