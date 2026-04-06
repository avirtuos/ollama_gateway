use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use tokio::sync::{Mutex, Notify, RwLock};

use crate::config::{BackendConfig, LangfuseConfig, ProcessorRule, ServerConfig};
use crate::langfuse::LangfuseCollector;
use crate::metrics::MetricsCollector;
use crate::processors::ProcessorRegistry;
use crate::registry::ModelRegistry;

pub struct AppState {
    pub config_path: PathBuf,
    pub admin_password: String,
    pub token_map: Arc<RwLock<HashMap<String, String>>>,
    pub langfuse_config: Arc<RwLock<LangfuseConfig>>,
    pub langfuse_collector: Arc<RwLock<Option<Arc<LangfuseCollector>>>>,
    pub backends: Arc<RwLock<Vec<BackendConfig>>>,
    /// Inner `Arc<ModelRegistry>` so readers can clone cheaply without holding the lock.
    pub model_registry: Arc<RwLock<Arc<ModelRegistry>>>,
    pub privacy_mode: Arc<RwLock<bool>>,
    pub metrics_collector: Arc<MetricsCollector>,
    pub http_client: Client<HttpConnector, Body>,
    pub server_config: ServerConfig,
    pub config_write_lock: Mutex<()>,
    /// Trigger an immediate model registry refresh from the admin API.
    pub registry_refresh_notify: Arc<Notify>,
    /// Built-in processor implementations (immutable after init).
    pub processor_registry: Arc<ProcessorRegistry>,
    /// User-configured rules mapping (model, backend) → processor IDs.
    pub processor_rules: Arc<RwLock<Vec<ProcessorRule>>>,
}
