use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use tokio::sync::{Mutex, RwLock};

use crate::config::{LangfuseConfig, ServerConfig};
use crate::langfuse::LangfuseCollector;

pub struct AppState {
    pub config_path: PathBuf,
    pub admin_password: String,
    pub token_map: Arc<RwLock<HashMap<String, String>>>,
    pub langfuse_config: Arc<RwLock<LangfuseConfig>>,
    pub langfuse_collector: Arc<RwLock<Option<Arc<LangfuseCollector>>>>,
    pub upstream_url: Arc<RwLock<String>>,
    pub privacy_mode: Arc<RwLock<bool>>,
    pub http_client: Client<HttpConnector, Body>,
    pub server_config: ServerConfig,
    pub config_write_lock: Mutex<()>,
}
