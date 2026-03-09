mod admin;
mod auth;
mod config;
mod connection_id;
mod error;
mod langfuse;
mod ollama;
mod proxy;
mod registry;
mod state;

use std::{net::SocketAddr, path::PathBuf, sync::Arc};

use axum::{middleware, routing::any, Router};
use clap::Parser;
use hyper_util::{client::legacy::Client, rt::TokioExecutor};
use tokio::sync::{Mutex, Notify, RwLock};
use tracing::info;

use admin::admin_router;
use auth::auth_middleware;
use config::Config;
use connection_id::ConnectionIdLayer;
use langfuse::LangfuseCollector;
use proxy::proxy_handler;
use registry::ModelRegistry;
use state::AppState;

#[derive(Parser, Debug)]
#[command(name = "ollama_gateway", about = "Authenticated reverse proxy for Ollama with Langfuse tracing")]
struct Cli {
    /// Path to the TOML configuration file
    #[arg(short, long)]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let config = Config::load_or_create(&cli.config)?;

    let admin_password = std::env::var("ADMIN_PASSWORD").unwrap_or_else(|_| {
        tracing::warn!("ADMIN_PASSWORD env var not set; admin UI will require empty password");
        String::new()
    });

    // Env vars override config file ports (config file values are preserved for saving)
    let proxy_port: u16 = std::env::var("PROXY_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(config.server.listen_port);

    let admin_port: u16 = std::env::var("ADMIN_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(config.server.admin_port);

    let backend_names: Vec<&str> = config.backends.iter().map(|b| b.name.as_str()).collect();
    info!(
        backends = ?backend_names,
        proxy_port,
        admin_port,
        "Starting Ollama Gateway",
    );

    // Build Langfuse collector if enabled
    let langfuse_collector: Option<Arc<LangfuseCollector>> = if config.langfuse.enabled {
        info!(host = %config.langfuse.host, "Langfuse tracing enabled");
        let collector = LangfuseCollector::new(&config.langfuse).await;
        Some(Arc::new(collector))
    } else {
        info!("Langfuse tracing disabled");
        None
    };

    let http_client = Client::builder(TokioExecutor::new()).build_http();

    // Build initial model registry
    info!("Building initial model registry...");
    let initial_registry = ModelRegistry::refresh(&http_client, &config.backends).await;
    let healthy_count = initial_registry.all_healthy_backends().len();
    info!(backends = config.backends.len(), healthy = healthy_count, "Model registry initialized");

    let registry_refresh_notify = Arc::new(Notify::new());

    let state = Arc::new(AppState {
        config_path: cli.config.clone(),
        admin_password,
        token_map: Arc::new(RwLock::new(config.token_map())),
        langfuse_config: Arc::new(RwLock::new(config.langfuse.clone())),
        langfuse_collector: Arc::new(RwLock::new(langfuse_collector.clone())),
        backends: Arc::new(RwLock::new(config.backends.clone())),
        model_registry: Arc::new(RwLock::new(initial_registry)),
        privacy_mode: Arc::new(RwLock::new(config.server.privacy_mode)),
        http_client: http_client.clone(),
        server_config: config.server.clone(),
        config_write_lock: Mutex::new(()),
        registry_refresh_notify: registry_refresh_notify.clone(),
    });

    // Background registry refresh task
    {
        let state = state.clone();
        let refresh_interval_secs = config.server.model_refresh_interval_secs;
        tokio::spawn(async move {
            registry_refresh_loop(state, refresh_interval_secs, registry_refresh_notify).await;
        });
    }

    // Proxy app — Bearer token auth, all Ollama traffic
    let proxy_app = Router::new()
        .route("/{*path}", any(proxy_handler))
        .route("/", any(proxy_handler))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .with_state(state.clone());

    // Admin app — HTTP Basic auth, separate port
    let admin_app = admin_router(state.clone());

    let proxy_addr: SocketAddr = format!("{}:{}", config.server.listen_addr, proxy_port).parse()?;
    let admin_addr: SocketAddr = format!("{}:{}", config.server.listen_addr, admin_port).parse()?;

    let proxy_listener = tokio::net::TcpListener::bind(proxy_addr).await?;
    let admin_listener = tokio::net::TcpListener::bind(admin_addr).await?;

    info!(%proxy_addr, "Proxy listening");
    info!(%admin_addr, "Admin UI listening");

    // Shared shutdown signal
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);
    let mut shutdown_rx2 = shutdown_rx.clone();

    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C handler");
        info!("Shutdown signal received");
        let _ = shutdown_tx.send(true);
    });

    let proxy_server = axum::serve(
        proxy_listener,
        tower::Layer::layer(
            &ConnectionIdLayer,
            proxy_app.into_make_service_with_connect_info::<SocketAddr>(),
        ),
    )
    .with_graceful_shutdown(async move {
        shutdown_rx.wait_for(|v| *v).await.ok();
    });

    let admin_server = axum::serve(
        admin_listener,
        tower::Layer::layer(
            &ConnectionIdLayer,
            admin_app.into_make_service_with_connect_info::<SocketAddr>(),
        ),
    )
    .with_graceful_shutdown(async move {
        shutdown_rx2.wait_for(|v| *v).await.ok();
    });

    tokio::try_join!(proxy_server, admin_server)?;

    // Flush Langfuse on shutdown
    if let Some(collector) = langfuse_collector {
        info!("Flushing Langfuse buffer before exit...");
        collector.shutdown().await;
    }

    info!("Ollama Gateway stopped");
    Ok(())
}

async fn registry_refresh_loop(
    state: Arc<AppState>,
    refresh_interval_secs: u64,
    notify: Arc<Notify>,
) {
    let interval = std::time::Duration::from_secs(refresh_interval_secs);
    loop {
        // Wait for whichever comes first: the timer or a manual trigger
        tokio::select! {
            _ = tokio::time::sleep(interval) => {
                tracing::debug!("Periodic registry refresh triggered");
            }
            _ = notify.notified() => {
                tracing::debug!("Manual registry refresh triggered");
            }
        }

        let backends = state.backends.read().await.clone();
        let new_registry = ModelRegistry::refresh(&state.http_client, &backends).await;
        let healthy = new_registry.all_healthy_backends().len();
        let total = new_registry.backends.len();
        tracing::info!(healthy, total, "Model registry refreshed");

        let mut reg = state.model_registry.write().await;
        *reg = new_registry;
    }
}
