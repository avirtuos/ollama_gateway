mod admin;
mod auth;
mod config;
mod connection_id;
mod error;
mod langfuse;
mod ollama;
mod proxy;
mod state;

use std::{net::SocketAddr, path::PathBuf, sync::Arc};

use axum::{middleware, routing::any, Router};
use clap::Parser;
use hyper_util::{client::legacy::Client, rt::TokioExecutor};
use tokio::sync::{Mutex, RwLock};
use tracing::info;

use admin::admin_router;
use auth::auth_middleware;
use config::Config;
use connection_id::ConnectionIdLayer;
use langfuse::LangfuseCollector;
use proxy::proxy_handler;
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

    info!(
        upstream = %config.ollama.upstream_url,
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

    let state = Arc::new(AppState {
        config_path: cli.config.clone(),
        admin_password,
        token_map: Arc::new(RwLock::new(config.token_map())),
        langfuse_config: Arc::new(RwLock::new(config.langfuse.clone())),
        langfuse_collector: Arc::new(RwLock::new(langfuse_collector.clone())),
        upstream_url: Arc::new(RwLock::new(config.ollama.upstream_url.clone())),
        http_client,
        server_config: config.server.clone(),
        config_write_lock: Mutex::new(()),
    });

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

    let proxy_server = axum::serve(proxy_listener, tower::Layer::layer(&ConnectionIdLayer, proxy_app))
        .with_graceful_shutdown(async move {
            shutdown_rx.wait_for(|v| *v).await.ok();
        });

    let admin_server = axum::serve(admin_listener, admin_app)
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
