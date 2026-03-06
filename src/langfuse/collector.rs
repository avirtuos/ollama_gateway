use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use langfuse_ergonomic::{Batcher, ClientBuilder, IngestionResponse, Result as LangfuseResult};

use crate::config::LangfuseConfig;
use super::models::LangfuseEvent;

pub struct LangfuseCollector {
    batcher: Arc<Batcher>,
}

impl LangfuseCollector {
    pub async fn new(config: &LangfuseConfig) -> Self {
        let client = ClientBuilder::new()
            .public_key(&config.public_key)
            .secret_key(&config.secret_key)
            .base_url(config.host.clone())
            .build()
            .expect("Failed to build Langfuse client");

        // Disable the Batcher's internal timer — we run our own flush loop so
        // we can log results. The batch-size threshold is still respected.
        let batcher = Arc::new(
            Batcher::builder()
                .client(client)
                .max_events(config.batch_size)
                .flush_interval(Duration::from_secs(86400)) // effectively disabled
                .max_retries(3_u32)
                .build()
                .await,
        );

        // Spawn our timer-based flush loop
        tokio::spawn(flush_loop(Arc::clone(&batcher), config.flush_interval_ms));

        Self { batcher }
    }

    pub fn send(&self, event: LangfuseEvent) {
        let batcher = Arc::clone(&self.batcher);
        tokio::spawn(async move {
            info!(
                trace_id = %event.trace_id,
                app_name = %event.app_name,
                model = %event.model,
                endpoint = %event.endpoint,
                session_id = ?event.session_id,
                "queuing Langfuse event"
            );
            let ingestion_events = event.into_ingestion_events();
            debug!(count = ingestion_events.len(), "adding to Langfuse batcher");
            for ingestion_event in ingestion_events {
                if let Err(e) = batcher.add(ingestion_event).await {
                    warn!(error = %e, "Failed to queue event in Langfuse batcher");
                }
            }
        });
    }

    pub async fn shutdown(&self) {
        info!("Flushing Langfuse buffer before shutdown");
        log_flush_result("shutdown", self.batcher.flush().await);
    }
}

async fn flush_loop(batcher: Arc<Batcher>, interval_ms: u64) {
    let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
    interval.tick().await; // skip immediate first tick
    loop {
        interval.tick().await;
        debug!("Langfuse timer flush triggered");
        log_flush_result("timer", batcher.flush().await);
    }
}

fn log_flush_result(context: &str, result: LangfuseResult<IngestionResponse>) {
    match result {
        Ok(resp) if resp.failure_count == 0 && resp.success_count == 0 => {
            // Nothing to flush — skip noisy log
        }
        Ok(resp) if resp.is_success() => {
            info!(
                context,
                success_count = resp.success_count,
                "Langfuse flush successful"
            );
        }
        Ok(resp) => {
            error!(
                context,
                success_count = resp.success_count,
                failure_count = resp.failure_count,
                "Langfuse flush partial failure"
            );
            for f in &resp.failures {
                error!(event_id = %f.event_id, message = %f.message, retryable = f.retryable, "Langfuse event error");
            }
        }
        Err(e) => {
            error!(context, error = %e, "Langfuse flush failed");
        }
    }
}
