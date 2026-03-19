use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

use chrono::{Duration, Utc};
use rusqlite::{params, Connection};
use serde_json::json;
use tracing::error;

pub struct MetricsRecord {
    pub timestamp: String,
    pub backend_name: String,
    pub model: String,
    pub endpoint: String,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub tokens_per_sec: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub latency_ms: f64,
    pub status_code: u16,
}

pub struct MetricsCollector {
    tx: std::sync::mpsc::Sender<MetricsRecord>,
    reader: Arc<Mutex<Connection>>,
}

const SCHEMA: &str = "
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS api_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    backend           TEXT NOT NULL,
    model             TEXT NOT NULL,
    endpoint          TEXT NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    tokens_per_sec    REAL,
    ttft_ms           REAL,
    latency_ms        REAL NOT NULL,
    status_code       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_api_calls_timestamp ON api_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_calls_backend ON api_calls(backend);
";

impl MetricsCollector {
    pub fn new(db_path: &Path) -> Self {
        // Reader connection (also initialises schema)
        let reader_conn = Connection::open(db_path).expect("Failed to open metrics DB");
        reader_conn.execute_batch(SCHEMA).expect("Failed to create metrics schema");
        let reader = Arc::new(Mutex::new(reader_conn));

        let (tx, rx) = std::sync::mpsc::channel::<MetricsRecord>();

        // Dedicated writer thread — owns its own connection
        let db_path_owned = db_path.to_path_buf();
        thread::Builder::new()
            .name("metrics-writer".into())
            .spawn(move || {
                let conn = match Connection::open(&db_path_owned) {
                    Ok(c) => c,
                    Err(e) => {
                        error!("Failed to open metrics DB for writing: {}", e);
                        return;
                    }
                };
                if let Err(e) = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;") {
                    error!("Failed to configure metrics DB WAL: {}", e);
                    return;
                }
                writer_loop(conn, rx);
            })
            .expect("Failed to spawn metrics writer thread");

        MetricsCollector { tx, reader }
    }

    pub fn record(&self, record: MetricsRecord) {
        // Fire-and-forget; ignore send errors (channel closed on shutdown)
        let _ = self.tx.send(record);
    }

    pub async fn query_backend_summary(&self) -> Vec<serde_json::Value> {
        let reader = self.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.lock().unwrap();
            let mut stmt = match conn.prepare(
                "SELECT backend,
                        COUNT(*) as calls,
                        COALESCE(SUM(prompt_tokens), 0) as pt,
                        COALESCE(SUM(completion_tokens), 0) as ct,
                        AVG(tokens_per_sec) as avg_tps,
                        AVG(latency_ms) as avg_lat
                 FROM api_calls GROUP BY backend ORDER BY backend",
            ) {
                Ok(s) => s,
                Err(e) => {
                    error!("Failed to prepare backend summary query: {}", e);
                    return vec![];
                }
            };
            stmt.query_map([], |row| {
                Ok(json!({
                    "backend":           row.get::<_, String>(0)?,
                    "calls":             row.get::<_, i64>(1)?,
                    "prompt_tokens":     row.get::<_, i64>(2)?,
                    "completion_tokens": row.get::<_, i64>(3)?,
                    "avg_tokens_per_sec": row.get::<_, Option<f64>>(4)?,
                    "avg_latency_ms":    row.get::<_, Option<f64>>(5)?,
                }))
            })
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
        })
        .await
        .unwrap_or_default()
    }

    pub async fn query_summary(&self, range: &str) -> serde_json::Value {
        let since = range_to_since(range);
        let reader = self.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.lock().unwrap();
            conn.query_row(
                "SELECT COUNT(*),
                        COALESCE(SUM(prompt_tokens), 0),
                        COALESCE(SUM(completion_tokens), 0),
                        AVG(tokens_per_sec),
                        AVG(latency_ms)
                 FROM api_calls WHERE timestamp >= ?1",
                params![since],
                |row| {
                    Ok(json!({
                        "total_calls":             row.get::<_, i64>(0)?,
                        "total_prompt_tokens":     row.get::<_, i64>(1)?,
                        "total_completion_tokens": row.get::<_, i64>(2)?,
                        "avg_tokens_per_sec":      row.get::<_, Option<f64>>(3)?,
                        "avg_latency_ms":          row.get::<_, Option<f64>>(4)?,
                    }))
                },
            )
            .unwrap_or_else(|_| {
                json!({
                    "total_calls": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                })
            })
        })
        .await
        .unwrap_or_else(|_| json!({}))
    }

    pub async fn query_timeseries(
        &self,
        range: &str,
        backend_filter: Option<String>,
    ) -> Vec<serde_json::Value> {
        let since = range_to_since(range);
        let bucket_sql = range_to_bucket_sql(range).to_string();
        let reader = self.reader.clone();
        tokio::task::spawn_blocking(move || {
            let conn = reader.lock().unwrap();
            let cols = "COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), AVG(tokens_per_sec), AVG(latency_ms)";
            if let Some(ref backend) = backend_filter {
                let sql = format!(
                    "SELECT {bucket_sql} as bucket, {cols} \
                     FROM api_calls WHERE timestamp >= ?1 AND backend = ?2 \
                     GROUP BY bucket ORDER BY bucket"
                );
                conn.prepare(&sql)
                    .ok()
                    .and_then(|mut s| {
                        s.query_map(params![since, backend], row_to_ts_json)
                            .ok()
                            .map(|rows| rows.filter_map(|r| r.ok()).collect::<Vec<_>>())
                    })
                    .unwrap_or_default()
            } else {
                let sql = format!(
                    "SELECT {bucket_sql} as bucket, {cols} \
                     FROM api_calls WHERE timestamp >= ?1 \
                     GROUP BY bucket ORDER BY bucket"
                );
                conn.prepare(&sql)
                    .ok()
                    .and_then(|mut s| {
                        s.query_map(params![since], row_to_ts_json)
                            .ok()
                            .map(|rows| rows.filter_map(|r| r.ok()).collect::<Vec<_>>())
                    })
                    .unwrap_or_default()
            }
        })
        .await
        .unwrap_or_default()
    }
}

fn row_to_ts_json(row: &rusqlite::Row<'_>) -> rusqlite::Result<serde_json::Value> {
    Ok(json!({
        "bucket":             row.get::<_, String>(0)?,
        "calls":              row.get::<_, i64>(1)?,
        "prompt_tokens":      row.get::<_, i64>(2)?,
        "completion_tokens":  row.get::<_, i64>(3)?,
        "avg_tokens_per_sec": row.get::<_, Option<f64>>(4)?,
        "avg_latency_ms":     row.get::<_, Option<f64>>(5)?,
    }))
}

fn range_to_since(range: &str) -> String {
    let dur = match range {
        "1h"  => Duration::hours(1),
        "6h"  => Duration::hours(6),
        "7d"  => Duration::days(7),
        "30d" => Duration::days(30),
        _     => Duration::hours(24),
    };
    (Utc::now() - dur).to_rfc3339()
}

fn range_to_bucket_sql(range: &str) -> &'static str {
    match range {
        "1h"  => "strftime('%Y-%m-%dT%H:%M', timestamp)",
        "6h"  => "strftime('%Y-%m-%dT%H:', timestamp) || printf('%02d', (cast(strftime('%M', timestamp) as integer)/5)*5)",
        "7d"  => "strftime('%Y-%m-%dT%H:00', timestamp)",
        "30d" => "strftime('%Y-%m-%dT', timestamp) || printf('%02d', (cast(strftime('%H', timestamp) as integer)/6)*6) || ':00'",
        _     => "strftime('%Y-%m-%dT%H:', timestamp) || printf('%02d', (cast(strftime('%M', timestamp) as integer)/15)*15)",
    }
}

fn writer_loop(mut conn: Connection, rx: std::sync::mpsc::Receiver<MetricsRecord>) {
    while let Ok(first) = rx.recv() {
        let mut batch = vec![first];
        while let Ok(record) = rx.try_recv() {
            batch.push(record);
        }
        if let Err(e) = insert_batch(&mut conn, &batch) {
            error!("Failed to insert metrics batch (size={}): {}", batch.len(), e);
        }
    }
}

fn insert_batch(conn: &mut Connection, batch: &[MetricsRecord]) -> rusqlite::Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO api_calls
             (timestamp, backend, model, endpoint, prompt_tokens, completion_tokens,
              tokens_per_sec, ttft_ms, latency_ms, status_code)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
        )?;
        for r in batch {
            stmt.execute(params![
                r.timestamp,
                r.backend_name,
                r.model,
                r.endpoint,
                r.prompt_tokens.map(|v| v as i64),
                r.completion_tokens.map(|v| v as i64),
                r.tokens_per_sec,
                r.ttft_ms,
                r.latency_ms,
                r.status_code as i64,
            ])?;
        }
    }
    tx.commit()
}
