use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    /// Legacy single-backend config; migrated to `backends` on load.
    #[serde(skip_serializing)]
    pub ollama: Option<OllamaConfig>,
    #[serde(default)]
    pub backends: Vec<BackendConfig>,
    pub langfuse: LangfuseConfig,
    #[serde(default)]
    pub tokens: Vec<TokenEntry>,
    pub server: ServerConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    Ollama,
    Llamacpp,
}

impl Default for BackendType {
    fn default() -> Self {
        BackendType::Ollama
    }
}

/// Legacy single-backend config — kept for migration only.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OllamaConfig {
    pub upstream_url: String,
    #[serde(default)]
    pub backend_type: BackendType,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BackendConfig {
    pub name: String,
    pub url: String,
    #[serde(default)]
    pub backend_type: BackendType,
    #[serde(default)]
    pub priority: i32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LangfuseConfig {
    pub enabled: bool,
    pub host: String,
    pub public_key: String,
    pub secret_key: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_flush_interval_ms")]
    pub flush_interval_ms: u64,
}

fn default_batch_size() -> usize {
    10
}

fn default_flush_interval_ms() -> u64 {
    10000
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TokenEntry {
    pub token: String,
    pub app_name: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ServerConfig {
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,
    #[serde(default = "default_listen_port")]
    pub listen_port: u16,
    #[serde(default = "default_admin_port")]
    pub admin_port: u16,
    #[serde(default)]
    pub privacy_mode: bool,
    #[serde(default = "default_model_refresh_interval_secs")]
    pub model_refresh_interval_secs: u64,
}

fn default_listen_addr() -> String {
    "0.0.0.0".to_string()
}

fn default_listen_port() -> u16 {
    8080
}

fn default_admin_port() -> u16 {
    8081
}

fn default_model_refresh_interval_secs() -> u64 {
    60
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&content)?;
        config.normalize();
        Ok(config)
    }

    /// Load config from `path`, or create a default config file there if it doesn't exist.
    pub fn load_or_create(path: &Path) -> anyhow::Result<Self> {
        if path.exists() {
            return Self::load(path);
        }

        let config = Self::default();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        config.save(path)?;
        tracing::info!(path = %path.display(), "Created default config file");
        Ok(config)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let content = toml::to_string(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Migrate legacy `[ollama]` config to `[[backends]]` if needed.
    pub fn normalize(&mut self) {
        if self.backends.is_empty() {
            if let Some(ref ollama) = self.ollama {
                self.backends.push(BackendConfig {
                    name: "default".to_string(),
                    url: ollama.upstream_url.clone(),
                    backend_type: ollama.backend_type.clone(),
                    priority: 0,
                });
            }
        }
    }

    pub fn token_map(&self) -> HashMap<String, String> {
        self.tokens
            .iter()
            .map(|t| (t.token.clone(), t.app_name.clone()))
            .collect()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ollama: None,
            backends: vec![BackendConfig {
                name: "default".to_string(),
                url: "http://localhost:11434".to_string(),
                backend_type: BackendType::default(),
                priority: 0,
            }],
            langfuse: LangfuseConfig {
                enabled: false,
                host: "https://cloud.langfuse.com".to_string(),
                public_key: String::new(),
                secret_key: String::new(),
                batch_size: default_batch_size(),
                flush_interval_ms: default_flush_interval_ms(),
            },
            tokens: Vec::new(),
            server: ServerConfig {
                listen_addr: default_listen_addr(),
                listen_port: default_listen_port(),
                admin_port: default_admin_port(),
                privacy_mode: false,
                model_refresh_interval_secs: default_model_refresh_interval_secs(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_legacy_config() {
        let toml_content = r#"
[ollama]
upstream_url = "http://localhost:11434"

[langfuse]
enabled = true
host = "https://cloud.langfuse.com"
public_key = "pk-lf-test"
secret_key = "sk-lf-test"
batch_size = 5
flush_interval_ms = 1000

[[tokens]]
token = "sk-myapp-abc123"
app_name = "my-frontend-app"

[[tokens]]
token = "sk-backend-def456"
app_name = "backend-service"

[server]
listen_addr = "0.0.0.0"
listen_port = 8080
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(toml_content.as_bytes()).unwrap();
        let config = Config::load(file.path()).unwrap();

        // Legacy config should be migrated to backends
        assert_eq!(config.backends.len(), 1);
        assert_eq!(config.backends[0].url, "http://localhost:11434");
        assert_eq!(config.backends[0].name, "default");
        assert!(config.langfuse.enabled);
        assert_eq!(config.tokens.len(), 2);

        let map = config.token_map();
        assert_eq!(map.get("sk-myapp-abc123").unwrap(), "my-frontend-app");
        assert_eq!(map.get("sk-backend-def456").unwrap(), "backend-service");
        assert_eq!(config.backends[0].backend_type, BackendType::Ollama);
    }

    #[test]
    fn test_parse_new_config() {
        let toml_content = r#"
[[backends]]
name = "local-ollama"
url = "http://localhost:11434"
backend_type = "ollama"
priority = 0

[[backends]]
name = "remote-gpu"
url = "http://gpu-server:8080"
backend_type = "llamacpp"
priority = 10

[langfuse]
enabled = false
host = "https://cloud.langfuse.com"
public_key = ""
secret_key = ""

[server]
listen_addr = "0.0.0.0"
listen_port = 8080
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(toml_content.as_bytes()).unwrap();
        let config = Config::load(file.path()).unwrap();

        assert_eq!(config.backends.len(), 2);
        assert_eq!(config.backends[0].name, "local-ollama");
        assert_eq!(config.backends[0].priority, 0);
        assert_eq!(config.backends[1].name, "remote-gpu");
        assert_eq!(config.backends[1].backend_type, BackendType::Llamacpp);
        assert_eq!(config.server.model_refresh_interval_secs, 60); // default
    }
}
