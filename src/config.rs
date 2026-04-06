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
    #[serde(default)]
    pub processor_rules: Vec<ProcessorRule>,
}

/// A rule that assigns pre/post processors to a (model_pattern, backend) pair.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProcessorRule {
    /// Glob-style pattern matched against the model name (e.g. `"gemma4*"`, `"*"`).
    pub model_pattern: String,
    /// Backend name to match, or empty string to match all backends.
    #[serde(default)]
    pub backend_name: String,
    /// Processor IDs to run before sending the request upstream.
    #[serde(default)]
    pub preprocessors: Vec<String>,
    /// Processor IDs to run on the response before returning to the client.
    #[serde(default)]
    pub postprocessors: Vec<String>,
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

impl Config {
    /// Resolve all processor IDs that apply for a given model + backend pair.
    pub fn resolve_processors(
        rules: &[ProcessorRule],
        model: &str,
        backend_name: &str,
    ) -> (Vec<String>, Vec<String>) {
        let mut pre = Vec::new();
        let mut post = Vec::new();
        for rule in rules {
            let backend_matches =
                rule.backend_name.is_empty() || rule.backend_name == backend_name;
            if backend_matches && glob_match(&rule.model_pattern, model) {
                for id in &rule.preprocessors {
                    if !pre.contains(id) {
                        pre.push(id.clone());
                    }
                }
                for id in &rule.postprocessors {
                    if !post.contains(id) {
                        post.push(id.clone());
                    }
                }
            }
        }
        (pre, post)
    }
}

/// Simple glob matching: supports `*` as wildcard for any sequence of characters.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pat = pattern.to_lowercase();
    let txt = text.to_lowercase();
    let pat_chars: Vec<char> = pat.chars().collect();
    let txt_chars: Vec<char> = txt.chars().collect();
    let mut dp = vec![vec![false; txt_chars.len() + 1]; pat_chars.len() + 1];
    dp[0][0] = true;
    for i in 1..=pat_chars.len() {
        if pat_chars[i - 1] == '*' {
            dp[i][0] = dp[i - 1][0];
        }
    }
    for i in 1..=pat_chars.len() {
        for j in 1..=txt_chars.len() {
            if pat_chars[i - 1] == '*' {
                dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
            } else if pat_chars[i - 1] == '?' || pat_chars[i - 1] == txt_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }
    dp[pat_chars.len()][txt_chars.len()]
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
            processor_rules: Vec::new(),
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
    fn test_glob_match() {
        assert!(super::glob_match("gemma4*", "gemma4:31b"));
        assert!(super::glob_match("gemma4*", "Gemma4-31B-IT"));
        assert!(super::glob_match("*gemma*", "nvidia/Gemma-4-31B-IT-NVFP4"));
        assert!(super::glob_match("*", "anything"));
        assert!(!super::glob_match("gemma4*", "llama3:8b"));
    }

    #[test]
    fn test_resolve_processors() {
        let rules = vec![
            ProcessorRule {
                model_pattern: "*gemma*".to_string(),
                backend_name: "".to_string(),
                preprocessors: vec!["gemma4-tool-call-fix".to_string()],
                postprocessors: vec!["gemma4-tool-call-fix".to_string()],
            },
        ];
        let (pre, post) = Config::resolve_processors(&rules, "nvidia/Gemma-4-31B-IT", "local");
        assert_eq!(pre, vec!["gemma4-tool-call-fix"]);
        assert_eq!(post, vec!["gemma4-tool-call-fix"]);

        let (pre, post) = Config::resolve_processors(&rules, "llama3:8b", "local");
        assert!(pre.is_empty());
        assert!(post.is_empty());
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
