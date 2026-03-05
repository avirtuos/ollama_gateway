#![allow(dead_code)]
use serde::{Deserialize, Serialize};

/// Minimal representation of an Ollama chat request.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Minimal representation of an Ollama generate request.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Minimal representation of an Ollama embed request.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EmbedRequest {
    pub model: String,
    pub input: serde_json::Value,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Usage info from a completed Ollama response.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct OllamaUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub eval_count: Option<u64>,
}

/// Non-streaming chat response.
#[derive(Debug, Deserialize, Clone)]
pub struct ChatResponse {
    pub model: Option<String>,
    pub message: Option<serde_json::Value>,
    pub done: Option<bool>,
    pub prompt_eval_count: Option<u64>,
    pub eval_count: Option<u64>,
}

/// Non-streaming generate response.
#[derive(Debug, Deserialize, Clone)]
pub struct GenerateResponse {
    pub model: Option<String>,
    pub response: Option<String>,
    pub done: Option<bool>,
    pub prompt_eval_count: Option<u64>,
    pub eval_count: Option<u64>,
}

/// A single streaming chunk from Ollama.
#[derive(Debug, Deserialize, Clone)]
pub struct StreamChunk {
    pub model: Option<String>,
    pub message: Option<serde_json::Value>,
    pub response: Option<String>,
    pub done: Option<bool>,
    pub prompt_eval_count: Option<u64>,
    pub eval_count: Option<u64>,
}

impl StreamChunk {
    pub fn is_done(&self) -> bool {
        self.done.unwrap_or(false)
    }
}

/// Extract the text output from a chat response message.
pub fn extract_chat_output(message: &serde_json::Value) -> String {
    message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

/// Determine if a request body has streaming enabled.
/// Ollama defaults to stream=true for chat/generate.
pub fn is_streaming(body: &serde_json::Value) -> bool {
    body.get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_streaming_default_true() {
        let body = serde_json::json!({ "model": "llama3" });
        assert!(is_streaming(&body));
    }

    #[test]
    fn test_is_streaming_explicit_false() {
        let body = serde_json::json!({ "model": "llama3", "stream": false });
        assert!(!is_streaming(&body));
    }

    #[test]
    fn test_is_streaming_explicit_true() {
        let body = serde_json::json!({ "model": "llama3", "stream": true });
        assert!(is_streaming(&body));
    }

    #[test]
    fn test_extract_chat_output() {
        let msg = serde_json::json!({ "role": "assistant", "content": "Hello!" });
        assert_eq!(extract_chat_output(&msg), "Hello!");
    }
}
