//! Gemma 4 tool-call sanitizer.
//!
//! Gemma 4 models emit tool calls using a custom token format with `<|"|>` string
//! escaping and `<|tool_call>` / `<tool_call|>` delimiters.  When serving backends
//! (Ollama, llama.cpp, vLLM) lack a parser for this format, the raw special tokens
//! leak into the OpenAI-compatible JSON output, producing invalid JSON that
//! downstream callers cannot parse.
//!
//! This processor cleans up the leaked tokens in both requests (message history that
//! may contain prior bad responses) and responses (fresh model output).

use serde_json::Value;

use super::{Processor, ProcessorPhase};

pub struct Gemma4ToolCallFix;

impl Processor for Gemma4ToolCallFix {
    fn id(&self) -> &'static str {
        "gemma4-tool-call-fix"
    }

    fn description(&self) -> &'static str {
        "Fix malformed JSON in Gemma 4 tool calls — strips leaked special tokens \
         (<|\">, <|tool_call>, etc.) and repairs invalid escape sequences produced \
         when the serving backend lacks a native Gemma 4 tool-call parser"
    }

    fn phase(&self) -> ProcessorPhase {
        ProcessorPhase::Both
    }

    fn process_request(&self, body: &mut Value) {
        // Clean up message history that may contain prior bad tool-call output.
        if let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) {
            for msg in messages.iter_mut() {
                sanitize_message(msg);
            }
        }
    }

    fn process_response(&self, body: &mut Value) {
        // Non-streaming: clean the top-level response message.
        if let Some(choices) = body.get_mut("choices").and_then(|c| c.as_array_mut()) {
            for choice in choices.iter_mut() {
                if let Some(msg) = choice.get_mut("message") {
                    sanitize_message(msg);
                }
                if let Some(delta) = choice.get_mut("delta") {
                    sanitize_message(delta);
                }
            }
        }
        // Ollama native /api/chat format
        if let Some(msg) = body.get_mut("message") {
            sanitize_message(msg);
        }
    }

    fn process_response_chunk(&self, chunk: &mut Value) {
        // Streaming chunks: clean deltas
        if let Some(choices) = chunk.get_mut("choices").and_then(|c| c.as_array_mut()) {
            for choice in choices.iter_mut() {
                if let Some(delta) = choice.get_mut("delta") {
                    sanitize_message(delta);
                }
            }
        }
        // Ollama native streaming
        if let Some(msg) = chunk.get_mut("message") {
            sanitize_message(msg);
        }
    }
}

/// Clean a single message object (assistant message with tool_calls).
fn sanitize_message(msg: &mut Value) {
    if let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut()) {
        for tc in tool_calls.iter_mut() {
            sanitize_tool_call(tc);
        }
    }
    // Also clean content if it contains leaked tool-call delimiters
    if let Some(content) = msg.get_mut("content").and_then(|c| c.as_str()).map(|s| s.to_string()) {
        if content.contains("<|") || content.contains("|>") {
            let cleaned = sanitize_string(&content);
            if cleaned != content {
                *msg.get_mut("content").unwrap() = Value::String(cleaned);
            }
        }
    }
}

/// Clean a single tool_call object.
fn sanitize_tool_call(tc: &mut Value) {
    let arguments = tc
        .get_mut("function")
        .and_then(|f| f.get_mut("arguments"));

    let Some(args_val) = arguments else { return };

    match args_val {
        Value::String(ref s) => {
            let cleaned = sanitize_tool_call_arguments(s);
            if cleaned != *s {
                *args_val = Value::String(cleaned);
            }
        }
        _ => {}
    }
}

/// Repair a tool-call arguments string that may contain Gemma 4 special tokens.
///
/// The model emits patterns like:
///   `<|"|>some value<|"|>` instead of `"some value"`
///   `<|tool_call>` / `<tool_call|>` delimiters
///   `<|\` prefix artifacts
///
/// This function strips those patterns and attempts to produce valid JSON.
fn sanitize_tool_call_arguments(raw: &str) -> String {
    let mut s = raw.to_string();

    // Remove tool-call block delimiters
    s = s.replace("<|tool_call>", "");
    s = s.replace("<tool_call|>", "");
    s = s.replace("<|tool_call|>", "");

    // Replace `<|"|>` → `"` (Gemma 4 string delimiter)
    s = s.replace("<|\"|>", "\"");

    // Handle partial patterns: `<|\"...<|\"|` (common in Ollama output)
    // The pattern `<|\"` is the opening delimiter, `<|\"|` is the closing.
    s = s.replace("<|\\\"|", "\"");
    s = s.replace("<|\\\"", "\"");

    // Handle `<|"` and `"|>` variants
    s = s.replace("<|\"", "\"");
    s = s.replace("\"|>", "\"");

    // Clean up `<|` and `|>` that may remain around values
    // Process `<|\` before `<|` to avoid partial matches
    s = s.replace("<|\\", "");
    s = s.replace("<|", "");
    s = s.replace("|>", "");

    // Remove stray `"|` and `|"` that remain from partial delimiters
    // e.g., `<|"value"|` where only outer delimiters were partially stripped.
    // `"|` at end of a value → `"` (closing quote)
    s = s.replace("\"|", "\"");
    // `|"` at start of a value → `"` (opening quote)
    s = s.replace("|\"", "\"");

    // Fix doubled quotes that may result from the above replacements.
    // After stripping Gemma tokens, we can end up with patterns like:
    //   "value""  (token was a closing delimiter that left an extra quote)
    //   ""value"  (token was an opening delimiter that left an extra quote)
    // Strategy: collapse any run of 2+ quotes into a single quote, EXCEPT
    // when it's an intentional empty string like `"key": ""` (quote-quote
    // preceded by `: ` or `:`).
    loop {
        // First collapse triple+ quotes
        if s.contains("\"\"\"") {
            s = s.replace("\"\"\"", "\"");
            continue;
        }
        // Then fix doubled quotes at value boundaries:
        //   `"",` `""}` `""]` → keep as empty string (these are valid)
        //   `"": ` → keep as empty key (valid)
        //   All other `""` → collapse to single `"`
        // We do this by checking what follows/precedes the `""`
        if let Some(pos) = s.find("\"\"") {
            let after = s.get(pos + 2..pos + 3).unwrap_or("");
            let before = if pos > 0 { s.get(pos - 1..pos).unwrap_or("") } else { "" };
            // Keep `""` if it looks like an empty string value: after `:` and before `,` `}` `]`
            let is_empty_string = (before == ":" || before == " ")
                && (after == "," || after == "}" || after == "]" || after.is_empty());
            if is_empty_string {
                break; // Legit empty string, stop collapsing
            }
            // Otherwise collapse this `""` to `"`
            s = format!("{}\"{}", &s[..pos], &s[pos + 2..]);
            continue;
        }
        break;
    }

    // If the result is valid JSON, return it. Otherwise try to repair common issues.
    if serde_json::from_str::<Value>(&s).is_ok() {
        return s;
    }

    // Try wrapping in braces if it looks like key-value pairs without them
    if !s.starts_with('{') && s.contains(':') {
        let wrapped = format!("{{{}}}", s);
        if serde_json::from_str::<Value>(&wrapped).is_ok() {
            return wrapped;
        }
    }

    // Return best-effort cleaned string even if not valid JSON
    s
}

/// Clean leaked special tokens from arbitrary string content.
fn sanitize_string(raw: &str) -> String {
    let mut s = raw.to_string();
    s = s.replace("<|tool_call>", "");
    s = s.replace("<tool_call|>", "");
    s = s.replace("<|tool_call|>", "");
    s = s.replace("<|\"|>", "\"");
    s = s.replace("<|\\\"", "\"");
    s = s.replace("<|\"", "\"");
    s = s.replace("\"|>", "\"");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_real_hermes_payload() {
        // Exact pattern from production: Hermes sends the Gemma 4 output as a JSON
        // string where `<|\"` is `<|\` after JSON unescaping.
        // The raw JSON string (before JSON parsing) looks like:
        //   "<|\\Setup working directory<|\\\"|\""
        // After JSON string unescaping, the Rust string is:
        //   <|\Setup working directory<|\"|"
        // But our processor sees the arguments as a String value already parsed by serde,
        // so we test with what serde gives us.
        let raw = r#"{"todos": [{"content": "<|\"Setup working directory<|\"|", "id": "<|\"setup<|\"|", "status": "<|\"in_progress<|\""}]}"#;
        // After JSON unescaping of the outer string, the inner content is:
        // <|"Setup working directory<|"|
        // which should clean to: Setup working directory
        let cleaned = sanitize_tool_call_arguments(raw);
        assert!(serde_json::from_str::<Value>(&cleaned).is_ok(), "Result was not valid JSON: {}", cleaned);
    }

    #[test]
    fn test_clean_backslash_pipe_pattern() {
        // Pattern from the actual gateway logs: <|\value<|\"|
        // In a JSON string context, the backslash is escaped, so the Rust string has literal <|\
        let raw = r#"{"content": "<|\hello<|\"|", "id": "<|\test<|\"|"}"#;
        let cleaned = sanitize_tool_call_arguments(raw);
        assert!(serde_json::from_str::<Value>(&cleaned).is_ok(), "Result was not valid JSON: {}", cleaned);
    }

    #[test]
    fn test_clean_gemma4_pipe_delimiters() {
        // Gemma 4 uses <|"|> as string delimiters instead of quotes.
        // Inside JSON arguments string, the raw bytes look like this
        // (after JSON string unescaping):
        let raw = r#"{"todos": [{"content": <|"|>Clone the repo<|"|>, "id": <|"|>clone<|"|>, "status": <|"|>in_progress<|"|>}]}"#;
        let cleaned = sanitize_tool_call_arguments(raw);
        assert!(serde_json::from_str::<Value>(&cleaned).is_ok(), "Result was not valid JSON: {}", cleaned);
    }

    #[test]
    fn test_clean_gemma4_partial_delimiters() {
        // Partial delimiter pattern from Ollama where <|"  and  "|  appear
        let raw = r#"{"content": <|"hello"|}"#;
        let cleaned = sanitize_tool_call_arguments(raw);
        assert!(serde_json::from_str::<Value>(&cleaned).is_ok(), "Result was not valid JSON: {}", cleaned);
    }

    #[test]
    fn test_clean_pipe_delimited_values() {
        let raw = r#"{"items": [<|"|>light<|"|>, <|"|>dark<|"|>]}"#;
        let cleaned = sanitize_tool_call_arguments(raw);
        assert!(cleaned.contains("\"light\""), "Expected cleaned light, got: {}", cleaned);
        assert!(cleaned.contains("\"dark\""), "Expected cleaned dark, got: {}", cleaned);
    }

    #[test]
    fn test_already_valid_json_unchanged() {
        let valid = r#"{"command": "ls -la"}"#;
        let cleaned = sanitize_tool_call_arguments(valid);
        assert_eq!(cleaned, valid);
    }

    #[test]
    fn test_tool_call_delimiters_stripped() {
        let raw = r#"<|tool_call>{"fn": "test"}<tool_call|>"#;
        let cleaned = sanitize_tool_call_arguments(raw);
        assert_eq!(cleaned, r#"{"fn": "test"}"#);
    }

    #[test]
    fn test_sanitize_message_cleans_tool_calls() {
        // Simulate what the serving backend puts in the arguments string:
        // The raw arguments JSON has <|"|> instead of quotes around string values.
        let args_raw = r#"{"text": <|"|>hello<|"|>}"#;
        let mut msg = serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": "todo",
                    "arguments": args_raw,
                }
            }]
        });
        sanitize_message(&mut msg);
        let args = msg["tool_calls"][0]["function"]["arguments"].as_str().unwrap();
        assert!(serde_json::from_str::<Value>(args).is_ok(), "Args not valid JSON: {}", args);
        let parsed: Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["text"], "hello");
    }
}
