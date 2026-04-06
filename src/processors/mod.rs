pub mod gemma4;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Whether a processor operates on requests, responses, or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProcessorPhase {
    Pre,
    Post,
    Both,
}

/// A built-in processor that can transform request/response bodies.
pub trait Processor: Send + Sync {
    fn id(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn phase(&self) -> ProcessorPhase;

    /// Transform the request body before it is sent upstream.
    fn process_request(&self, body: &mut Value);

    /// Transform a non-streaming response body before it is returned to the client.
    fn process_response(&self, body: &mut Value);

    /// Transform a single streaming response chunk.
    fn process_response_chunk(&self, chunk: &mut Value);

    /// Attempt to repair raw response text that is not valid JSON.
    ///
    /// Called when the response body fails JSON parsing.  Processors that know
    /// about model-specific token leakage (e.g. Gemma 4 `<|"|>` delimiters)
    /// can strip those tokens from the raw text so that a subsequent JSON parse
    /// may succeed.  Returns `Some(repaired)` if the processor made changes,
    /// `None` otherwise.
    fn repair_raw_response(&self, _raw: &str) -> Option<String> {
        None
    }
}

/// Serializable metadata about a built-in processor (for the admin API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorInfo {
    pub id: String,
    pub description: String,
    pub phase: ProcessorPhase,
}

/// Registry of all available built-in processors.
pub struct ProcessorRegistry {
    processors: Vec<Box<dyn Processor>>,
}

impl ProcessorRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            processors: Vec::new(),
        };
        // Register all built-in processors
        registry.register(Box::new(gemma4::Gemma4ToolCallFix));
        registry
    }

    fn register(&mut self, processor: Box<dyn Processor>) {
        self.processors.push(processor);
    }

    /// List metadata for all available processors.
    pub fn list(&self) -> Vec<ProcessorInfo> {
        self.processors
            .iter()
            .map(|p| ProcessorInfo {
                id: p.id().to_string(),
                description: p.description().to_string(),
                phase: p.phase(),
            })
            .collect()
    }

    /// Look up a processor by id.
    pub fn get(&self, id: &str) -> Option<&dyn Processor> {
        self.processors.iter().find(|p| p.id() == id).map(|p| &**p)
    }

    /// Apply all matching preprocessors for the given ids to a request body.
    pub fn apply_preprocessors(&self, ids: &[String], body: &mut Value) {
        for id in ids {
            if let Some(p) = self.get(id) {
                match p.phase() {
                    ProcessorPhase::Pre | ProcessorPhase::Both => p.process_request(body),
                    ProcessorPhase::Post => {}
                }
            }
        }
    }

    /// Apply all matching postprocessors for the given ids to a response body.
    pub fn apply_postprocessors(&self, ids: &[String], body: &mut Value) {
        for id in ids {
            if let Some(p) = self.get(id) {
                match p.phase() {
                    ProcessorPhase::Post | ProcessorPhase::Both => p.process_response(body),
                    ProcessorPhase::Pre => {}
                }
            }
        }
    }

    /// Try to repair raw response text using matching postprocessors.
    ///
    /// Called when the response body is not valid JSON.  Each processor's
    /// `repair_raw_response` is tried in order; the first `Some` result wins.
    pub fn try_repair_raw(&self, ids: &[String], raw: &str) -> Option<String> {
        for id in ids {
            if let Some(p) = self.get(id) {
                match p.phase() {
                    ProcessorPhase::Post | ProcessorPhase::Both => {
                        if let Some(repaired) = p.repair_raw_response(raw) {
                            return Some(repaired);
                        }
                    }
                    ProcessorPhase::Pre => {}
                }
            }
        }
        None
    }

    /// Apply all matching postprocessors to a streaming chunk.
    pub fn apply_chunk_postprocessors(&self, ids: &[String], chunk: &mut Value) {
        for id in ids {
            if let Some(p) = self.get(id) {
                match p.phase() {
                    ProcessorPhase::Post | ProcessorPhase::Both => p.process_response_chunk(chunk),
                    ProcessorPhase::Pre => {}
                }
            }
        }
    }
}
