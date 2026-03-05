use chrono::{DateTime, Utc};
use langfuse_client_base::models::{
    CreateGenerationBody, IngestionEvent, IngestionEventOneOf, IngestionEventOneOf4,
    IngestionUsage, ModelUsageUnit, TraceBody, Usage,
    ingestion_event_one_of::Type as TraceType,
    ingestion_event_one_of_4::Type as GenerationType,
};
use uuid::Uuid;

/// A complete LLM call event ready to be sent to Langfuse.
#[derive(Debug, Clone)]
pub struct LangfuseEvent {
    pub trace_id: String,
    pub generation_id: String,
    pub app_name: String,
    pub model: String,
    pub endpoint: String,
    pub input: serde_json::Value,
    pub output: serde_json::Value,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}

impl LangfuseEvent {
    pub fn into_ingestion_events(self) -> Vec<IngestionEvent> {
        let timestamp = self.start_time.to_rfc3339();

        let trace = IngestionEvent::IngestionEventOneOf(Box::new(IngestionEventOneOf::new(
            Uuid::new_v4().to_string(),
            timestamp.clone(),
            TraceBody {
                id: Some(Some(self.trace_id.clone())),
                name: Some(Some(format!("{} - {}", self.app_name, self.endpoint))),
                user_id: Some(Some(self.app_name.clone())),
                tags: Some(Some(vec![self.app_name.clone(), self.endpoint.clone()])),
                timestamp: Some(Some(timestamp.clone())),
                ..Default::default()
            },
            TraceType::TraceCreate,
        )));

        let usage = IngestionUsage::Usage(Box::new(Usage {
            input: self.prompt_tokens.map(|t| Some(t as i32)),
            output: self.completion_tokens.map(|t| Some(t as i32)),
            unit: Some(ModelUsageUnit::Tokens),
            ..Default::default()
        }));

        let generation = IngestionEvent::IngestionEventOneOf4(Box::new(IngestionEventOneOf4::new(
            Uuid::new_v4().to_string(),
            timestamp.clone(),
            CreateGenerationBody {
                id: Some(Some(self.generation_id)),
                trace_id: Some(Some(self.trace_id)),
                name: Some(Some(self.endpoint)),
                model: Some(Some(self.model)),
                start_time: Some(Some(self.start_time.to_rfc3339())),
                end_time: Some(Some(self.end_time.to_rfc3339())),
                input: Some(Some(self.input)),
                output: Some(Some(self.output)),
                usage: Some(Box::new(usage)),
                ..Default::default()
            },
            GenerationType::GenerationCreate,
        )));

        vec![trace, generation]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use langfuse_client_base::models::IngestionBatchRequest;

    #[test]
    fn test_into_ingestion_events() {
        let event = LangfuseEvent {
            trace_id: "trace-1".to_string(),
            generation_id: "gen-1".to_string(),
            app_name: "test-app".to_string(),
            model: "llama3".to_string(),
            endpoint: "/api/chat".to_string(),
            input: serde_json::json!([{"role": "user", "content": "hi"}]),
            output: serde_json::json!("hello"),
            start_time: Utc::now(),
            end_time: Utc::now(),
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
        };
        let events = event.clone().into_ingestion_events();
        assert_eq!(events.len(), 2);

        // Verify the batch serializes correctly
        let batch = IngestionBatchRequest::new(events);
        let json = serde_json::to_value(&batch).unwrap();
        let gen_body = &json["batch"][1]["body"];
        assert_eq!(gen_body["model"], "llama3");
        assert_eq!(gen_body["traceId"], "trace-1");
        assert_eq!(gen_body["usage"]["unit"], "TOKENS");
        assert_eq!(gen_body["usage"]["input"], 10);
        assert_eq!(gen_body["usage"]["output"], 5);
    }
}
