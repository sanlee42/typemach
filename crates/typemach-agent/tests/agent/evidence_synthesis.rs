use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use typemach_agent::ToolDisposition;

#[derive(Clone)]
struct SynthesizingTools;

#[async_trait]
impl ToolRegistry for SynthesizingTools {
    async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok(vec![AgentToolSpec {
            name: "converge_orders".to_string(),
            description: "converge authoritative order evidence".to_string(),
            input_schema: json!({ "type": "object" }),
            output_schema: Value::Null,
            metadata: Value::Null,
            annotations: ToolAnnotations::default(),
        }])
    }

    async fn call_tool(&self, request: ToolCallRequest) -> Result<ToolResult, AgentError> {
        ToolResult::ok(
            &request.tool_use,
            json!({ "raw_rows": [40, 2], "competing_total": 999 }),
        )
        .synthesize(json!({
            "answer": { "orders": 42 },
            "basis": "validated warehouse rows"
        }))
    }
}

fn synthesis_tool_turn(ids: &[&str]) -> ModelResponse {
    ModelResponse {
        stop_reason: Some(StopReason::ToolUse),
        ..tool_response(
            "",
            ids.iter()
                .map(|id| ToolUse {
                    id: (*id).to_string(),
                    name: "converge_orders".to_string(),
                    input: json!({}),
                    raw: None,
                })
                .collect(),
        )
    }
}

fn synthesis_input(request: Option<AgentMessage>) -> AgentRunInput {
    AgentRunInput {
        messages: vec![
            AgentMessage::user_text("Earlier question"),
            AgentMessage::assistant_text("Earlier answer"),
            AgentMessage::user_text("How many orders?\nRAW_PAGE_TOTAL=999"),
        ],
        synthesis_request: request,
        context: json!({ "shop_id": "shop-1" }),
        retained_results: Vec::new(),
        budget: AgentBudget {
            max_model_turns: 4,
            max_tool_calls: 3,
        },
        human_input: None,
        system_suffix: Some("Current shop: shop-1".to_string()),
    }
}

#[test]
fn synthesis_requires_successful_non_empty_object_evidence() {
    let tool_use = ToolUse {
        id: "tool-1".to_string(),
        name: "converge_orders".to_string(),
        input: Value::Null,
        raw: None,
    };

    assert!(
        ToolResult::error(&tool_use, "failed")
            .synthesize(json!({ "orders": 42 }))
            .is_err()
    );
    for unusable in [Value::Null, json!({}), json!([]), json!("evidence")] {
        assert!(
            ToolResult::ok(&tool_use, Value::Null)
                .synthesize(unusable)
                .is_err()
        );
    }
    let result = ToolResult::ok(&tool_use, Value::Null)
        .synthesize(json!({ "orders": 42 }))
        .expect("valid evidence");
    assert_eq!(
        result.disposition,
        ToolDisposition::Synthesize {
            evidence: json!({ "orders": 42 })
        }
    );
    assert_eq!(
        serde_json::to_value(&result.disposition).expect("serialize disposition"),
        json!({
            "type": "synthesize",
            "evidence": { "orders": 42 }
        })
    );
}

#[tokio::test]
async fn synthesis_uses_only_the_current_request_and_final_evidence() {
    let evidence = json!({
        "answer": { "orders": 42 },
        "basis": "validated warehouse rows"
    });
    let model = ScriptedModel::new([
        synthesis_tool_turn(&["tool-1"]),
        ModelResponse {
            stop_reason: Some(StopReason::EndTurn),
            ..final_response("There were 42 orders.")
        },
    ]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        SynthesizingTools,
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(synthesis_input(Some(AgentMessage::user_text(
            "How many orders?",
        )))),
        StreamConfig::default(),
    ))
    .await;

    assert_eq!(completed(&events).answer, "There were 42 orders.");
    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    let synthesis = &requests[1];
    assert!(synthesis.tools.is_empty());
    assert_eq!(
        synthesis.tool_choice,
        Some(typemach_agent::ToolChoice::None)
    );
    assert_eq!(synthesis.context, json!({ "shop_id": "shop-1" }));
    let suffix = synthesis.system_suffix.as_deref().expect("system suffix");
    assert!(suffix.starts_with("Current shop: shop-1\n\n"));
    assert!(suffix.contains("sole factual source"));
    assert!(suffix.contains("Do not recompute any values"));
    assert_eq!(synthesis.messages.len(), 1);
    let AgentMessage::User { content } = &synthesis.messages[0] else {
        panic!("synthesis message must be the user request")
    };
    assert_eq!(
        content,
        &vec![
            ContentBlock::Text {
                text: "How many orders?".to_string()
            },
            ContentBlock::Text {
                text: format!("\n\n[Final evidence]\n{}", evidence)
            }
        ]
    );
    let encoded = serde_json::to_string(&synthesis.messages).expect("messages");
    assert!(!encoded.contains("Earlier question"));
    assert!(!encoded.contains("RAW_PAGE_TOTAL"));
    assert!(!encoded.contains("raw_rows"));

    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
    assert_eq!(state.phase, typemach_agent::AgentPhase::Synthesis);
    assert_eq!(state.synthesis_evidence, Some(evidence));
    let history = serde_json::to_string(&state.messages).expect("history");
    assert!(history.contains("Earlier question"));
    assert!(history.contains("RAW_PAGE_TOTAL"));
    assert!(history.contains("raw_rows"));
}

#[tokio::test]
async fn synthesis_without_an_explicit_request_fails_without_fallback() {
    let model = ScriptedModel::new([synthesis_tool_turn(&["tool-1"])]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        SynthesizingTools,
        AllowAllTools,
    );
    let events =
        collect(runner.stream(request(synthesis_input(None)), StreamConfig::default())).await;

    assert_eq!(model.requests().len(), 1);
    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Failed { error }
            if error.to_string().contains("synthesis_request is required")
    )));
}

#[tokio::test]
async fn concurrent_synthesis_capsules_are_rejected_as_ambiguous() {
    let model = ScriptedModel::new([synthesis_tool_turn(&["tool-1", "tool-2"])]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model,
        SynthesizingTools,
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(synthesis_input(Some(AgentMessage::user_text(
            "How many orders?",
        )))),
        StreamConfig::default(),
    ))
    .await;

    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Failed { error }
            if error.to_string().contains("more than one evidence capsule")
    )));
}

#[tokio::test]
async fn sequential_synthesis_skips_and_closes_remaining_calls() {
    #[derive(Clone)]
    struct SequentialTools(Arc<AtomicUsize>);

    #[async_trait]
    impl ToolRegistry for SequentialTools {
        async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
            Ok(vec![AgentToolSpec {
                name: "converge_orders".to_string(),
                description: "converge authoritative order evidence".to_string(),
                input_schema: json!({ "type": "object" }),
                output_schema: Value::Null,
                metadata: Value::Null,
                annotations: ToolAnnotations {
                    read_only: false,
                    ..ToolAnnotations::default()
                },
            }])
        }

        async fn call_tool(&self, request: ToolCallRequest) -> Result<ToolResult, AgentError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            if request.tool_use.id == "tool-1" {
                ToolResult::ok(&request.tool_use, json!({ "rows": 2 }))
                    .synthesize(json!({ "answer": { "orders": 42 } }))
            } else {
                Ok(ToolResult::ok(
                    &request.tool_use,
                    json!({ "mutated": true }),
                ))
            }
        }
    }

    let model = ScriptedModel::new([
        synthesis_tool_turn(&["tool-1", "tool-2"]),
        ModelResponse {
            stop_reason: Some(StopReason::EndTurn),
            ..final_response("There were 42 orders.")
        },
    ]);
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = build_agent_runner(
        MemorySaver::default(),
        model,
        SequentialTools(Arc::clone(&calls)),
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(synthesis_input(Some(AgentMessage::user_text(
            "How many orders?",
        )))),
        StreamConfig::default(),
    ))
    .await;

    assert_eq!(completed(&events).answer, "There were 42 orders.");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
    assert_eq!(state.tool_calls, 1);
    assert!(state.pending_tools.is_empty());
    let skipped = state
        .messages
        .iter()
        .flat_map(|message| match message {
            AgentMessage::User { content } | AgentMessage::Assistant { content } => content,
        })
        .find_map(|block| match block {
            ContentBlock::ToolResult(result) if result.tool_use_id == "tool-2" => Some(result),
            _ => None,
        })
        .expect("skipped tool result");
    assert!(skipped.is_error);
    assert_eq!(skipped.content["error"]["code"], "synthesis_started");
}
