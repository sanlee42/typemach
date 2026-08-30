use super::*;

type Event = RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>;

#[derive(Default)]
struct BuiltinTools;

#[async_trait]
impl ToolRegistry for BuiltinTools {
    async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok(vec![
            AgentToolSpec {
                name: "ask_user".to_string(),
                description: "ask user".to_string(),
                input_schema: json!({ "type": "object" }),
                output_schema: Value::Null,
                metadata: Value::Null,
                annotations: ToolAnnotations {
                    terminal: true,
                    ..ToolAnnotations::default()
                },
            },
            AgentToolSpec {
                name: "emit_artifact".to_string(),
                description: "emit artifact".to_string(),
                input_schema: json!({ "type": "object" }),
                output_schema: Value::Null,
                metadata: Value::Null,
                annotations: ToolAnnotations::default(),
            },
        ])
    }

    async fn call_tool(&self, _request: ToolCallRequest) -> Result<ToolResult, AgentError> {
        Err(AgentError::Tool(
            "built-in tools must not reach the registry".to_string(),
        ))
    }
}

#[tokio::test]
async fn terminal_annotated_invalid_ask_is_paired_before_the_model_corrects_it() {
    let model = ScriptedModel::new([
        ModelResponse {
            outcome: Some(ModelOutcome::ToolCalls {
                calls: vec![ToolUse {
                    id: "ask-bad".to_string(),
                    name: "ask_user".to_string(),
                    input: json!({ "question": " " }),
                    raw: None,
                }],
            }),
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse {
            outcome: Some(ModelOutcome::ToolCalls {
                calls: vec![ToolUse {
                    id: "ask-good".to_string(),
                    name: "ask_user".to_string(),
                    input: json!({ "question": "Which date?" }),
                    raw: None,
                }],
            }),
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse {
            outcome: Some(ModelOutcome::FinalAnswer {
                text: "Order count is 42.".to_string(),
            }),
            stop_reason: Some(StopReason::EndTurn),
            ..ModelResponse::default()
        },
    ]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        BuiltinTools,
        AllowAllTools,
    );
    let first = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Show order count")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert_error_lifecycle(&first, "ask-bad");
    assert!(first.iter().any(|event| matches!(
        event,
        RunStreamEvent::Interrupted { interrupt, .. }
            if interrupt.tool_use_id == "ask-good"
    )));

    let resume = RunRequest {
        command: RunCommand::Resume,
        input: AgentRunInput {
            messages: Vec::new(),
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: Some(HumanInputAnswer {
                tool_use_id: "ask-good".to_string(),
                answer: "2026-06-08".to_string(),
            }),
            system_suffix: None,
        },
        ..request(AgentRunInput {
            messages: Vec::new(),
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        })
    };
    let second = collect(runner.stream(resume, StreamConfig::default())).await;
    let output = completed(&second);
    assert_eq!(output.answer, "Order count is 42.");
    let requests = model.requests();
    assert_eq!(paired_results(&requests[1].messages, "ask-bad"), (1, 1));
    assert_eq!(requests.len(), 3);
}

#[tokio::test]
async fn invalid_artifacts_are_paired_before_the_model_corrects_them() {
    let model = ScriptedModel::new([
        ModelResponse {
            outcome: Some(ModelOutcome::ToolCalls {
                calls: vec![ToolUse {
                    id: "artifact-missing-type".to_string(),
                    name: "emit_artifact".to_string(),
                    input: json!({ "title": "Review", "content": "Review body" }),
                    raw: None,
                }],
            }),
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse {
            outcome: Some(ModelOutcome::ToolCalls {
                calls: vec![ToolUse {
                    id: "artifact-invalid-type".to_string(),
                    name: "emit_artifact".to_string(),
                    input: json!({
                        "title": "Review",
                        "type": "chart",
                        "content": "Review body"
                    }),
                    raw: None,
                }],
            }),
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse {
            outcome: Some(ModelOutcome::ToolCalls {
                calls: vec![ToolUse {
                    id: "artifact-invalid-source".to_string(),
                    name: "emit_artifact".to_string(),
                    input: json!({
                        "title": "Review",
                        "type": "markdown",
                        "content": "Review body",
                        "source": 42
                    }),
                    raw: None,
                }],
            }),
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse {
            outcome: Some(ModelOutcome::ToolCalls {
                calls: vec![ToolUse {
                    id: "artifact-good".to_string(),
                    name: "emit_artifact".to_string(),
                    input: json!({
                        "title": "Review",
                        "type": "table",
                        "content": "Review body"
                    }),
                    raw: None,
                }],
            }),
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse {
            outcome: Some(ModelOutcome::FinalAnswer {
                text: "Review created.".to_string(),
            }),
            stop_reason: Some(StopReason::EndTurn),
            ..ModelResponse::default()
        },
    ]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        BuiltinTools,
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Create a review")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    for tool_use_id in [
        "artifact-missing-type",
        "artifact-invalid-type",
        "artifact-invalid-source",
    ] {
        assert_error_lifecycle(&events, tool_use_id);
    }
    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::Artifact { artifact },
        } if artifact.tool_use_id == "artifact-good" && artifact.kind == "table"
    )));
    let output = completed(&events);
    assert_eq!(output.answer, "Review created.");
    let requests = model.requests();
    for tool_use_id in [
        "artifact-missing-type",
        "artifact-invalid-type",
        "artifact-invalid-source",
    ] {
        assert_eq!(paired_results(&requests[4].messages, tool_use_id), (1, 1));
    }
    assert_eq!(requests.len(), 5);
}

fn assert_error_lifecycle(events: &[Event], tool_use_id: &str) {
    let lifecycle = events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal:
                    AgentSignal::ToolStarted {
                        tool_use_id: id, ..
                    },
            } if id == tool_use_id => Some(("started", false)),
            RunStreamEvent::Signal {
                signal:
                    AgentSignal::ToolResult {
                        tool_use_id: id,
                        is_error,
                        ..
                    },
            } if id == tool_use_id => Some(("result", *is_error)),
            RunStreamEvent::Signal {
                signal:
                    AgentSignal::ToolCompleted {
                        tool_use_id: id,
                        is_error,
                        ..
                    },
            } if id == tool_use_id => Some(("completed", *is_error)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle,
        [("started", false), ("result", true), ("completed", true)]
    );
}

fn paired_results(messages: &[AgentMessage], tool_use_id: &str) -> (usize, usize) {
    messages
        .iter()
        .flat_map(|message| match message {
            AgentMessage::User { content } | AgentMessage::Assistant { content } => content,
        })
        .filter_map(|block| match block {
            ContentBlock::ToolResult(result) if result.tool_use_id == tool_use_id => Some(result),
            _ => None,
        })
        .fold((0, 0), |(count, errors), result| {
            (count + 1, errors + usize::from(result.is_error))
        })
}
