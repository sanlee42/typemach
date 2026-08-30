use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use typemach::CheckpointSaver;
use typemach_agent::AgentState;

#[tokio::test]
async fn answer_is_only_the_current_turn_final_model_step() {
    let model = ScriptedModel::new([
        ModelResponse {
            final_text: Some("I will inspect the metric first.".to_string()),
            tool_uses: vec![ToolUse {
                id: "tool-1".to_string(),
                name: "metric_point".to_string(),
                input: json!({ "metric_id": "paid_order_count", "ds": "2026-06-08" }),
                raw: None,
            }],
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse { ..respond() },
        ModelResponse {
            deltas: vec!["Order count is 42.".to_string()],
            stop_reason: Some(StopReason::EndTurn),
            ..ModelResponse::default()
        },
    ]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![
                AgentMessage::user_text("Previous question"),
                AgentMessage::assistant_text("Old answer must not enter the current answer."),
                AgentMessage::user_text("Yesterday's order count"),
            ],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::ToolResult { tool_use_id, .. },
        } if tool_use_id == "tool-1"
    )));
    assert!(!events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::AssistantDelta { .. },
        }
    )));
    assert_eq!(model.requests().len(), 3);
    assert_eq!(completed(&events).answer, "Order count is 42.");
    let requests = model.requests();
    assert_eq!(
        requests[0].tool_choice,
        Some(typemach_agent::ToolChoice::Required)
    );
    assert!(requests[0].tools.iter().any(|tool| tool.name == "respond"));
    assert_eq!(
        requests[1].tool_choice,
        Some(typemach_agent::ToolChoice::Required)
    );
    assert_eq!(
        requests[2].tool_choice,
        Some(typemach_agent::ToolChoice::None)
    );
    assert!(requests[2].tools.is_empty());
    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::FinalAnswerDelta { delta, index },
        } if delta == "Order count is 42." && *index == 0
    )));
}

#[tokio::test]
async fn terminal_tool_does_not_expose_or_commit_planning_draft() {
    let model = ScriptedModel::new([ModelResponse {
        deltas: vec!["Draft before terminal action.".to_string()],
        tool_uses: vec![ToolUse {
            id: "term-1".to_string(),
            name: "report".to_string(),
            input: json!({ "message": "Terminal message must not be appended." }),
            raw: None,
        }],
        stop_reason: Some(StopReason::ToolUse),
        ..ModelResponse::default()
    }]);
    let runner = build_agent_runner(MemorySaver::default(), model, FakeTools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Create a report")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    let deltas = events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal: AgentSignal::AssistantDelta { delta, .. },
            } => Some(delta.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(deltas.is_empty());
    assert!(completed(&events).answer.is_empty());

    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
    assert!(state.answer.is_empty());
}

#[derive(Clone, Default)]
struct CountingTools {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ToolRegistry for CountingTools {
    async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok(vec![AgentToolSpec {
            name: "metric_point".to_string(),
            description: "read metric point".to_string(),
            input_schema: json!({ "type": "object" }),
            output_schema: Value::Null,
            metadata: Value::Null,
            annotations: ToolAnnotations::default(),
        }])
    }

    async fn call_tool(&self, request: ToolCallRequest) -> Result<ToolResult, AgentError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(ToolResult::ok(&request.tool_use, json!({ "value": 42 })))
    }
}

fn paired_errors(messages: &[AgentMessage], tool_use_id: &str) -> (usize, usize) {
    messages
        .iter()
        .flat_map(|message| match message {
            AgentMessage::User { content } | AgentMessage::Assistant { content } => content,
        })
        .fold((0, 0), |(calls, errors), block| match block {
            ContentBlock::ToolUse(tool_use) if tool_use.id == tool_use_id => (calls + 1, errors),
            ContentBlock::ToolResult(result)
                if result.tool_use_id == tool_use_id && result.is_error =>
            {
                (calls, errors + 1)
            }
            _ => (calls, errors),
        })
}

#[tokio::test]
async fn rejected_respond_batches_are_paired_without_dispatch_and_recover() {
    let model = ScriptedModel::new([
        ModelResponse {
            tool_uses: vec![ToolUse {
                id: "respond-args".to_string(),
                name: "respond".to_string(),
                input: json!({ "unexpected": true }),
                raw: None,
            }],
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse {
            tool_uses: vec![
                ToolUse {
                    id: "respond-bad".to_string(),
                    name: "respond".to_string(),
                    input: json!({}),
                    raw: None,
                },
                ToolUse {
                    id: "metric-bad".to_string(),
                    name: "metric_point".to_string(),
                    input: json!({ "metric_id": "paid_order_count" }),
                    raw: None,
                },
            ],
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        respond(),
        ModelResponse {
            deltas: vec!["Recovered final answer.".to_string()],
            ..ModelResponse::default()
        },
    ]);
    let tools = CountingTools::default();
    let calls = tools.calls.clone();
    let runner = build_agent_runner(MemorySaver::default(), model.clone(), tools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Read the metric")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let requests = model.requests();
    for tool_use_id in ["respond-args", "respond-bad", "metric-bad"] {
        assert_eq!(paired_errors(&requests[2].messages, tool_use_id), (1, 1));
        assert!(!events.iter().any(|event| matches!(
            event,
            RunStreamEvent::Signal {
                signal: AgentSignal::ToolStarted { tool_use_id: id, .. }
                    | AgentSignal::ToolCompleted { tool_use_id: id, .. },
            } if id == tool_use_id
        )));
    }
    assert_eq!(completed(&events).answer, "Recovered final answer.");
    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
    assert_eq!(state.tool_calls, 0);
}

#[tokio::test]
async fn respond_on_last_planning_turn_still_gets_one_final_call() {
    let model = ScriptedModel::new([
        respond(),
        ModelResponse {
            deltas: vec!["Final ".to_string(), "answer".to_string()],
            ..ModelResponse::default()
        },
    ]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Answer directly")],
            context: Value::Null,
            budget: AgentBudget {
                max_model_turns: 1,
                max_tool_calls: 1,
            },
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert_eq!(completed(&events).answer, "Final answer");
    assert_eq!(model.requests().len(), 2);
}

#[tokio::test]
async fn bare_planning_prose_is_a_protocol_failure_not_an_answer() {
    let model = ScriptedModel::new([ModelResponse {
        deltas: vec!["INTERNAL_DRAFT".to_string()],
        ..ModelResponse::default()
    }]);
    let runner = build_agent_runner(MemorySaver::default(), model, FakeTools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Answer directly")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Failed { error }
            if error.to_string().contains("planning step did not call a tool")
    )));
    assert!(!events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::AssistantDelta { .. } | AgentSignal::FinalAnswerDelta { .. },
        }
    )));
}

#[tokio::test]
async fn final_text_without_stream_deltas_is_emitted_and_persisted() {
    let model = ScriptedModel::new([
        respond(),
        ModelResponse {
            final_text: Some("The order count is 42.".to_string()),
            ..ModelResponse::default()
        },
    ]);
    let runner = build_agent_runner(MemorySaver::default(), model, FakeTools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("What was yesterday's order count?")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::FinalAnswerDelta { delta, .. },
        } if delta == "The order count is 42."
    )));
    let output = completed(&events);
    assert_eq!(output.answer, "The order count is 42.");
    assert!(matches!(
        output.messages.last(),
        Some(AgentMessage::Assistant { content })
            if content == &vec![ContentBlock::Text {
                text: "The order count is 42.".to_string()
            }]
    ));
}

#[tokio::test]
async fn valid_artifact_is_nonterminal_before_final_answer() {
    let model = ScriptedModel::new([
        ModelResponse {
            tool_uses: vec![ToolUse {
                id: "artifact-1".to_string(),
                name: "emit_artifact".to_string(),
                input: json!({
                    "title": "Review",
                    "type": "markdown",
                    "content": "Review body"
                }),
                raw: None,
            }],
            ..ModelResponse::default()
        },
        respond(),
        ModelResponse {
            final_text: Some("The review is ready.".to_string()),
            ..ModelResponse::default()
        },
    ]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
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

    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::Artifact { artifact },
        } if artifact.tool_use_id == "artifact-1" && artifact.kind == "markdown"
    )));
    assert_eq!(completed(&events).finish_reason, FinishReason::Stop);
    assert_eq!(completed(&events).answer, "The review is ready.");
    assert_eq!(model.requests().len(), 3);
}
