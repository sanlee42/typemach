use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use typemach::CheckpointSaver;
use typemach_agent::AgentState;

#[tokio::test]
async fn planning_tools_and_natural_completion_emit_only_the_final_answer() {
    let model = ScriptedModel::new([
        ModelResponse {
            outcome: Some(ModelOutcome::ToolCalls {
                calls: vec![ToolUse {
                    id: "tool-1".to_string(),
                    name: "metric_point".to_string(),
                    input: json!({ "metric_id": "paid_order_count", "ds": "2026-06-08" }),
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
            budget: AgentBudget {
                max_model_turns: 2,
                max_tool_calls: 4,
            },
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
    assert_eq!(model.requests().len(), 2);
    assert_eq!(completed(&events).answer, "Order count is 42.");
    let requests = model.requests();
    assert_eq!(
        requests[0].tool_choice,
        Some(typemach_agent::ToolChoice::Auto)
    );
    assert_eq!(
        requests[0]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["metric_point", "ask_user", "emit_artifact", "report"]
    );
    assert_eq!(
        requests[1].tool_choice,
        Some(typemach_agent::ToolChoice::Auto)
    );
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
        outcome: Some(ModelOutcome::ToolCalls {
            calls: vec![ToolUse {
                id: "term-1".to_string(),
                name: "report".to_string(),
                input: json!({ "message": "Terminal message must not be appended." }),
                raw: None,
            }],
        }),
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

#[derive(Clone, Default)]
struct FailFirstFinalModel {
    calls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<ModelRequest>>>,
}

#[async_trait]
impl AgentModel for FailFirstFinalModel {
    async fn next_step(
        &self,
        request: ModelRequest,
        _stream: ModelStream,
    ) -> Result<ModelResponse, AgentError> {
        self.requests.lock().expect("requests lock").push(request);
        match self.calls.fetch_add(1, Ordering::Relaxed) {
            0 => Err(AgentError::Model("final transport failed".to_string())),
            _ => Ok(ModelResponse {
                outcome: Some(ModelOutcome::FinalAnswer {
                    text: "Recovered final answer.".to_string(),
                }),
                stop_reason: Some(StopReason::EndTurn),
                ..ModelResponse::default()
            }),
        }
    }
}

#[tokio::test]
async fn retrying_the_same_run_resumes_final_without_replaying_planning() {
    let model = FailFirstFinalModel::default();
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
        AllowAllTools,
    );
    let run = request(AgentRunInput {
        messages: vec![AgentMessage::user_text("Answer directly")],
        context: Value::Null,
        budget: AgentBudget {
            max_model_turns: 0,
            max_tool_calls: 4,
        },
        human_input: None,
        system_suffix: None,
    });
    let first = collect(runner.stream(run.clone(), StreamConfig::default())).await;
    assert!(
        first
            .iter()
            .any(|event| matches!(event, RunStreamEvent::Failed { .. }))
    );
    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    assert_eq!(
        checkpoint.next_step,
        Some(serde_json::to_value(AgentStep::FinalAnswer).expect("serialize step"))
    );

    let second = collect(runner.stream(run, StreamConfig::default())).await;
    assert_eq!(completed(&second).answer, "Recovered final answer.");
    let requests = model.requests.lock().expect("requests lock");
    assert_eq!(requests.len(), 2);
    for request in requests.iter() {
        assert_eq!(request.tool_choice, Some(typemach_agent::ToolChoice::None));
        assert!(request.tools.is_empty());
    }
}

#[tokio::test]
async fn final_refusal_fails_without_a_completed_output() {
    let model = ScriptedModel::new([ModelResponse {
        outcome: Some(ModelOutcome::FinalAnswer {
            text: "Rejected text must stay private.".to_string(),
        }),
        stop_reason: Some(StopReason::Refusal),
        ..ModelResponse::default()
    }]);
    let runner = build_agent_runner(MemorySaver::default(), model, FakeTools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Answer directly")],
            context: Value::Null,
            budget: AgentBudget {
                max_model_turns: 0,
                max_tool_calls: 4,
            },
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert!(!events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::FinalAnswerDelta { .. },
        }
    )));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, RunStreamEvent::Failed { .. }))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, RunStreamEvent::Completed { .. }))
    );
}

#[tokio::test]
async fn reached_model_or_tool_budget_after_evidence_still_finalizes() {
    for budget in [
        AgentBudget {
            max_model_turns: 1,
            max_tool_calls: 2,
        },
        AgentBudget {
            max_model_turns: 4,
            max_tool_calls: 1,
        },
    ] {
        let model = ScriptedModel::new([
            ModelResponse {
                outcome: Some(ModelOutcome::ToolCalls {
                    calls: vec![ToolUse {
                        id: "metric-1".to_string(),
                        name: "metric_point".to_string(),
                        input: json!({}),
                        raw: None,
                    }],
                }),
                stop_reason: Some(StopReason::ToolUse),
                ..ModelResponse::default()
            },
            ModelResponse {
                outcome: Some(ModelOutcome::FinalAnswer {
                    text: "Budgeted answer.".to_string(),
                }),
                stop_reason: Some(StopReason::EndTurn),
                ..ModelResponse::default()
            },
        ]);
        let tools = CountingTools::default();
        let calls = tools.calls.clone();
        let runner =
            build_agent_runner(MemorySaver::default(), model.clone(), tools, AllowAllTools);
        let events = collect(runner.stream(
            request(AgentRunInput {
                messages: vec![AgentMessage::user_text("Read the metric")],
                context: Value::Null,
                budget,
                human_input: None,
                system_suffix: None,
            }),
            StreamConfig::default(),
        ))
        .await;

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(model.requests().len(), 2);
        assert_eq!(completed(&events).answer, "Budgeted answer.");
    }
}

#[tokio::test]
async fn oversized_tool_batch_is_not_partially_dispatched() {
    let model = ScriptedModel::new([
        ModelResponse {
            outcome: Some(ModelOutcome::ToolCalls {
                calls: ["metric-1", "metric-2"]
                    .into_iter()
                    .map(|id| ToolUse {
                        id: id.to_string(),
                        name: "metric_point".to_string(),
                        input: json!({}),
                        raw: None,
                    })
                    .collect(),
            }),
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse {
            outcome: Some(ModelOutcome::FinalAnswer {
                text: "No partial evidence.".to_string(),
            }),
            stop_reason: Some(StopReason::EndTurn),
            ..ModelResponse::default()
        },
    ]);
    let tools = CountingTools::default();
    let calls = tools.calls.clone();
    let runner = build_agent_runner(MemorySaver::default(), model, tools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Read both metrics")],
            context: Value::Null,
            budget: AgentBudget {
                max_model_turns: 4,
                max_tool_calls: 1,
            },
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(!events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::ToolStarted { .. } | AgentSignal::ToolCompleted { .. },
        }
    )));
    let output = completed(&events);
    assert_eq!(output.answer, "No partial evidence.");
    assert!(!output.messages.iter().any(|message| matches!(
        message,
        AgentMessage::Assistant { content }
            if content.iter().any(|block| matches!(block, ContentBlock::ToolUse(_)))
    )));
}

#[tokio::test]
async fn final_text_without_stream_deltas_is_emitted_and_persisted() {
    let model = ScriptedModel::new([ModelResponse {
        outcome: Some(ModelOutcome::FinalAnswer {
            text: "The order count is 42.".to_string(),
        }),
        ..ModelResponse::default()
    }]);
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
            outcome: Some(ModelOutcome::ToolCalls {
                calls: vec![ToolUse {
                    id: "artifact-1".to_string(),
                    name: "emit_artifact".to_string(),
                    input: json!({
                        "title": "Review",
                        "type": "markdown",
                        "content": "Review body"
                    }),
                    raw: None,
                }],
            }),
            ..ModelResponse::default()
        },
        ModelResponse {
            outcome: Some(ModelOutcome::FinalAnswer {
                text: "The review is ready.".to_string(),
            }),
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
    assert_eq!(model.requests().len(), 2);
}
