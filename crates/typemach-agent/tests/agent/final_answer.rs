use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::time::sleep;
use typemach::CheckpointSaver;

type Event = RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>;

#[derive(Clone, Default)]
struct StreamingFinal {
    calls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<ModelRequest>>>,
}

#[async_trait]
impl AgentModel for StreamingFinal {
    async fn next_step(
        &self,
        request: ModelRequest,
        stream: ModelStream,
    ) -> Result<ModelResponse, AgentError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.requests.lock().expect("requests lock").push(request);
        let message = message_item(
            "provider-message-1",
            0,
            AssistantMessagePhase::FinalAnswer,
            "The answer is 42.",
        );
        emit_message(&stream, &message, &["The answer ", "is 42."])?;
        Ok(ModelResponse {
            assistant_messages: vec![message],
            stop_reason: Some(StopReason::EndTurn),
            ..ModelResponse::default()
        })
    }
}

#[tokio::test]
async fn terminal_output_is_generated_once_and_promoted_in_place() {
    let model = StreamingFinal::default();
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("What was the order count?")],
            context: Value::Null,
            budget: AgentBudget {
                max_model_turns: 1,
                max_tool_calls: 4,
            },
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
                signal:
                    AgentSignal::AssistantMessageDelta {
                        message_id, delta, ..
                    },
            } => Some((message_id.as_str(), delta.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    let done = events.iter().find_map(|event| match event {
        RunStreamEvent::Signal {
            signal: AgentSignal::AssistantMessageDone { message_id, phase },
        } => Some((message_id.as_str(), *phase)),
        _ => None,
    });
    let output = completed(&events);

    assert_eq!(model.calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        deltas.iter().map(|(_, delta)| *delta).collect::<String>(),
        output.answer
    );
    assert_eq!(output.answer, "The answer is 42.");
    assert_eq!(
        done,
        Some((deltas[0].0, AssistantMessagePhase::FinalAnswer))
    );
    {
        let requests = model.requests.lock().expect("requests lock");
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].tool_choice,
            Some(typemach_agent::ToolChoice::Auto)
        );
        assert!(!requests[0].tools.is_empty());
    }

    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
    assert_eq!(state.answer, output.answer);
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

fn metric_call() -> ModelResponse {
    ModelResponse {
        stop_reason: Some(StopReason::ToolUse),
        ..tool_response(
            "Checking. ",
            vec![ToolUse {
                id: "tool-1".to_string(),
                name: "metric_point".to_string(),
                input: json!({ "metric": "orders" }),
                raw: None,
            }],
        )
    }
}

#[tokio::test]
async fn tool_call_followup_remains_tool_capable_and_commits_its_text() {
    let model = ScriptedModel::new([
        metric_call(),
        ModelResponse {
            stop_reason: Some(StopReason::EndTurn),
            ..final_response("There were 42 orders.")
        },
    ]);
    let tools = CountingTools::default();
    let calls = tools.calls.clone();
    let runner = build_agent_runner(MemorySaver::default(), model.clone(), tools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("How many orders?")],
            context: Value::Null,
            budget: AgentBudget {
                max_model_turns: 2,
                max_tool_calls: 2,
            },
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(completed(&events).answer, "There were 42 orders.");
    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|request| {
        request.tool_choice == Some(typemach_agent::ToolChoice::Auto)
            && request.tools.iter().any(|tool| tool.name == "metric_point")
    }));
    assert!(requests[1].messages.iter().any(|message| matches!(
        message,
        AgentMessage::User { content }
            if content.iter().any(|block| matches!(
                block,
                ContentBlock::ToolResult(result) if result.tool_use_id == "tool-1"
            ))
    )));
}

#[derive(Clone, Copy)]
enum AbortCase {
    MaxTokens,
    MaxTokensToolCalls,
    Refusal,
    Protocol,
    Provider,
}

#[derive(Clone, Copy)]
struct AbortModel(AbortCase);

#[async_trait]
impl AgentModel for AbortModel {
    async fn next_step(
        &self,
        _request: ModelRequest,
        stream: ModelStream,
    ) -> Result<ModelResponse, AgentError> {
        emit_pending(&stream, "pending-message", "uncommitted candidate")?;
        match self.0 {
            AbortCase::MaxTokens => Ok(ModelResponse {
                stop_reason: Some(StopReason::MaxTokens),
                ..ModelResponse::default()
            }),
            AbortCase::MaxTokensToolCalls => Ok(ModelResponse {
                tool_calls: vec![ToolUse {
                    id: "truncated-tool".to_string(),
                    name: "metric_point".to_string(),
                    input: json!({}),
                    raw: None,
                }],
                stop_reason: Some(StopReason::MaxTokens),
                ..ModelResponse::default()
            }),
            AbortCase::Refusal => Ok(ModelResponse {
                stop_reason: Some(StopReason::Refusal),
                ..ModelResponse::default()
            }),
            AbortCase::Protocol => Ok(ModelResponse {
                stop_reason: Some(StopReason::EndTurn),
                ..ModelResponse::default()
            }),
            AbortCase::Provider => Err(AgentError::Model("provider failed".to_string())),
        }
    }
}

#[tokio::test]
async fn aborted_candidates_never_complete_or_persist() {
    for case in [
        AbortCase::MaxTokens,
        AbortCase::MaxTokensToolCalls,
        AbortCase::Refusal,
        AbortCase::Protocol,
        AbortCase::Provider,
    ] {
        let runner = build_agent_runner(
            MemorySaver::default(),
            AbortModel(case),
            FakeTools,
            AllowAllTools,
        );
        let events = collect(runner.stream(
            request(AgentRunInput {
                messages: vec![AgentMessage::user_text("Answer")],
                context: Value::Null,
                budget: AgentBudget::default(),
                human_input: None,
                system_suffix: None,
            }),
            StreamConfig::default(),
        ))
        .await;

        assert!(
            events
                .iter()
                .any(|event| matches!(event, RunStreamEvent::Failed { .. }))
        );
        assert!(!events.iter().any(|event| matches!(
            event,
            RunStreamEvent::Completed { .. }
                | RunStreamEvent::Signal {
                    signal: AgentSignal::AssistantMessageDone {
                        phase: AssistantMessagePhase::FinalAnswer,
                        ..
                    },
                }
        )));
        assert!(!events.iter().any(|event| matches!(
            event,
            RunStreamEvent::Signal {
                signal: AgentSignal::ToolStarted { .. },
            }
        )));
        let checkpoint = runner
            .checkpointer()
            .load("thread-1")
            .await
            .expect("load checkpoint")
            .expect("checkpoint");
        let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
        assert!(state.answer.is_empty());
    }

    let model = ScriptedModel::new(Vec::<ModelResponse>::new());
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Answer")],
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
    assert!(model.requests().is_empty());
    assert!(
        events
            .iter()
            .any(|event| matches!(event, RunStreamEvent::Failed { .. }))
    );
}

#[tokio::test]
async fn empty_terminal_text_fails_without_finalizing() {
    let model = ScriptedModel::new([ModelResponse {
        stop_reason: Some(StopReason::EndTurn),
        ..final_response(" \n")
    }]);
    let runner = build_agent_runner(MemorySaver::default(), model, FakeTools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Answer")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

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
struct RecoveringModel {
    calls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<ModelRequest>>>,
}

#[async_trait]
impl AgentModel for RecoveringModel {
    async fn next_step(
        &self,
        request: ModelRequest,
        stream: ModelStream,
    ) -> Result<ModelResponse, AgentError> {
        self.requests.lock().expect("requests lock").push(request);
        match self.calls.fetch_add(1, Ordering::Relaxed) {
            0 => Ok(metric_call()),
            1 => {
                emit_pending(&stream, "failed-message", "discard this candidate")?;
                Err(AgentError::Model("provider failed".to_string()))
            }
            _ => Ok(ModelResponse {
                stop_reason: Some(StopReason::EndTurn),
                ..final_response("Recovered from saved evidence.")
            }),
        }
    }
}

#[tokio::test]
async fn retry_resumes_after_tool_dispatch_without_replaying_the_tool() {
    let model = RecoveringModel::default();
    let tools = CountingTools::default();
    let tool_calls = tools.calls.clone();
    let runner = build_agent_runner(MemorySaver::default(), model.clone(), tools, AllowAllTools);
    let run = request(AgentRunInput {
        messages: vec![AgentMessage::user_text("How many orders?")],
        context: Value::Null,
        budget: AgentBudget {
            max_model_turns: 4,
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
    assert_eq!(checkpoint.next_step, Some(json!("model_step")));
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
    assert!(state.answer.is_empty());
    assert_eq!(tool_calls.load(Ordering::Relaxed), 1);

    let second = collect(runner.stream(run, StreamConfig::default())).await;
    assert_eq!(completed(&second).answer, "Recovered from saved evidence.");
    assert_eq!(tool_calls.load(Ordering::Relaxed), 1);
    assert_eq!(model.calls.load(Ordering::Relaxed), 3);
    let requests = model.requests.lock().expect("requests lock");
    assert!(requests[2].messages.iter().any(|message| matches!(
        message,
        AgentMessage::User { content }
            if content.iter().any(|block| matches!(
                block,
                ContentBlock::ToolResult(result) if result.tool_use_id == "tool-1"
            ))
    )));
}

#[derive(Clone)]
struct BatchTools {
    annotations: ToolAnnotations,
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
}

impl BatchTools {
    fn new(annotations: ToolAnnotations) -> Self {
        Self {
            annotations,
            active: Arc::new(AtomicUsize::new(0)),
            max_active: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl ToolRegistry for BatchTools {
    async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok(vec![AgentToolSpec {
            name: "metric_point".to_string(),
            description: "read metric point".to_string(),
            input_schema: json!({ "type": "object" }),
            output_schema: Value::Null,
            metadata: Value::Null,
            annotations: self.annotations,
        }])
    }

    async fn call_tool(&self, request: ToolCallRequest) -> Result<ToolResult, AgentError> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        raise_max(&self.max_active, active);
        if request.tool_use.id == "tool-1" {
            sleep(Duration::from_millis(40)).await;
        } else {
            sleep(Duration::from_millis(10)).await;
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(ToolResult::ok(
            &request.tool_use,
            json!({ "id": request.tool_use.id }),
        ))
    }
}

fn raise_max(max: &AtomicUsize, value: usize) {
    let mut current = max.load(Ordering::SeqCst);
    while value > current {
        match max.compare_exchange(current, value, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return,
            Err(next) => current = next,
        }
    }
}

fn two_tool_model() -> ScriptedModel {
    ScriptedModel::new([
        ModelResponse {
            stop_reason: Some(StopReason::ToolUse),
            ..tool_response(
                "",
                ["tool-1", "tool-2"]
                    .into_iter()
                    .map(|id| ToolUse {
                        id: id.to_string(),
                        name: "metric_point".to_string(),
                        input: json!({}),
                        raw: None,
                    })
                    .collect(),
            )
        },
        ModelResponse {
            stop_reason: Some(StopReason::EndTurn),
            ..final_response("Done.")
        },
    ])
}

async fn run_batch(tools: BatchTools) -> (Vec<Event>, usize) {
    let max_active = tools.max_active.clone();
    let runner = build_agent_runner(
        MemorySaver::default(),
        two_tool_model(),
        tools,
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Read both metrics")],
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
    (events, max_active.load(Ordering::SeqCst))
}

#[tokio::test]
async fn read_only_batches_overlap_but_unsafe_batches_do_not() {
    let (read_events, read_max) = run_batch(BatchTools::new(ToolAnnotations::default())).await;
    let (write_events, write_max) = run_batch(BatchTools::new(ToolAnnotations {
        read_only: false,
        destructive: true,
        ..ToolAnnotations::default()
    }))
    .await;

    assert_eq!(read_max, 2);
    assert_eq!(write_max, 1);
    for events in [&read_events, &write_events] {
        assert_eq!(tool_result_ids(events), ["tool-1", "tool-2"]);
        assert_eq!(completed(events).answer, "Done.");
    }
}

#[tokio::test]
async fn oversized_tool_batch_aborts_without_partial_dispatch() {
    let model = ScriptedModel::new([ModelResponse {
        stop_reason: Some(StopReason::ToolUse),
        ..tool_response(
            "",
            ["metric-1", "metric-2"]
                .into_iter()
                .map(|id| ToolUse {
                    id: id.to_string(),
                    name: "metric_point".to_string(),
                    input: json!({}),
                    raw: None,
                })
                .collect(),
        )
    }]);
    let tools = CountingTools::default();
    let calls = tools.calls.clone();
    let runner = build_agent_runner(MemorySaver::default(), model, tools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Read both metrics")],
            context: Value::Null,
            budget: AgentBudget {
                max_model_turns: 2,
                max_tool_calls: 1,
            },
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, RunStreamEvent::Failed { .. }))
    );
    assert!(!events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::ToolStarted { .. } | AgentSignal::ToolCompleted { .. },
        }
    )));
}

fn tool_result_ids(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal: AgentSignal::ToolResult { tool_use_id, .. },
            } => Some(tool_use_id.clone()),
            _ => None,
        })
        .collect()
}
