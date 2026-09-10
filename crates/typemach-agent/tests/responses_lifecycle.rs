use async_trait::async_trait;
use serde_json::{Value, json};
use typemach::{
    CheckpointSaver, MemorySaver, RunCommand, RunId, RunRequest, RunStreamEvent, RuntimeLimits,
    SessionId, StreamConfig, ThreadId,
};
use typemach_agent::{
    AgentBudget, AgentError, AgentEventReceiver, AgentMessage, AgentRunInput, AgentRunOutput,
    AgentSignal, AgentState, AgentStep, AgentToolSpec, AllowAllTools, Artifact, AskUserQuestion,
    AssistantMessagePhase, ConfiguredModel, ContentBlock, ToolAnnotations, ToolCallRequest,
    ToolRegistry, ToolResult, build_agent_runner,
};

#[path = "responses_lifecycle/fixtures.rs"]
mod fixtures;
use fixtures::{MockTurn, captured_bodies, spawn_server, sse};

#[tokio::test]
async fn provider_sse_to_agent_lifecycle_streams_and_persists_answer_once() {
    let (base_url, captured) = spawn_server(vec![
        MockTurn::ok(tool_reasoning_call_sse()),
        MockTurn::ok(final_answer_sse()),
    ])
    .await;
    let runner = build_agent_runner(
        MemorySaver::default(),
        model(base_url),
        FakeTools,
        AllowAllTools,
    );

    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("What was the order count?")],
            context: Value::Null,
            retained_results: Vec::new(),
            budget: AgentBudget {
                max_model_turns: 2,
                max_tool_calls: 4,
            },
            human_input: None,
            synthesis_request: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert_eq!(
        message_deltas(&events),
        [
            "Checking orders. ",
            "Summarizing. ",
            "The answer ",
            "is 42."
        ]
    );
    assert_eq!(
        message_done_phases(&events),
        [
            AssistantMessagePhase::Commentary,
            AssistantMessagePhase::Commentary,
            AssistantMessagePhase::FinalAnswer,
            AssistantMessagePhase::FinalAnswer
        ]
    );
    assert_eq!(
        message_stream_phases(&events),
        [
            AssistantMessagePhase::Commentary,
            AssistantMessagePhase::Commentary,
            AssistantMessagePhase::Commentary,
            AssistantMessagePhase::Commentary,
            AssistantMessagePhase::FinalAnswer,
            AssistantMessagePhase::FinalAnswer,
            AssistantMessagePhase::FinalAnswer,
            AssistantMessagePhase::FinalAnswer,
        ]
    );
    let completed = completed(&events);
    assert_eq!(completed.answer, "The answer is 42.");
    assert!(!completed.answer.contains("privately"));
    assert_eq!(
        assistant_texts(&completed.messages),
        vec!["Checking orders. ", "Summarizing. The answer is 42."]
    );
    let bodies = captured_bodies(&captured);
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0]["tool_choice"], "auto");
    assert_eq!(bodies[1]["tool_choice"], "none");
    assert!(bodies[1].get("tools").is_none());
    assert!(input_has_type(&bodies[1], "function_call"));
    assert!(input_has_type(&bodies[1], "function_call_output"));
    assert_ordered_input_types(&bodies[1], &["function_call", "function_call_output"]);
    let function_output = bodies[1]["input"]
        .as_array()
        .expect("input array")
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .and_then(|item| item["output"].as_str())
        .expect("function call output");
    assert_eq!(
        serde_json::from_str::<Value>(function_output).expect("tool result content"),
        json!({ "value": 42, "unit": "orders" })
    );
    assert!(!bodies[1].to_string().contains("Artifact-only review"));
    assert!(input_has_type(&bodies[1], "reasoning"));
    assert!(
        bodies[1]["input"]
            .as_array()
            .expect("input array")
            .iter()
            .any(|item| item["type"] == "message" && item["role"] == "assistant")
    );
    let assistant = bodies[1]["input"]
        .as_array()
        .expect("input array")
        .iter()
        .find(|item| item["id"] == "msg-commentary")
        .expect("provider assistant item retained in next request");
    assert_eq!(assistant["phase"], "commentary");
    assert_eq!(
        assistant["content"],
        json!([{ "type": "output_text", "text": "Checking orders. " }])
    );
    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
    assert_eq!(state.answer, completed.answer);
    assert_eq!(state.artifacts, completed.artifacts);
    assert_eq!(completed.artifacts.len(), 1);
    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::Artifact { artifact },
        } if artifact.title == "Artifact-only review"
    )));
}

#[tokio::test]
async fn non_stream_mixed_text_is_emitted_once_and_the_call_dispatches() {
    let mixed = json!({
        "id": "resp-tools",
        "status": "completed",
        "output": [
            {
                "id": "msg-commentary",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "phase": "commentary",
                "content": [{ "type": "output_text", "text": "Checking orders. " }]
            },
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "metric_point",
                "arguments": "{\"metric_id\":\"paid_order_count\"}"
            }
        ]
    });
    let final_answer = json!({
        "id": "resp-final",
        "status": "completed",
        "output": [{
            "id": "msg-final",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{ "type": "output_text", "text": "The answer is 42." }]
        }]
    });
    let (base_url, captured) = spawn_server(vec![
        MockTurn::ok(mixed.to_string()),
        MockTurn::ok(final_answer.to_string()),
    ])
    .await;
    let runner = build_agent_runner(
        MemorySaver::default(),
        model_with_stream(base_url, false),
        FakeTools,
        AllowAllTools,
    );

    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("What was the order count?")],
            context: Value::Null,
            retained_results: Vec::new(),
            budget: AgentBudget {
                max_model_turns: 2,
                max_tool_calls: 4,
            },
            human_input: None,
            synthesis_request: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert_eq!(
        message_deltas(&events),
        ["Checking orders. ", "The answer is 42."]
    );
    assert_eq!(completed(&events).answer, "The answer is 42.");
    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::ToolResult { tool_use_id, .. },
        } if tool_use_id == "call-1"
    )));
    assert_eq!(captured.lock().expect("captured").len(), 2);
}

#[tokio::test]
async fn native_plaintext_output_replays_in_provider_order_with_exact_call_arguments() {
    let valid_arguments = r#"{ "metric_id": "paid_order_count", "note": "\u0061" }"#;
    let malformed_arguments = r#"{"metric_id":"paid_order_count""#;
    let native_output = vec![
        json!({
            "type": "reasoning",
            "content": [{ "type": "reasoning_text", "text": "Inspect the metric." }]
        }),
        json!({
            "id": "msg-commentary",
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{ "type": "output_text", "text": "Checking both forms. " }]
        }),
        json!({
            "type": "function_call",
            "call_id": "call-valid",
            "name": "metric_point",
            "arguments": valid_arguments
        }),
        json!({
            "type": "reasoning",
            "content": [{ "type": "reasoning_text", "text": "Try the malformed form." }]
        }),
        json!({
            "type": "function_call",
            "call_id": "call-malformed",
            "name": "metric_point",
            "arguments": malformed_arguments
        }),
    ];
    let first = json!({
        "id": "resp-tools",
        "status": "completed",
        "output": native_output.clone()
    });
    let final_answer = json!({
        "id": "resp-final",
        "status": "completed",
        "output": [{
            "id": "msg-final",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{ "type": "output_text", "text": "Done." }]
        }]
    });
    let (base_url, captured) = spawn_server(vec![
        MockTurn::ok(first.to_string()),
        MockTurn::ok(final_answer.to_string()),
    ])
    .await;
    let runner = build_agent_runner(
        MemorySaver::default(),
        model_with_stream(base_url, false),
        FakeTools,
        AllowAllTools,
    );

    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Read the metric twice")],
            context: Value::Null,
            retained_results: Vec::new(),
            budget: AgentBudget {
                max_model_turns: 2,
                max_tool_calls: 4,
            },
            human_input: None,
            synthesis_request: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert_eq!(completed(&events).answer, "Done.");
    let bodies = captured_bodies(&captured);
    let replay = &bodies[1]["input"].as_array().expect("input array")[1..];
    let mut expected = native_output;
    expected.extend(["call-valid", "call-malformed"].map(|call_id| {
        json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": "{\"unit\":\"orders\",\"value\":42}"
        })
    }));
    assert_eq!(replay, expected, "native plaintext replay changed");
}

fn tool_reasoning_call_sse() -> String {
    let mut events = message_events("msg-commentary", 1, "commentary", &["Checking orders. "]);
    events.extend([
        json!({
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {
                "type": "function_call",
                "call_id": "call-1",
                "name": "metric_point",
                "arguments": ""
            }
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 2,
            "delta": "{\"metric_id\""
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 2,
            "delta": ":\"paid_order_count\"}"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 2,
            "item": {
                "type": "function_call",
                "call_id": "call-1",
                "name": "metric_point",
                "arguments": "{\"metric_id\":\"paid_order_count\"}"
            }
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp-tools",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "content": [{
                            "type": "reasoning_text",
                            "text": "Inspect metric privately."
                        }]
                    },
                    {
                        "id": "msg-commentary",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [{
                            "type": "output_text",
                            "text": "Checking orders. "
                        }]
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "metric_point",
                        "arguments": "{\"metric_id\":\"paid_order_count\"}"
                    }
                ]
            }
        }),
    ]);
    sse(events)
}

fn final_answer_sse() -> String {
    let mut events = message_events("msg-summary", 0, "commentary", &["Summarizing. "]);
    events.extend(message_events(
        "msg-final-1",
        1,
        "final_answer",
        &["The answer "],
    ));
    events.extend(message_events(
        "msg-final-2",
        2,
        "final_answer",
        &["is 42."],
    ));
    events.push(json!({
        "type": "response.completed",
        "response": {
            "id": "resp-final",
            "status": "completed",
            "output": [
                {
                    "id": "msg-summary",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{
                        "type": "output_text",
                        "text": "Summarizing. "
                    }]
                },
                {
                    "id": "msg-final-1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "The answer "
                    }]
                },
                {
                    "id": "msg-final-2",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "is 42."
                    }]
                }
            ]
        }
    }));
    sse(events)
}

fn message_prefix(id: &str, output_index: usize, phase: &str, deltas: &[&str]) -> Vec<Value> {
    let mut events = vec![
        json!({
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "id": id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "phase": phase,
                "content": []
            }
        }),
        json!({
            "type": "response.content_part.added",
            "item_id": id,
            "output_index": output_index,
            "content_index": 0,
            "part": { "type": "output_text", "text": "" }
        }),
    ];
    events.extend(deltas.iter().map(|delta| {
        json!({
            "type": "response.output_text.delta",
            "item_id": id,
            "output_index": output_index,
            "content_index": 0,
            "delta": delta
        })
    }));
    events
}

fn message_events(id: &str, output_index: usize, phase: &str, deltas: &[&str]) -> Vec<Value> {
    let text = deltas.concat();
    let mut events = message_prefix(id, output_index, phase, deltas);
    events.extend([
        json!({
            "type": "response.output_text.done",
            "item_id": id,
            "output_index": output_index,
            "content_index": 0,
            "text": text
        }),
        json!({
            "type": "response.content_part.done",
            "item_id": id,
            "output_index": output_index,
            "content_index": 0,
            "part": { "type": "output_text", "text": text }
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {
                "id": id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "phase": phase,
                "content": [{ "type": "output_text", "text": text }]
            }
        }),
    ]);
    events
}

#[tokio::test]
async fn max_tokens_candidate_is_streamed_but_not_committed() {
    let mut events = message_prefix("msg-partial", 0, "final_answer", &["Partial answer"]);
    events.push(json!({
        "type": "response.incomplete",
        "response": {
            "id": "resp-partial",
            "status": "incomplete",
            "incomplete_details": { "reason": "max_output_tokens" },
            "output": [{
                "id": "msg-partial",
                "type": "message",
                "status": "incomplete",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{ "type": "output_text", "text": "Partial answer" }]
            }]
        }
    }));
    let (base_url, _captured) = spawn_server(vec![MockTurn::ok(sse(events))]).await;
    let runner = build_agent_runner(
        MemorySaver::default(),
        model(base_url),
        FakeTools,
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Write a long answer")],
            context: Value::Null,
            retained_results: Vec::new(),
            budget: AgentBudget {
                max_model_turns: 1,
                max_tool_calls: 4,
            },
            human_input: None,
            synthesis_request: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert_eq!(message_deltas(&events), ["Partial answer"]);
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
    assert!(!events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::AssistantMessageDone { .. },
            ..
        }
    )));
}

#[tokio::test]
async fn max_tokens_without_message_fails_without_an_answer() {
    let (base_url, _captured) = spawn_server(vec![MockTurn::ok(sse([json!({
        "type": "response.incomplete",
        "response": {
            "id": "resp-empty",
            "status": "incomplete",
            "incomplete_details": { "reason": "max_output_tokens" },
            "output": []
        }
    })]))])
    .await;
    let runner = build_agent_runner(
        MemorySaver::default(),
        model(base_url),
        FakeTools,
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Write a long answer")],
            context: Value::Null,
            retained_results: Vec::new(),
            budget: AgentBudget::default(),
            human_input: None,
            synthesis_request: None,
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
                signal: AgentSignal::AssistantMessageDone { .. },
            }
    )));
}

#[derive(Default)]
struct FakeTools;

#[async_trait]
impl ToolRegistry for FakeTools {
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
        ToolResult::ok(&request.tool_use, json!({ "value": 42, "unit": "orders" })).with_artifacts(
            vec![Artifact {
                tool_use_id: request.tool_use.id,
                title: "Artifact-only review".to_string(),
                kind: "markdown".to_string(),
                content: "This text must not reach the model input.".to_string(),
                source: None,
                window: None,
                updated_at: None,
            }],
        )
    }
}

fn model(base_url: String) -> ConfiguredModel {
    model_with_stream(base_url, true)
}

fn model_with_stream(base_url: String, stream: bool) -> ConfiguredModel {
    let mut config = typemach_agent::AgentConfig::new("sk-test", "deepseek-v4-flash");
    config.base_url = base_url;
    config.max_retries = 0;
    config.stream = stream;
    ConfiguredModel::new(config).expect("model")
}

fn request(input: AgentRunInput) -> RunRequest<AgentRunInput> {
    RunRequest {
        run_id: RunId::from("run-1"),
        session_id: SessionId::from("session-1"),
        thread_id: ThreadId::from("thread-1"),
        command: RunCommand::Start,
        input,
        snapshot: None,
        runtime_limits: RuntimeLimits::new(32),
    }
}

async fn collect(
    mut rx: AgentEventReceiver,
) -> Vec<RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>> {
    let mut events = Vec::new();
    while let Some(event) = rx.next_event().await {
        let terminal = matches!(
            event,
            RunStreamEvent::Completed { .. }
                | RunStreamEvent::Interrupted { .. }
                | RunStreamEvent::Failed { .. }
                | RunStreamEvent::Cancelled
        );
        events.push(event);
        if terminal {
            break;
        }
    }
    events
}

fn completed(
    events: &[RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>],
) -> &AgentRunOutput {
    events
        .iter()
        .find_map(|event| match event {
            RunStreamEvent::Completed { output, .. } => Some(output),
            RunStreamEvent::Failed { error } => panic!("failed: {error}"),
            _ => None,
        })
        .expect("completed")
}

fn message_deltas(
    events: &[RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>],
) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal: AgentSignal::AssistantMessageDelta { delta, .. },
                ..
            } => Some(delta.clone()),
            _ => None,
        })
        .collect()
}

fn message_done_phases(
    events: &[RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>],
) -> Vec<AssistantMessagePhase> {
    events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal: AgentSignal::AssistantMessageDone { phase, .. },
                ..
            } => Some(*phase),
            _ => None,
        })
        .collect()
}

fn message_stream_phases(
    events: &[RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>],
) -> Vec<AssistantMessagePhase> {
    events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal: AgentSignal::AssistantMessageStarted { phase, .. },
                ..
            }
            | RunStreamEvent::Signal {
                signal: AgentSignal::AssistantMessageDelta { phase, .. },
                ..
            } => Some(*phase),
            _ => None,
        })
        .collect()
}

fn assistant_texts(messages: &[AgentMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Assistant { content } => Some(
                content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.clone()),
                        ContentBlock::AssistantMessage(message) => Some(message.text()),
                        _ => None,
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .filter(|text| !text.is_empty())
        .collect()
}

fn input_has_type(body: &Value, kind: &str) -> bool {
    body["input"]
        .as_array()
        .expect("input array")
        .iter()
        .any(|item| item["type"] == kind)
}

fn assert_ordered_input_types(body: &Value, expected: &[&str]) {
    let input = body["input"].as_array().expect("input array");
    let mut next = 0;
    for item in input {
        if next < expected.len() && item["type"] == expected[next] {
            next += 1;
        }
    }
    assert_eq!(
        next,
        expected.len(),
        "missing ordered input types {expected:?}"
    );
}
