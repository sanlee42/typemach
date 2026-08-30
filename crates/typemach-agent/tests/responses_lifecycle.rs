use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
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

    assert_eq!(
        message_deltas(&events, AssistantMessagePhase::Commentary),
        ["Checking orders. ", "The answer ", "is 42."]
    );
    let completed = completed(&events);
    assert_eq!(completed.answer, "The answer is 42.");
    assert!(!completed.answer.contains("privately"));
    assert_eq!(
        assistant_texts(&completed.messages),
        vec!["Checking orders. ", "The answer is 42."]
    );
    let bodies = captured_bodies(&captured);
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0]["tool_choice"], "auto");
    assert_eq!(bodies[1]["tool_choice"], "auto");
    assert!(
        bodies[1]["tools"]
            .as_array()
            .is_some_and(|tools| !tools.is_empty())
    );
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
                "type": "message",
                "role": "assistant",
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
            "type": "message",
            "role": "assistant",
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

    assert_eq!(
        message_deltas(&events, AssistantMessagePhase::Commentary),
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

fn tool_reasoning_call_sse() -> String {
    sse([
        json!({ "type": "response.output_text.delta", "delta": "Checking orders. " }),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call-1",
                "name": "metric_point",
                "arguments": ""
            }
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "{\"metric_id\""
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": ":\"paid_order_count\"}"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
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
                        "type": "message",
                        "role": "assistant",
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
    ])
}

fn final_answer_sse() -> String {
    sse([
        json!({ "type": "response.output_text.delta", "delta": "The answer " }),
        json!({ "type": "response.output_text.delta", "delta": "is 42." }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp-final",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "The answer is 42."
                    }]
                }]
            }
        }),
    ])
}

#[tokio::test]
async fn max_tokens_candidate_is_streamed_but_not_committed() {
    let (base_url, _captured) = spawn_server(vec![MockTurn::ok(sse([
        json!({ "type": "response.output_text.delta", "delta": "Partial answer" }),
        json!({
            "type": "response.incomplete",
            "response": {
                "id": "resp-partial",
                "status": "incomplete",
                "incomplete_details": { "reason": "max_output_tokens" },
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "Partial answer" }]
                }]
            }
        }),
    ]))])
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

    assert_eq!(
        message_deltas(&events, AssistantMessagePhase::Commentary),
        ["Partial answer"]
    );
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
    expected_phase: AssistantMessagePhase,
) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal: AgentSignal::AssistantMessageDelta { phase, delta, .. },
                ..
            } if *phase == expected_phase => Some(delta.clone()),
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
                        ContentBlock::Text { text } => Some(text.as_str()),
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

struct MockTurn {
    body: String,
}

impl MockTurn {
    fn ok(body: String) -> Self {
        Self { body }
    }
}

#[derive(Debug)]
struct CapturedRequest {
    body: Value,
}

async fn spawn_server(turns: Vec<MockTurn>) -> (String, Arc<Mutex<Vec<CapturedRequest>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let captured = Arc::new(Mutex::new(Vec::new()));
    let captured_for_task = Arc::clone(&captured);
    tokio::spawn(async move {
        for turn in turns {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let request = read_request(&mut socket).await;
            captured_for_task
                .lock()
                .expect("captured lock")
                .push(request);
            write_response(&mut socket, &turn).await;
        }
    });
    (format!("http://{addr}"), captured)
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> CapturedRequest {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 1024];
    let header_end = loop {
        let n = socket.read(&mut chunk).await.expect("read request");
        assert_ne!(n, 0, "connection closed before headers");
        buffer.extend_from_slice(&chunk[..n]);
        if let Some(index) = find_header_end(&buffer) {
            break index;
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let content_length = content_length(&headers);
    while buffer.len() < header_end + 4 + content_length {
        let n = socket.read(&mut chunk).await.expect("read body");
        assert_ne!(n, 0, "connection closed before body");
        buffer.extend_from_slice(&chunk[..n]);
    }
    let body = &buffer[header_end + 4..header_end + 4 + content_length];
    CapturedRequest {
        body: serde_json::from_slice(body).expect("json body"),
    }
}

async fn write_response(socket: &mut tokio::net::TcpStream, turn: &MockTurn) {
    let response = format!(
        "HTTP/1.1 200\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        turn.body.len(),
        turn.body
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("write response");
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn content_length(headers: &str) -> usize {
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().expect("content length"))
        })
        .expect("content-length")
}

fn captured_bodies(captured: &Arc<Mutex<Vec<CapturedRequest>>>) -> Vec<Value> {
    captured
        .lock()
        .expect("captured")
        .iter()
        .map(|request| request.body.clone())
        .collect()
}

fn sse(events: impl IntoIterator<Item = Value>) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str("data: ");
        body.push_str(&event.to_string());
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}
