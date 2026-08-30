use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use typemach_agent::{
    AgentConfig, AgentMessage, AgentModel, AgentToolSpec, ConfiguredModel, ModelOutcome,
    ModelRequest, ModelStream, StopReason, ToolAnnotations,
};

#[tokio::test]
async fn origin_base_posts_to_responses_and_stale_chat_base_is_invalid() {
    let (base_url, captured) = spawn_server(vec![MockTurn::ok(ok_message("Done."))]).await;
    let model = ConfiguredModel::new(config(base_url, false)).expect("model");
    let (stream, _rx) = ModelStream::channel();

    model
        .next_step(
            request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
            stream,
        )
        .await
        .expect("response");

    assert_eq!(captured.lock().expect("captured")[0].target, "/responses");
    let err = match ConfiguredModel::new(config(
        "http://127.0.0.1:9/chat/completions".to_string(),
        false,
    )) {
        Ok(_) => panic!("stale chat endpoint must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("explicit /responses endpoint"));
}

#[tokio::test]
async fn streaming_final_text_has_one_live_sink() {
    let (base_url, captured) = spawn_server(vec![MockTurn::ok(sse([
        json!({ "type": "response.output_text.delta", "delta": "A" }),
        json!({ "type": "response.output_text.delta", "delta": "B" }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "AB" }]
                }],
                "usage": { "input_tokens": 11, "output_tokens": 5 }
            }
        }),
    ]))])
    .await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, mut rx) = ModelStream::channel();
    let response = model
        .next_step(
            request(vec![tool_spec()], Some(typemach_agent::ToolChoice::Auto)),
            stream,
        )
        .await
        .expect("response");

    assert_eq!(rx.recv().await.expect("first").0, "A");
    assert_eq!(rx.recv().await.expect("second").0, "B");
    assert!(rx.try_recv().is_err());
    assert_eq!(
        response.outcome,
        Some(ModelOutcome::FinalAnswer {
            text: String::new()
        })
    );
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(response.usage.expect("usage").input_tokens, 11);
    let body = &captured.lock().expect("captured")[0].body;
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["tools"][0]["name"], "metric_point");
}

#[tokio::test]
async fn no_delta_completed_message_returns_text_once() {
    let (base_url, _captured) = spawn_server(vec![MockTurn::ok(sse([json!({
        "type": "response.completed",
        "response": completed_message("No live delta.")
    })]))])
    .await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, mut rx) = ModelStream::channel();
    let response = model
        .next_step(
            request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
            stream,
        )
        .await
        .expect("response");

    assert!(rx.try_recv().is_err());
    assert_eq!(
        response.outcome,
        Some(ModelOutcome::FinalAnswer {
            text: "No live delta.".to_string()
        })
    );
}

#[tokio::test]
async fn stream_buffers_split_multibyte_utf8_until_line_boundary() {
    let body = sse([
        json!({ "type": "response.output_text.delta", "delta": "Orders" }),
        json!({ "type": "response.output_text.delta", "delta": "订单" }),
        json!({
            "type": "response.completed",
            "response": completed_message("Orders订单")
        }),
    ]);
    let split_at = body.find("订单").expect("multibyte text") + 1;
    let (base_url, _captured) = spawn_server(vec![MockTurn {
        status: 200,
        content_type: "text/event-stream",
        body,
        delivery: Delivery::SplitBodyAt(split_at),
    }])
    .await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, mut rx) = ModelStream::channel();

    model
        .next_step(
            request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
            stream,
        )
        .await
        .expect("response");

    assert_eq!(rx.recv().await.expect("first").0, "Orders");
    assert_eq!(rx.recv().await.expect("second").0, "订单");
}

#[tokio::test]
async fn stream_handles_delivery_split_after_data_line_newline() {
    let body = sse([
        json!({ "type": "response.output_text.delta", "delta": "A" }),
        json!({ "type": "response.output_text.delta", "delta": "B" }),
        json!({
            "type": "response.completed",
            "response": completed_message("AB")
        }),
    ]);
    let split_at = body.find('\n').expect("first line newline") + 1;
    let (base_url, _captured) = spawn_server(vec![MockTurn {
        status: 200,
        content_type: "text/event-stream",
        body,
        delivery: Delivery::SplitBodyAt(split_at),
    }])
    .await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, mut rx) = ModelStream::channel();

    model
        .next_step(
            request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
            stream,
        )
        .await
        .expect("response");

    assert_eq!(rx.recv().await.expect("first").0, "A");
    assert_eq!(rx.recv().await.expect("second").0, "B");
}

#[tokio::test]
async fn function_call_arguments_are_private_and_decoded() {
    let (base_url, _captured) = spawn_server(vec![MockTurn::ok(sse([
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
            "type": "response.completed",
            "response": {
                "id": "resp-2",
                "status": "completed",
                "output": [{
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "metric_point",
                    "arguments": "{\"metric_id\":\"paid_order_count\"}"
                }]
            }
        }),
    ]))])
    .await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, mut rx) = ModelStream::channel();
    let response = model
        .next_step(
            request(vec![tool_spec()], Some(typemach_agent::ToolChoice::Auto)),
            stream,
        )
        .await
        .expect("response");

    assert!(rx.try_recv().is_err());
    let Some(ModelOutcome::ToolCalls { calls }) = response.outcome else {
        panic!("expected tool calls");
    };
    assert_eq!(calls[0].id, "call-1");
    assert_eq!(calls[0].input["metric_id"], "paid_order_count");
}

#[tokio::test]
async fn fallback_request_serializes_explicit_none_without_tools() {
    let (base_url, captured) = spawn_server(vec![MockTurn::ok(ok_message("Done."))]).await;
    let model = ConfiguredModel::new(config(base_url, false)).expect("model");
    let (stream, _rx) = ModelStream::channel();

    model
        .next_step(
            request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
            stream,
        )
        .await
        .expect("response");

    let body = &captured.lock().expect("captured")[0].body;
    assert!(body.get("tools").is_none());
    assert_eq!(body["tool_choice"], "none");
}

#[tokio::test]
async fn malformed_refusal_and_mixed_responses_fail_structurally() {
    for response in [
        completed_refusal(),
        json!({
            "id": "resp-mixed",
            "status": "completed",
            "output": [
                message_item("Text"),
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "metric_point",
                    "arguments": "{}"
                }
            ]
        }),
        json!({
            "id": "resp-missing-call-id",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "id": "item-1",
                "name": "metric_point",
                "arguments": "{}"
            }]
        }),
        json!({
            "id": "resp-missing-arguments",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "call_id": "call-1",
                "name": "metric_point"
            }]
        }),
    ] {
        let (base_url, captured) = spawn_server(vec![
            MockTurn::ok(response.to_string()),
            MockTurn::ok(response.to_string()),
            MockTurn::ok(response.to_string()),
        ])
        .await;
        let mut config = config(base_url, false);
        config.max_retries = 2;
        let model = ConfiguredModel::new(config).expect("model");
        let (stream, _rx) = ModelStream::channel();
        let err = model
            .next_step(
                request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
                stream,
            )
            .await
            .expect_err("structural failure");
        assert!(
            err.to_string()
                .contains("model request failed after 1 attempts")
        );
        assert_eq!(captured.lock().expect("captured").len(), 1);
    }
}

#[tokio::test]
async fn retry_stops_after_public_answer_delta() {
    let first = sse([json!({ "type": "response.output_text.delta", "delta": "A" })]);
    let (base_url, captured) = spawn_server(vec![MockTurn {
        status: 200,
        content_type: "text/event-stream",
        body: first,
        delivery: Delivery::Truncate,
    }])
    .await;
    let mut config = config(base_url, true);
    config.max_retries = 2;
    let model = ConfiguredModel::new(config).expect("model");
    let (stream, mut rx) = ModelStream::channel();

    let err = model
        .next_step(
            request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
            stream,
        )
        .await
        .expect_err("must not retry");

    assert!(err.to_string().contains("after 1 attempts"));
    assert_eq!(rx.recv().await.expect("delta").0, "A");
    assert_eq!(captured.lock().expect("captured").len(), 1);
}

fn config(base_url: String, stream: bool) -> AgentConfig {
    let mut config = AgentConfig::new("sk-test", "deepseek-v4-flash");
    config.base_url = base_url;
    config.stream = stream;
    config.max_retries = 0;
    config.request_timeout_secs = 1;
    config
}

fn request(
    tools: Vec<AgentToolSpec>,
    tool_choice: Option<typemach_agent::ToolChoice>,
) -> ModelRequest {
    ModelRequest {
        messages: vec![AgentMessage::user_text("Read the metric")],
        tools,
        context: Value::Null,
        turn: 1,
        system_suffix: None,
        tool_choice,
    }
}

fn tool_spec() -> AgentToolSpec {
    AgentToolSpec {
        name: "metric_point".to_string(),
        description: "read metric point".to_string(),
        input_schema: json!({ "type": "object" }),
        output_schema: Value::Null,
        metadata: Value::Null,
        annotations: ToolAnnotations::default(),
    }
}

fn ok_message(text: &str) -> String {
    completed_message(text).to_string()
}

fn completed_message(text: &str) -> Value {
    json!({
        "id": "resp-ok",
        "status": "completed",
        "output": [message_item(text)]
    })
}

fn message_item(text: &str) -> Value {
    json!({
        "type": "message",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": text }]
    })
}

fn completed_refusal() -> Value {
    json!({
        "id": "resp-refusal",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "refusal", "refusal": "Cannot comply." }]
        }]
    })
}

enum Delivery {
    Complete,
    Truncate,
    SplitBodyAt(usize),
}

struct MockTurn {
    status: u16,
    content_type: &'static str,
    body: String,
    delivery: Delivery,
}

impl MockTurn {
    fn ok(body: String) -> Self {
        Self {
            status: 200,
            content_type: "application/json",
            body,
            delivery: Delivery::Complete,
        }
    }
}

#[derive(Debug)]
struct CapturedRequest {
    target: String,
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
    let target = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .expect("request target")
        .to_string();
    let content_length = content_length(&headers);
    while buffer.len() < header_end + 4 + content_length {
        let n = socket.read(&mut chunk).await.expect("read body");
        assert_ne!(n, 0, "connection closed before body");
        buffer.extend_from_slice(&chunk[..n]);
    }
    let body = &buffer[header_end + 4..header_end + 4 + content_length];
    CapturedRequest {
        target,
        body: serde_json::from_slice(body).expect("json body"),
    }
}

async fn write_response(socket: &mut tokio::net::TcpStream, turn: &MockTurn) {
    let advertised_len = match turn.delivery {
        Delivery::Complete | Delivery::SplitBodyAt(_) => turn.body.len(),
        Delivery::Truncate => turn.body.len() + 16,
    };
    let head = format!(
        "HTTP/1.1 {}\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        turn.status, turn.content_type, advertised_len
    );
    socket
        .write_all(head.as_bytes())
        .await
        .expect("write response head");
    match turn.delivery {
        Delivery::Complete | Delivery::Truncate => {
            socket
                .write_all(turn.body.as_bytes())
                .await
                .expect("write response body");
        }
        Delivery::SplitBodyAt(index) => {
            let (first, second) = turn.body.as_bytes().split_at(index);
            socket.write_all(first).await.expect("write first body");
            socket.flush().await.expect("flush first body");
            tokio::task::yield_now().await;
            socket.write_all(second).await.expect("write second body");
        }
    }
    if !matches!(turn.delivery, Delivery::Truncate) {
        socket.shutdown().await.expect("shutdown response");
    }
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
