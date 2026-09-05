use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use typemach_agent::{
    AgentConfig, AgentMessage, AgentModel, AssistantMessagePhase, ConfiguredModel, ModelRequest,
    ModelStream, ModelStreamEvent, StopReason,
};

#[tokio::test]
async fn reasoning_lifecycle_precedes_a_function_call_without_emitting_text() {
    let mut events = reasoning_events("reasoning-tool", 0, "private reasoning");
    events.extend([
        function_added(1),
        function_done(1),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp-tool",
                "status": "completed",
                "output": [reasoning_item("reasoning-tool", "private reasoning"), function_item()]
            }
        }),
    ]);
    let model = ConfiguredModel::new(config(server(sse(events)).await)).expect("model");
    let (stream, mut rx) = ModelStream::channel();

    let response = model
        .next_step(request(Some(typemach_agent::ToolChoice::Auto)), stream)
        .await
        .expect("response");

    assert_eq!(response.tool_calls[0].id, "call-1");
    assert_eq!(response.reasoning, ["private reasoning"]);
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn reasoning_lifecycle_keeps_final_answer_streaming_typed() {
    let mut events = reasoning_events("reasoning-final", 0, "private reasoning");
    events.extend(message_events(
        "msg-final",
        1,
        "final_answer",
        "Visible answer.",
    ));
    events.push(json!({
        "type": "response.completed",
        "response": {
            "id": "resp-final",
            "status": "completed",
            "output": [
                reasoning_item("reasoning-final", "private reasoning"),
                message_item("msg-final", "final_answer", "Visible answer.")
            ]
        }
    }));
    let model = ConfiguredModel::new(config(server(sse(events)).await)).expect("model");
    let (stream, mut rx) = ModelStream::channel();

    let response = model
        .next_step(request(Some(typemach_agent::ToolChoice::None)), stream)
        .await
        .expect("response");

    assert!(matches!(
        rx.recv().await.expect("started"),
        ModelStreamEvent::AssistantMessageStarted {
            phase: AssistantMessagePhase::FinalAnswer,
            ..
        }
    ));
    assert!(matches!(
        rx.recv().await.expect("delta"),
        ModelStreamEvent::AssistantMessageDelta { delta, .. } if delta == "Visible answer."
    ));
    assert!(matches!(
        rx.recv().await.expect("done"),
        ModelStreamEvent::AssistantMessageDone { .. }
    ));
    assert!(rx.try_recv().is_err());
    assert_eq!(response.assistant_messages[0].text(), "Visible answer.");
    assert_eq!(response.reasoning, ["private reasoning"]);
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
}

fn config(base_url: String) -> AgentConfig {
    let mut config = AgentConfig::new("sk-test", "deepseek-v4-flash");
    config.base_url = base_url;
    config.stream = true;
    config.max_retries = 0;
    config.request_timeout_secs = 1;
    config
}

fn request(tool_choice: Option<typemach_agent::ToolChoice>) -> ModelRequest {
    ModelRequest {
        messages: vec![AgentMessage::user_text("Read the metric")],
        tools: Vec::new(),
        context: Value::Null,
        turn: 1,
        system_suffix: None,
        tool_choice,
    }
}

fn reasoning_events(id: &str, output_index: usize, text: &str) -> Vec<Value> {
    vec![
        json!({
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "id": id,
                "type": "reasoning",
                "status": "in_progress",
                "content": [],
                "summary": []
            }
        }),
        json!({
            "type": "response.content_part.added",
            "item_id": id,
            "output_index": output_index,
            "content_index": 0,
            "part": { "type": "reasoning_text", "text": "" }
        }),
        json!({
            "type": "response.reasoning_text.delta",
            "item_id": id,
            "output_index": output_index,
            "content_index": 0,
            "delta": text
        }),
        json!({
            "type": "response.reasoning_text.done",
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
            "part": { "type": "reasoning_text", "text": text }
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": reasoning_item(id, text)
        }),
    ]
}

fn message_events(id: &str, output_index: usize, phase: &str, text: &str) -> Vec<Value> {
    vec![
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
        json!({
            "type": "response.output_text.delta",
            "item_id": id,
            "output_index": output_index,
            "content_index": 0,
            "delta": text
        }),
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
            "item": message_item(id, phase, text)
        }),
    ]
}

fn reasoning_item(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "reasoning",
        "status": "completed",
        "content": [{ "type": "reasoning_text", "text": text }],
        "summary": []
    })
}

fn message_item(id: &str, phase: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "phase": phase,
        "content": [{ "type": "output_text", "text": text }]
    })
}

fn function_added(output_index: usize) -> Value {
    json!({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": function_item()
    })
}

fn function_done(output_index: usize) -> Value {
    json!({
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": function_item()
    })
}

fn function_item() -> Value {
    json!({
        "type": "function_call",
        "call_id": "call-1",
        "name": "metric_point",
        "arguments": "{\"metric_id\":\"paid_order_count\"}"
    })
}

fn sse(events: Vec<Value>) -> String {
    events
        .into_iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
}

async fn server(body: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let address = listener.local_addr().expect("local address");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
            let read = socket.read(&mut buffer).await.expect("read request");
            assert_ne!(read, 0, "request ended before headers");
            request.extend_from_slice(&buffer[..read]);
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
    });
    format!("http://{address}")
}
