use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use typemach_agent::{
    AgentConfig, AgentMessage, AgentModel, AgentToolSpec, AssistantMessageItem,
    AssistantMessagePhase, ConfiguredModel, ContentBlock, ModelRequest, ModelResponse, ModelStream,
    ModelStreamEvent, StopReason, ToolAnnotations, ToolUse,
};

#[path = "deepseek/fixtures.rs"]
mod fixtures;
use fixtures::{Delivery, MockTurn, spawn_server, sse};

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
    let mut events = message_events("msg-final", 0, "final_answer", &["A", "B"]);
    events.push(json!({
        "type": "response.completed",
        "response": {
            "id": "resp-1",
            "status": "completed",
            "output": [message_item("msg-final", "final_answer", "AB")],
            "usage": { "input_tokens": 11, "output_tokens": 5 }
        }
    }));
    let (base_url, captured) = spawn_server(vec![MockTurn::ok(sse(events))]).await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, mut rx) = ModelStream::channel();
    let response = model
        .next_step(
            request(vec![tool_spec()], Some(typemach_agent::ToolChoice::Auto)),
            stream,
        )
        .await
        .expect("response");

    assert!(matches!(
        rx.recv().await.expect("started"),
        ModelStreamEvent::AssistantMessageStarted {
            phase: AssistantMessagePhase::FinalAnswer,
            ..
        }
    ));
    assert_eq!(next_delta(&mut rx).await, "A");
    assert_eq!(next_delta(&mut rx).await, "B");
    assert!(matches!(
        rx.recv().await.expect("done"),
        ModelStreamEvent::AssistantMessageDone { .. }
    ));
    assert!(rx.try_recv().is_err());
    assert_eq!(assistant_messages(&response)[0].text(), "AB");
    assert_eq!(
        assistant_messages(&response)[0].phase,
        AssistantMessagePhase::FinalAnswer
    );
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(response.usage.expect("usage").input_tokens, 11);
    let body = &captured.lock().expect("captured")[0].body;
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["tools"][0]["name"], "metric_point");
}

#[tokio::test]
async fn streamed_text_remains_live_when_the_response_also_calls_a_tool() {
    let mut events = message_events("msg-commentary", 0, "commentary", &["Checking orders. "]);
    events.push(json!({
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {
            "type": "function_call",
            "call_id": "call-1",
            "name": "metric_point",
            "arguments": "{\"metric_id\":\"paid_order_count\"}"
        }
    }));
    events.push(json!({
        "type": "response.completed",
        "response": mixed_response("Checking orders. ")
    }));
    let body = sse(events);
    let delta_end = body
        .find("response.output_text.delta")
        .and_then(|start| body[start..].find("\n\n").map(|end| start + end + 2))
        .expect("delta event boundary");
    let release = Arc::new(Notify::new());
    let (base_url, _captured) = spawn_server(vec![MockTurn {
        status: 200,
        content_type: "text/event-stream",
        body,
        delivery: Delivery::HoldAfter(delta_end, Arc::clone(&release)),
    }])
    .await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, mut rx) = ModelStream::channel();
    let response = tokio::spawn(async move {
        model
            .next_step(
                request(vec![tool_spec()], Some(typemach_agent::ToolChoice::Auto)),
                stream,
            )
            .await
    });

    let delta = tokio::time::timeout(Duration::from_secs(1), next_delta(&mut rx))
        .await
        .expect("delta before terminal response");
    assert_eq!(delta, "Checking orders. ");
    assert!(!response.is_finished());
    release.notify_one();

    let response = response.await.expect("model task").expect("response");
    assert_eq!(assistant_messages(&response)[0].text(), "Checking orders. ");
    assert_eq!(
        assistant_messages(&response)[0].phase,
        AssistantMessagePhase::Commentary
    );
    assert_eq!(tool_calls(&response)[0].id, "call-1");
    assert_eq!(response.stop_reason, Some(StopReason::ToolUse));
}

#[tokio::test]
async fn content_part_text_without_delta_is_forwarded_once() {
    let mut events =
        message_events_with_initial("msg-no-delta", 0, "final_answer", "No live delta.");
    events.push(json!({
        "type": "response.completed",
        "response": completed_message_with("msg-no-delta", "No live delta.")
    }));
    let (base_url, _captured) = spawn_server(vec![MockTurn::ok(sse(events))]).await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, mut rx) = ModelStream::channel();
    let response = model
        .next_step(
            request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
            stream,
        )
        .await
        .expect("response");

    assert_eq!(next_delta(&mut rx).await, "No live delta.");
    assert_eq!(assistant_messages(&response)[0].text(), "No live delta.");
}

#[tokio::test]
async fn stream_buffers_split_multibyte_utf8_until_line_boundary() {
    let mut events = message_events("msg-utf8", 0, "final_answer", &["Orders", "订单"]);
    events.push(json!({
        "type": "response.completed",
        "response": completed_message_with("msg-utf8", "Orders订单")
    }));
    let body = sse(events);
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

    assert_eq!(next_delta(&mut rx).await, "Orders");
    assert_eq!(next_delta(&mut rx).await, "订单");
}

#[tokio::test]
async fn stream_handles_delivery_split_after_data_line_newline() {
    let mut events = message_events("msg-split", 0, "final_answer", &["A", "B"]);
    events.push(json!({
        "type": "response.completed",
        "response": completed_message_with("msg-split", "AB")
    }));
    let body = sse(events);
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

    assert_eq!(next_delta(&mut rx).await, "A");
    assert_eq!(next_delta(&mut rx).await, "B");
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
    assert_eq!(tool_calls(&response)[0].id, "call-1");
    assert_eq!(
        tool_calls(&response)[0].input["metric_id"],
        "paid_order_count"
    );
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
async fn malformed_responses_fail_structurally() {
    for response in [
        completed_refusal(),
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
    let mut events = message_events("msg-truncated", 0, "final_answer", &["A"]);
    events.truncate(3);
    let first = sse(events);
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
    assert_eq!(next_delta(&mut rx).await, "A");
    assert_eq!(captured.lock().expect("captured").len(), 1);
}

#[tokio::test]
async fn conflicting_completed_item_is_rejected_without_done() {
    let mut events = message_events("msg-conflict", 0, "final_answer", &["A"]);
    events.last_mut().expect("item done")["item"]["content"][0]["text"] = json!("B");
    events.push(json!({
        "type": "response.completed",
        "response": completed_message_with("msg-conflict", "B")
    }));
    let (base_url, _captured) = spawn_server(vec![MockTurn::ok(sse(events))]).await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, mut rx) = ModelStream::channel();

    let error = model
        .next_step(
            request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
            stream,
        )
        .await
        .expect_err("conflicting item must fail");

    assert!(
        error
            .to_string()
            .contains("streamed message bytes differed")
    );
    assert_eq!(next_delta(&mut rx).await, "A");
    while let Ok(event) = rx.try_recv() {
        assert!(!matches!(
            event,
            ModelStreamEvent::AssistantMessageDone { .. }
        ));
    }
}

#[tokio::test]
async fn completed_phase_is_normalized_to_the_added_message() {
    let mut events = message_events("msg-conflict", 0, "commentary", &["A"]);
    events.last_mut().expect("item done")["item"]["phase"] = json!("final_answer");
    events.push(json!({
        "type": "response.completed",
        "response": completed_message_with("msg-conflict", "A")
    }));
    let (base_url, _captured) = spawn_server(vec![MockTurn::ok(sse(events))]).await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, mut rx) = ModelStream::channel();

    let response = model
        .next_step(
            request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
            stream,
        )
        .await
        .expect("phase mismatch is display metadata");
    assert!(matches!(
        rx.recv().await.expect("started"),
        ModelStreamEvent::AssistantMessageStarted {
            phase: AssistantMessagePhase::Commentary,
            ..
        }
    ));
    assert_eq!(next_delta(&mut rx).await, "A");
    assert!(matches!(
        rx.recv().await.expect("done"),
        ModelStreamEvent::AssistantMessageDone {
            message: typemach_agent::AssistantMessageItem {
                phase: AssistantMessagePhase::Commentary,
                ..
            }
        }
    ));
    assert_eq!(
        assistant_messages(&response)[0].phase,
        AssistantMessagePhase::Commentary
    );
}

#[tokio::test]
async fn changed_completed_message_id_is_rejected() {
    let mut events = message_events("msg-original", 0, "final_answer", &["A"]);
    events.last_mut().expect("item done")["item"]["id"] = json!("msg-changed");
    events.push(json!({
        "type": "response.completed",
        "response": completed_message_with("msg-changed", "A")
    }));
    let (base_url, _captured) = spawn_server(vec![MockTurn::ok(sse(events))]).await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, _rx) = ModelStream::channel();

    let error = model
        .next_step(
            request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
            stream,
        )
        .await
        .expect_err("changed id must fail");

    assert!(
        error
            .to_string()
            .contains("completed message id differed from active item")
    );
}

#[tokio::test]
async fn conflicting_terminal_message_bytes_are_rejected() {
    let mut events = message_events("msg-terminal", 0, "final_answer", &["A"]);
    events.push(json!({
        "type": "response.completed",
        "response": completed_message_with("msg-terminal", "B")
    }));
    let (base_url, _captured) = spawn_server(vec![MockTurn::ok(sse(events))]).await;
    let model = ConfiguredModel::new(config(base_url, true)).expect("model");
    let (stream, _rx) = ModelStream::channel();

    let error = model
        .next_step(
            request(Vec::new(), Some(typemach_agent::ToolChoice::None)),
            stream,
        )
        .await
        .expect_err("conflicting terminal snapshot must fail");

    assert!(
        error
            .to_string()
            .contains("completed message 0 bytes differed from response snapshot")
    );
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
    completed_message_with("msg-final", text)
}

fn completed_message_with(id: &str, text: &str) -> Value {
    json!({
        "id": "resp-ok",
        "status": "completed",
        "output": [message_item(id, "final_answer", text)]
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

fn message_events(id: &str, output_index: usize, phase: &str, deltas: &[&str]) -> Vec<Value> {
    let text = deltas.concat();
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
            "item": message_item(id, phase, &text)
        }),
    ]);
    events
}

fn message_events_with_initial(
    id: &str,
    output_index: usize,
    phase: &str,
    text: &str,
) -> Vec<Value> {
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
            "part": { "type": "output_text", "text": text }
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": message_item(id, phase, text)
        }),
    ]
}

async fn next_delta(rx: &mut tokio::sync::mpsc::UnboundedReceiver<ModelStreamEvent>) -> String {
    loop {
        if let ModelStreamEvent::AssistantMessageDelta { delta, .. } =
            rx.recv().await.expect("model stream event")
        {
            return delta;
        }
    }
}

fn mixed_response(text: &str) -> Value {
    json!({
        "id": "resp-mixed",
        "status": "completed",
        "output": [
            message_item("msg-commentary", "commentary", text),
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "metric_point",
                "arguments": "{\"metric_id\":\"paid_order_count\"}"
            }
        ]
    })
}

fn completed_refusal() -> Value {
    json!({
        "id": "resp-refusal",
        "status": "completed",
        "output": [{
            "id": "msg-refusal",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{ "type": "refusal", "refusal": "Cannot comply." }]
        }]
    })
}

fn assistant_messages(response: &ModelResponse) -> Vec<&AssistantMessageItem> {
    response
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::AssistantMessage(message) => Some(message),
            _ => None,
        })
        .collect()
}

fn tool_calls(response: &ModelResponse) -> Vec<&ToolUse> {
    response
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse(tool_use) => Some(tool_use),
            _ => None,
        })
        .collect()
}
