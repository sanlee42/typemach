use super::*;

#[tokio::test]
async fn default_endpoint_uses_responses_and_streams_typed_final_text() {
    let response = sse([
        json!({
            "type": "response.output_text.delta",
            "delta": "订单"
        }),
        json!({
            "type": "response.output_text.delta",
            "delta": "量 42。"
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "订单量 42。"
                    }]
                }],
                "usage": { "input_tokens": 11, "output_tokens": 5 }
            }
        }),
    ]);
    let (base_url, captured) = spawn_responses_server(response, "text/event-stream").await;
    let mut config = AgentConfig::new("sk-test", "deepseek-v4-flash");
    config.base_url = base_url;
    let model = ConfiguredModel::new(config).expect("model");
    let (stream, mut rx) = ModelStream::channel();
    let response = model
        .next_step(
            ModelRequest {
                messages: vec![AgentMessage::user_text("昨天订单量")],
                tools: vec![tool_spec()],
                context: Value::Null,
                turn: 1,
                system_suffix: Some("Current shop: demo.".to_string()),
                tool_choice: Some(typemach_agent::ToolChoice::Auto),
            },
            stream,
        )
        .await
        .expect("response");

    let first = rx.recv().await.expect("first delta");
    assert_eq!(first.text, "订单");
    assert!(first.final_answer);
    let second = rx.recv().await.expect("second delta");
    assert_eq!(second.text, "量 42。");
    assert!(second.final_answer);
    assert_eq!(response.stop_reason, Some(StopReason::EndTurn));
    assert!(response.final_answer);
    assert_eq!(response.usage.expect("usage").input_tokens, 11);
    let body = captured_json(&captured);
    assert_eq!(body["model"], "deepseek-v4-flash");
    assert_eq!(body["stream"], true);
    assert_eq!(body["instructions"], "Current shop: demo.");
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["tools"][0]["name"], "metric_point");
    assert_eq!(body["reasoning"]["effort"], "none");
}

#[tokio::test]
async fn responses_function_call_arguments_are_not_user_streamed() {
    let response = sse([
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
    ]);
    let (base_url, _captured) = spawn_responses_server(response, "text/event-stream").await;
    let mut config = AgentConfig::new("sk-test", "deepseek-v4-flash");
    config.base_url = base_url;
    let model = ConfiguredModel::new(config).expect("model");
    let (stream, mut rx) = ModelStream::channel();
    let response = model
        .next_step(
            ModelRequest {
                messages: vec![AgentMessage::user_text("昨天订单量")],
                tools: vec![tool_spec()],
                context: Value::Null,
                turn: 1,
                system_suffix: None,
                tool_choice: Some(typemach_agent::ToolChoice::Auto),
            },
            stream,
        )
        .await
        .expect("response");

    assert!(rx.try_recv().is_err());
    assert_eq!(response.stop_reason, Some(StopReason::ToolUse));
    assert!(!response.final_answer);
    assert!(response.content.iter().any(|block| matches!(
        block,
        ContentBlock::ToolUse(tool)
            if tool.id == "call-1"
                && tool.name == "metric_point"
                && tool.input["metric_id"] == "paid_order_count"
    )));
}
