use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use typemach_agent::{
    AgentConfig, AgentMessage, AgentModel, AgentToolSpec, ConfiguredModel, ContentBlock,
    ModelRequest, ModelStream, StopReason, ToolAnnotations,
};

#[tokio::test]
async fn flash_request_omits_thinking_and_decodes_tool_call() {
    let response = json!({
        "id": "chatcmpl-1",
        "choices": [{
            "message": {
                "reasoning_content": "inspect metric",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "metric_point",
                        "arguments": "{\"metric_id\":\"paid_order_count\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 3 }
    })
    .to_string();
    let (base_url, captured) = spawn_server(response, "application/json").await;
    let mut config = AgentConfig::new("sk-test", "deepseek-v4-flash");
    config.base_url = base_url;
    config.stream = false;
    let model = ConfiguredModel::new(config).expect("model");
    let (stream, _rx) = ModelStream::channel();
    let response = model
        .next_step(
            ModelRequest {
                messages: vec![AgentMessage::user_text("昨天订单量")],
                tools: vec![tool_spec()],
                context: Value::Null,
                turn: 1,
                system_suffix: None,
            },
            stream,
        )
        .await
        .expect("response");

    let body = captured_json(&captured);
    assert_eq!(body["model"], "deepseek-v4-flash");
    assert_eq!(body["stream"], false);
    assert!(body.get("thinking").is_none());
    assert!(body.get("reasoning_effort").is_none());
    assert_eq!(body["tools"][0]["function"]["name"], "metric_point");
    assert_eq!(response.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(response.usage.expect("usage").input_tokens, 10);
    assert!(matches!(
        response.content.first(),
        Some(ContentBlock::Thinking { text, .. }) if text == "inspect metric"
    ));
    assert!(response.content.iter().any(|block| matches!(
        block,
        ContentBlock::ToolUse(tool)
            if tool.id == "call-1"
                && tool.name == "metric_point"
                && tool.input["metric_id"] == "paid_order_count"
    )));
}

#[tokio::test]
async fn streaming_emits_text_and_assembles_tool_call() {
    let response = sse([
        json!({
            "id": "chatcmpl-2",
            "choices": [{
                "delta": { "reasoning_content": "think " },
                "finish_reason": null
            }]
        }),
        json!({
            "id": "chatcmpl-2",
            "choices": [{
                "delta": { "content": "订单" },
                "finish_reason": null
            }]
        }),
        json!({
            "id": "chatcmpl-2",
            "choices": [{
                "delta": { "content": "量" },
                "finish_reason": null
            }]
        }),
        json!({
            "id": "chatcmpl-2",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-2",
                        "type": "function",
                        "function": {
                            "name": "metric_point",
                            "arguments": "{\"metric_id\""
                        }
                    }]
                },
                "finish_reason": null
            }]
        }),
        json!({
            "id": "chatcmpl-2",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": { "arguments": ":\"paid_order_count\"}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }),
    ]);
    let (base_url, _captured) = spawn_server(response, "text/event-stream").await;
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
            },
            stream,
        )
        .await
        .expect("response");

    assert_eq!(rx.recv().await.as_deref(), Some("订单"));
    assert_eq!(rx.recv().await.as_deref(), Some("量"));
    assert_eq!(response.stop_reason, Some(StopReason::ToolUse));
    assert!(matches!(
        response.content.first(),
        Some(ContentBlock::Thinking { text, .. }) if text == "think "
    ));
    assert!(response.content.iter().any(|block| matches!(
        block,
        ContentBlock::ToolUse(tool)
            if tool.id == "call-2"
                && tool.input["metric_id"] == "paid_order_count"
    )));
}

#[tokio::test]
async fn system_suffix_is_appended_to_system_message() {
    let response = json!({
        "id": "chatcmpl-3",
        "choices": [{
            "message": { "content": "好的" },
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let (base_url, captured) = spawn_server(response, "application/json").await;
    let mut config = AgentConfig::new("sk-test", "deepseek-v4-flash");
    config.base_url = base_url;
    config.stream = false;
    config.system = Some("你是经营分析助手。".to_string());
    let model = ConfiguredModel::new(config).expect("model");
    let (stream, _rx) = ModelStream::channel();
    model
        .next_step(
            ModelRequest {
                messages: vec![AgentMessage::user_text("订单量")],
                tools: Vec::new(),
                context: Value::Null,
                turn: 1,
                system_suffix: Some("当前店铺:demo。".to_string()),
            },
            stream,
        )
        .await
        .expect("response");

    let body = captured_json(&captured);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(
        body["messages"][0]["content"],
        "你是经营分析助手。\n\n当前店铺:demo。"
    );

    // Without a static base prompt the suffix becomes the whole system message.
    let response = json!({
        "id": "chatcmpl-4",
        "choices": [{
            "message": { "content": "好的" },
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let (base_url, captured) = spawn_server(response, "application/json").await;
    let mut config = AgentConfig::new("sk-test", "deepseek-v4-flash");
    config.base_url = base_url;
    config.stream = false;
    let model = ConfiguredModel::new(config).expect("model");
    let (stream, _rx) = ModelStream::channel();
    model
        .next_step(
            ModelRequest {
                messages: vec![AgentMessage::user_text("订单量")],
                tools: Vec::new(),
                context: Value::Null,
                turn: 1,
                system_suffix: Some("当前店铺:demo。".to_string()),
            },
            stream,
        )
        .await
        .expect("response");
    let body = captured_json(&captured);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "当前店铺:demo。");
}

fn tool_spec() -> AgentToolSpec {
    AgentToolSpec {
        name: "metric_point".to_string(),
        description: "read metric point".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "metric_id": { "type": "string" }
            },
            "required": ["metric_id"]
        }),
        output_schema: Value::Null,
        metadata: Value::Null,
        annotations: ToolAnnotations::default(),
    }
}

fn sse<const N: usize>(chunks: [Value; N]) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str("data: ");
        body.push_str(&chunk.to_string());
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}

async fn spawn_server(
    response: String,
    content_type: &'static str,
) -> (String, Arc<Mutex<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let captured = Arc::new(Mutex::new(String::new()));
    let captured_for_task = Arc::clone(&captured);
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let request = read_request(&mut socket).await;
        *captured_for_task.lock().expect("captured lock") = request;
        let bytes = response.as_bytes();
        let header = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            bytes.len()
        );
        socket.write_all(header.as_bytes()).await.expect("header");
        socket.write_all(bytes).await.expect("body");
    });
    (format!("http://{addr}"), captured)
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0_u8; 1024];
    loop {
        let read = socket.read(&mut tmp).await.expect("read");
        if read == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..read]);
        if request_complete(&buf) {
            break;
        }
    }
    String::from_utf8(buf).expect("request utf8")
}

fn request_complete(buf: &[u8]) -> bool {
    let Some(header_end) = buf.windows(4).position(|window| window == b"\r\n\r\n") else {
        return false;
    };
    let headers = String::from_utf8_lossy(&buf[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or_default();
    buf.len() >= header_end + 4 + content_length
}

fn captured_json(captured: &Arc<Mutex<String>>) -> Value {
    let request = captured.lock().expect("captured lock");
    let (_, body) = request.split_once("\r\n\r\n").expect("request body");
    serde_json::from_str(body).expect("json body")
}
