use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::deepseek::{
    combined_system, decode_arguments, effort_value, encode_arguments, tool_choice_value,
    tool_result_content,
};
use crate::{
    AgentConfig, AgentError, AgentMessage, AgentToolSpec, ContentBlock, ModelRequest,
    ModelResponse, SpeedProfile, StopReason, ToolUse, Usage,
};

#[derive(Debug, Serialize)]
pub(crate) struct ResponseRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    input: Vec<Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
    reasoning: ResponseReasoning,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct ResponseReasoning {
    effort: &'static str,
}

#[derive(Debug, Deserialize)]
struct ResponseUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

pub(crate) fn responses_request(
    config: &AgentConfig,
    request: ModelRequest,
) -> Result<ResponseRequest, AgentError> {
    let tools = tools_to_responses(&request.tools);
    let tool_choice = if tools.is_empty() {
        None
    } else {
        request
            .tool_choice
            .or(config.tool_choice)
            .map(tool_choice_value)
    };
    Ok(ResponseRequest {
        model: config.model.clone(),
        instructions: combined_system(config, &request),
        input: messages_to_responses(&request.messages)?,
        stream: config.stream,
        tools,
        reasoning: reasoning(config),
        max_output_tokens: config.max_tokens,
        tool_choice,
    })
}

pub(crate) async fn decode_response(
    response: reqwest::Response,
) -> Result<ModelResponse, AgentError> {
    let raw: Value = response
        .json()
        .await
        .map_err(|err| AgentError::Model(format!("model response was not JSON: {err}")))?;
    model_response_from_value(raw, true)
}

pub(crate) fn model_response_from_value(
    raw: Value,
    include_message_text: bool,
) -> Result<ModelResponse, AgentError> {
    fail_if_error(&raw)?;
    let mut content = Vec::new();
    let mut saw_message = false;
    let mut saw_tool = false;
    if let Some(output) = raw.get("output").and_then(Value::as_array) {
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("reasoning") => extend_reasoning(&mut content, item),
                Some("message") => {
                    saw_message = true;
                    if include_message_text {
                        extend_message(&mut content, item);
                    }
                }
                Some("function_call") => {
                    saw_tool = true;
                    content.push(ContentBlock::ToolUse(tool_use_from_item(item)?));
                }
                _ => {}
            }
        }
    }
    let usage = raw.get("usage").cloned().map(decode_usage).transpose()?;
    Ok(ModelResponse {
        content,
        final_answer: saw_message,
        stop_reason: stop_reason(&raw, saw_tool, saw_message),
        response_id: raw
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        raw: Some(raw),
        usage,
        ..ModelResponse::default()
    })
}

pub(crate) fn tool_use_from_item(item: &Value) -> Result<ToolUse, AgentError> {
    let call_id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
        .ok_or_else(|| AgentError::Model("function_call missing call_id".to_string()))?;
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentError::Model("function_call missing name".to_string()))?;
    let arguments = item
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}");
    Ok(ToolUse {
        id: call_id.to_string(),
        name: name.to_string(),
        input: decode_arguments(arguments),
        raw: Some(item.clone()),
    })
}

pub(crate) fn usage_from_value(value: Value) -> Result<Usage, AgentError> {
    decode_usage(value)
}

fn reasoning(config: &AgentConfig) -> ResponseReasoning {
    let enabled = matches!(config.speed_profile, SpeedProfile::FlashWithAutoThinking)
        && config.thinking.enabled;
    ResponseReasoning {
        effort: if enabled {
            effort_value(config.thinking.reasoning_effort)
        } else {
            "none"
        },
    }
}

fn messages_to_responses(messages: &[AgentMessage]) -> Result<Vec<Value>, AgentError> {
    let mut out = Vec::new();
    for message in messages {
        match message {
            AgentMessage::User { content } => append_user_items(&mut out, content)?,
            AgentMessage::Assistant { content } => append_assistant_items(&mut out, content)?,
        }
    }
    Ok(out)
}

fn append_user_items(out: &mut Vec<Value>, content: &[ContentBlock]) -> Result<(), AgentError> {
    let mut text = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text { text: value } => text.push(value.clone()),
            ContentBlock::ConversationDigest(digest) => text.push(format!(
                "CONVERSATION_DIGEST_JSON:\n{}",
                serde_json::to_string(digest).map_err(|err| {
                    AgentError::Model(format!("failed to encode conversation digest: {err}"))
                })?
            )),
            ContentBlock::ToolResult(result) => {
                flush_message(out, "user", &mut text);
                out.push(json!({
                    "type": "function_call_output",
                    "call_id": result.tool_use_id,
                    "output": tool_result_content(&result.content)?
                }));
            }
            ContentBlock::Thinking { .. } | ContentBlock::ToolUse(_) => {
                return Err(AgentError::Model(
                    "user message contains assistant-only content block".to_string(),
                ));
            }
        }
    }
    flush_message(out, "user", &mut text);
    Ok(())
}

fn append_assistant_items(
    out: &mut Vec<Value>,
    content: &[ContentBlock],
) -> Result<(), AgentError> {
    let mut text = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text { text: value } => text.push(value.clone()),
            ContentBlock::Thinking { text: value, .. } => out.push(json!({
                "type": "reasoning",
                "content": [{ "type": "reasoning_text", "text": value }]
            })),
            ContentBlock::ToolUse(tool_use) => {
                flush_message(out, "assistant", &mut text);
                out.push(json!({
                    "type": "function_call",
                    "call_id": tool_use.id,
                    "name": tool_use.name,
                    "arguments": encode_arguments(&tool_use.input)?
                }));
            }
            ContentBlock::ConversationDigest(_) | ContentBlock::ToolResult(_) => {
                return Err(AgentError::Model(
                    "assistant message contains non-assistant content block".to_string(),
                ));
            }
        }
    }
    flush_message(out, "assistant", &mut text);
    Ok(())
}

fn flush_message(out: &mut Vec<Value>, role: &str, text: &mut Vec<String>) {
    if text.is_empty() {
        return;
    }
    out.push(json!({
        "type": "message",
        "role": role,
        "content": text.join("")
    }));
    text.clear();
}

fn tools_to_responses(tools: &[AgentToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": if tool.input_schema.is_null() {
                    json!({ "type": "object", "properties": {} })
                } else {
                    tool.input_schema.clone()
                }
            })
        })
        .collect()
}

fn extend_reasoning(content: &mut Vec<ContentBlock>, item: &Value) {
    let Some(parts) = item.get("content").and_then(Value::as_array) else {
        return;
    };
    let text = parts
        .iter()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<String>();
    if !text.is_empty() {
        content.push(ContentBlock::Thinking {
            text,
            signature: None,
        });
    }
}

fn extend_message(content: &mut Vec<ContentBlock>, item: &Value) {
    let Some(parts) = item.get("content").and_then(Value::as_array) else {
        return;
    };
    let text = parts
        .iter()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<String>();
    if !text.is_empty() {
        content.push(ContentBlock::Text { text });
    }
}

fn fail_if_error(raw: &Value) -> Result<(), AgentError> {
    let status = raw.get("status").and_then(Value::as_str);
    if !matches!(status, Some("failed" | "cancelled")) {
        return Ok(());
    }
    let message = raw
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("responses request failed");
    Err(AgentError::Model(message.to_string()))
}

fn stop_reason(raw: &Value, saw_tool: bool, saw_message: bool) -> Option<StopReason> {
    if saw_tool {
        return Some(StopReason::ToolUse);
    }
    match raw.get("status").and_then(Value::as_str) {
        Some("completed") if saw_message => Some(StopReason::EndTurn),
        Some("incomplete") => Some(incomplete_reason(raw)),
        Some("failed") | Some("cancelled") => Some(StopReason::Refusal),
        Some(other) => Some(StopReason::Other(other.to_string())),
        None if saw_message => Some(StopReason::EndTurn),
        None => None,
    }
}

fn incomplete_reason(raw: &Value) -> StopReason {
    match raw
        .get("incomplete_details")
        .and_then(|value| value.get("reason"))
        .and_then(Value::as_str)
    {
        Some("max_output_tokens") => StopReason::MaxTokens,
        Some("content_filter") => StopReason::Refusal,
        Some(other) => StopReason::Other(other.to_string()),
        None => StopReason::MaxTokens,
    }
}

fn decode_usage(value: Value) -> Result<Usage, AgentError> {
    let usage: ResponseUsage = serde_json::from_value(value)
        .map_err(|err| AgentError::Model(format!("response usage shape was invalid: {err}")))?;
    Ok(Usage {
        input_tokens: usage.input_tokens.unwrap_or_default(),
        output_tokens: usage.output_tokens.unwrap_or_default(),
    })
}
