use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::deepseek::{
    combined_system, decode_arguments, effort_value, encode_arguments, tool_choice_value,
    tool_result_content,
};
use crate::{
    AgentConfig, AgentError, AgentMessage, AgentToolSpec, ContentBlock, ModelOutcome, ModelRequest,
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
    let tool_choice = if request.tool_choice == Some(crate::ToolChoice::None) {
        Some("none")
    } else if tools.is_empty() {
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
    model_response_from_value(raw)
}

pub(crate) fn model_response_from_value(raw: Value) -> Result<ModelResponse, AgentError> {
    fail_if_error(&raw)?;
    let decoded = decode_output(&raw, true)?;
    let usage = raw.get("usage").cloned().map(decode_usage).transpose()?;
    let stop_reason = stop_reason(&raw, &decoded);
    Ok(ModelResponse {
        outcome: decoded.outcome,
        stop_reason,
        response_id: raw
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        raw: Some(raw),
        usage,
        ..ModelResponse::default()
    })
}

pub(crate) fn model_response_shape(
    raw: &Value,
    include_message_text: bool,
) -> Result<DecodedOutput, AgentError> {
    fail_if_error(raw)?;
    decode_output(raw, include_message_text)
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

#[derive(Debug)]
pub(crate) struct DecodedOutput {
    pub(crate) outcome: Option<ModelOutcome>,
    pub(crate) message_seen: bool,
    pub(crate) tool_seen: bool,
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

fn decode_output(raw: &Value, include_message_text: bool) -> Result<DecodedOutput, AgentError> {
    let output = raw
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| AgentError::Model("responses output must be an array".to_string()))?;
    let mut kind = OutputKind::Undecided;
    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => {}
            Some("message") => {
                let text = message_text(item)?;
                kind = kind.message(if include_message_text {
                    text
                } else {
                    String::new()
                })?;
            }
            Some("function_call") => {
                let call = tool_use_from_item(item)?;
                kind = kind.function_call(call)?;
            }
            Some(other) => {
                return Err(AgentError::Model(format!(
                    "unsupported responses output item: {other}"
                )));
            }
            None => {
                return Err(AgentError::Model(
                    "responses output item missing type".to_string(),
                ));
            }
        }
    }
    Ok(kind.into_decoded())
}

fn message_text(item: &Value) -> Result<String, AgentError> {
    let parts = item
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| AgentError::Model("message output missing content".to_string()))?;
    let mut text = String::new();
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("output_text") => {
                let part_text = part
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| AgentError::Model("output_text missing text".to_string()))?;
                text.push_str(part_text);
            }
            Some("refusal") => {
                return Err(AgentError::Model("model returned a refusal".to_string()));
            }
            Some(other) => {
                return Err(AgentError::Model(format!(
                    "unsupported message content item: {other}"
                )));
            }
            None => {
                return Err(AgentError::Model(
                    "message content item missing type".to_string(),
                ));
            }
        }
    }
    Ok(text)
}

enum OutputKind {
    Undecided,
    Message(String),
    FunctionCalls(Vec<ToolUse>),
}

impl OutputKind {
    fn message(self, text: String) -> Result<Self, AgentError> {
        match self {
            Self::Undecided => Ok(Self::Message(text)),
            Self::Message(mut existing) => {
                existing.push_str(&text);
                Ok(Self::Message(existing))
            }
            Self::FunctionCalls(_) => Err(AgentError::Model(
                "responses output mixed messages and function calls".to_string(),
            )),
        }
    }

    fn function_call(self, call: ToolUse) -> Result<Self, AgentError> {
        match self {
            Self::Undecided => Ok(Self::FunctionCalls(vec![call])),
            Self::FunctionCalls(mut calls) => {
                calls.push(call);
                Ok(Self::FunctionCalls(calls))
            }
            Self::Message(_) => Err(AgentError::Model(
                "responses output mixed messages and function calls".to_string(),
            )),
        }
    }

    fn into_decoded(self) -> DecodedOutput {
        match self {
            Self::Undecided => DecodedOutput {
                outcome: None,
                message_seen: false,
                tool_seen: false,
            },
            Self::Message(text) => DecodedOutput {
                outcome: Some(ModelOutcome::FinalAnswer { text }),
                message_seen: true,
                tool_seen: false,
            },
            Self::FunctionCalls(calls) => DecodedOutput {
                outcome: Some(ModelOutcome::ToolCalls { calls }),
                message_seen: false,
                tool_seen: true,
            },
        }
    }
}

fn stop_reason(raw: &Value, decoded: &DecodedOutput) -> Option<StopReason> {
    match raw.get("status").and_then(Value::as_str) {
        Some("incomplete") => Some(incomplete_reason(raw)),
        Some("completed") if decoded.message_seen => Some(StopReason::EndTurn),
        Some("completed") if decoded.tool_seen => Some(StopReason::ToolUse),
        Some("failed") | Some("cancelled") => Some(StopReason::Refusal),
        Some(other) => Some(StopReason::Other(other.to_string())),
        None if decoded.message_seen => Some(StopReason::EndTurn),
        None if decoded.tool_seen => Some(StopReason::ToolUse),
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
