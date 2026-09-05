use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::deepseek::{
    combined_system, decode_arguments, effort_value, encode_arguments, tool_choice_value,
    tool_result_content,
};
use crate::{
    AgentConfig, AgentError, AgentMessage, AgentToolSpec, AssistantMessageId, AssistantMessageItem,
    AssistantMessagePhase, AssistantTextPart, ContentBlock, ModelRequest, ModelResponse,
    ResponseContentIndex, ResponseOutputIndex, SpeedProfile, StopReason, ToolUse, Usage,
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
) -> Result<ModelResponse, DecodeFailure> {
    let raw: Value = response
        .json()
        .await
        .map_err(|err| DecodeFailure::body(err, "model response was not JSON"))?;
    model_response_from_value(raw).map_err(DecodeFailure::protocol)
}

#[derive(Debug)]
pub(crate) enum DecodeFailure {
    Transport(String),
    Protocol(String),
}

impl DecodeFailure {
    pub(crate) fn body(err: reqwest::Error, context: &str) -> Self {
        let message = format!("{context}: {err}");
        if err.is_decode() {
            Self::Protocol(message)
        } else {
            Self::Transport(message)
        }
    }

    pub(crate) fn protocol(err: AgentError) -> Self {
        Self::Protocol(err.to_string())
    }

    pub(crate) fn protocol_message(message: impl Into<String>) -> Self {
        Self::Protocol(message.into())
    }

    pub(crate) fn message(&self) -> &str {
        match self {
            Self::Transport(message) | Self::Protocol(message) => message,
        }
    }

    pub(crate) fn retryable(&self) -> bool {
        matches!(self, Self::Transport(_))
    }
}

impl From<AgentError> for DecodeFailure {
    fn from(err: AgentError) -> Self {
        Self::protocol(err)
    }
}

pub(crate) fn model_response_from_value(raw: Value) -> Result<ModelResponse, AgentError> {
    fail_if_error(&raw)?;
    let decoded = decode_output(&raw)?;
    let usage = raw.get("usage").cloned().map(decode_usage).transpose()?;
    let stop_reason = stop_reason(&raw, &decoded);
    Ok(ModelResponse {
        assistant_messages: decoded.assistant_messages,
        tool_calls: decoded.tool_calls,
        reasoning: decoded.reasoning,
        stop_reason,
        response_id: raw
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        raw: Some(raw),
        usage,
    })
}

pub(crate) fn model_response_shape(raw: &Value) -> Result<DecodedOutput, AgentError> {
    fail_if_error(raw)?;
    decode_output(raw)
}

pub(crate) fn tool_use_from_item(item: &Value) -> Result<ToolUse, AgentError> {
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentError::Model("function_call missing call_id".to_string()))?;
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentError::Model("function_call missing name".to_string()))?;
    let arguments = item
        .get("arguments")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentError::Model("function_call missing arguments".to_string()))?;
    Ok(ToolUse {
        id: call_id.to_string(),
        name: name.to_string(),
        input: decode_arguments(arguments),
        raw: Some(item.clone()),
    })
}

#[derive(Debug)]
pub(crate) struct DecodedOutput {
    pub(crate) assistant_messages: Vec<AssistantMessageItem>,
    pub(crate) tool_calls: Vec<ToolUse>,
    pub(crate) reasoning: Vec<String>,
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
            ContentBlock::AssistantMessage(_)
            | ContentBlock::Thinking { .. }
            | ContentBlock::ToolUse(_) => {
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
            ContentBlock::AssistantMessage(message) => {
                flush_message(out, "assistant", &mut text);
                out.push(json!({
                    "id": message.id.as_str(),
                    "type": "message",
                    "role": "assistant",
                    "phase": phase_value(message.phase),
                    "content": assistant_content(message)?
                }));
            }
            ContentBlock::Thinking { text: value, .. } => {
                flush_message(out, "assistant", &mut text);
                out.push(json!({
                    "type": "reasoning",
                    "content": [{ "type": "reasoning_text", "text": value }]
                }));
            }
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

fn assistant_content(message: &AssistantMessageItem) -> Result<Vec<Value>, AgentError> {
    message
        .content
        .iter()
        .enumerate()
        .map(|(index, part)| {
            if part.index.get() != index {
                return Err(AgentError::Model(format!(
                    "assistant message content index {} did not match position {index}",
                    part.index.get()
                )));
            }
            Ok(json!({
                "type": "output_text",
                "text": part.text,
            }))
        })
        .collect()
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

fn phase_value(phase: AssistantMessagePhase) -> &'static str {
    match phase {
        AssistantMessagePhase::Commentary => "commentary",
        AssistantMessagePhase::FinalAnswer => "final_answer",
    }
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

fn decode_output(raw: &Value) -> Result<DecodedOutput, AgentError> {
    let items = raw
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| AgentError::Model("responses output must be an array".to_string()))?;
    let mut assistant_messages = Vec::new();
    let mut tool_calls = Vec::new();
    let mut reasoning = Vec::new();
    for (index, item) in items.iter().enumerate() {
        if matches!(
            item.get("status").and_then(Value::as_str),
            Some("in_progress" | "incomplete")
        ) {
            continue;
        }
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => extend_reasoning(&mut reasoning, item)?,
            Some("message") => assistant_messages.push(assistant_message_from_item(
                item,
                ResponseOutputIndex::new(index),
            )?),
            Some("function_call") => tool_calls.push(tool_use_from_item(item)?),
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
    Ok(DecodedOutput {
        assistant_messages,
        tool_calls,
        reasoning,
    })
}

fn extend_reasoning(content: &mut Vec<String>, item: &Value) -> Result<(), AgentError> {
    let Some(parts) = item.get("content").and_then(Value::as_array) else {
        return Ok(());
    };
    let text = parts.iter().try_fold(String::new(), |mut text, part| {
        match part.get("type").and_then(Value::as_str) {
            Some("reasoning_text") => {
                let part_text = part
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| AgentError::Model("reasoning_text missing text".to_string()))?;
                text.push_str(part_text);
                Ok(text)
            }
            Some(other) => Err(AgentError::Model(format!(
                "unsupported reasoning content item: {other}"
            ))),
            None => Err(AgentError::Model(
                "reasoning content item missing type".to_string(),
            )),
        }
    })?;
    if !text.is_empty() {
        content.push(text);
    }
    Ok(())
}

pub(crate) fn assistant_message_from_item(
    item: &Value,
    output_index: ResponseOutputIndex,
) -> Result<AssistantMessageItem, AgentError> {
    let id = item
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| AgentError::Model("message output missing id".to_string()))?;
    if item.get("role").and_then(Value::as_str) != Some("assistant") {
        return Err(AgentError::Model(
            "message output role must be assistant".to_string(),
        ));
    }
    if item
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|status| status != "completed")
    {
        return Err(AgentError::Model(
            "message output was not completed".to_string(),
        ));
    }
    let phase = assistant_message_phase(item)?;
    let parts = item
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| AgentError::Model("message output missing content".to_string()))?;
    let content = parts
        .iter()
        .enumerate()
        .map(
            |(index, part)| match part.get("type").and_then(Value::as_str) {
                Some("output_text") => {
                    let text = part
                        .get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| AgentError::Model("output_text missing text".to_string()))?;
                    Ok(AssistantTextPart {
                        index: ResponseContentIndex::new(index),
                        text: text.to_string(),
                    })
                }
                Some("refusal") => Err(AgentError::Model("model returned a refusal".to_string())),
                Some(other) => Err(AgentError::Model(format!(
                    "unsupported message content item: {other}"
                ))),
                None => Err(AgentError::Model(
                    "message content item missing type".to_string(),
                )),
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AssistantMessageItem {
        id: AssistantMessageId::new(id),
        output_index,
        phase,
        content,
    })
}

pub(crate) fn assistant_message_phase(item: &Value) -> Result<AssistantMessagePhase, AgentError> {
    match item.get("phase").and_then(Value::as_str) {
        Some("commentary") => Ok(AssistantMessagePhase::Commentary),
        Some("final_answer") => Ok(AssistantMessagePhase::FinalAnswer),
        Some(other) => Err(AgentError::Model(format!(
            "unsupported message output phase: {other}"
        ))),
        None => Err(AgentError::Model(
            "message output missing phase".to_string(),
        )),
    }
}

fn stop_reason(raw: &Value, decoded: &DecodedOutput) -> Option<StopReason> {
    match raw.get("status").and_then(Value::as_str) {
        Some("incomplete") => Some(incomplete_reason(raw)),
        Some("completed") if !decoded.tool_calls.is_empty() => Some(StopReason::ToolUse),
        Some("completed") if !decoded.assistant_messages.is_empty() => Some(StopReason::EndTurn),
        Some("failed") | Some("cancelled") => Some(StopReason::Refusal),
        Some(other) => Some(StopReason::Other(other.to_string())),
        None if !decoded.tool_calls.is_empty() => Some(StopReason::ToolUse),
        None if !decoded.assistant_messages.is_empty() => Some(StopReason::EndTurn),
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
