use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::deepseek_stream::decode_stream;
use crate::{
    AgentConfig, AgentError, AgentMessage, AgentModel, AgentToolSpec, ContentBlock, ModelRequest,
    ModelResponse, ModelStream, ReasoningEffort, SpeedProfile, StopReason, ToolResult, ToolUse,
    Usage,
};

#[derive(Clone)]
pub struct ConfiguredModel {
    client: reqwest::Client,
    config: AgentConfig,
    endpoint: String,
}

impl ConfiguredModel {
    pub fn new(config: AgentConfig) -> Result<Self, AgentError> {
        validate_config(&config)?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .map_err(|err| AgentError::Config(format!("failed to build HTTP client: {err}")))?;
        let endpoint = chat_endpoint(&config.base_url);
        Ok(Self {
            client,
            config,
            endpoint,
        })
    }

    pub fn with_client(client: reqwest::Client, config: AgentConfig) -> Result<Self, AgentError> {
        validate_config(&config)?;
        let endpoint = chat_endpoint(&config.base_url);
        Ok(Self {
            client,
            config,
            endpoint,
        })
    }
}

#[async_trait]
impl AgentModel for ConfiguredModel {
    async fn next_step(
        &self,
        request: ModelRequest,
        stream: ModelStream,
    ) -> Result<ModelResponse, AgentError> {
        let body = chat_request(&self.config, request)?;
        let response = self
            .client
            .post(&self.endpoint)
            .headers(headers(&self.config)?)
            .json(&body)
            .send()
            .await
            .map_err(|err| AgentError::Model(format!("model request failed: {err}")))?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|err| format!("failed to read error body: {err}"));
            return Err(AgentError::Model(format!(
                "model request failed ({status}): {body}"
            )));
        }
        if self.config.stream {
            decode_stream(response, stream).await
        } else {
            decode_response(response).await
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    id: Option<String>,
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ChatToolCall>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ChatToolCall {
    id: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    function: ChatFunctionCall,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ChatFunctionCall {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatUsage {
    pub(crate) prompt_tokens: Option<u64>,
    pub(crate) completion_tokens: Option<u64>,
}

fn validate_config(config: &AgentConfig) -> Result<(), AgentError> {
    if config.api_key.trim().is_empty() {
        return Err(AgentError::Config("api_key must not be empty".to_string()));
    }
    if config.model.trim().is_empty() {
        return Err(AgentError::Config("model must not be empty".to_string()));
    }
    if config.base_url.trim().is_empty() {
        return Err(AgentError::Config("base_url must not be empty".to_string()));
    }
    if config.request_timeout_secs == 0 {
        return Err(AgentError::Config(
            "request_timeout_secs must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn chat_endpoint(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    }
}

fn headers(config: &AgentConfig) -> Result<HeaderMap, AgentError> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", config.api_key.trim())).map_err(|err| {
            AgentError::Config(format!("invalid authorization header value: {err}"))
        })?,
    );
    Ok(headers)
}

fn chat_request(config: &AgentConfig, request: ModelRequest) -> Result<ChatRequest, AgentError> {
    let mut messages = Vec::new();
    if let Some(system) = config
        .system
        .as_ref()
        .filter(|value| !value.trim().is_empty())
    {
        messages.push(json!({ "role": "system", "content": system }));
    }
    messages.extend(messages_to_chat(&request.messages)?);
    let thinking = thinking_body(config);
    Ok(ChatRequest {
        model: config.model.clone(),
        messages,
        stream: config.stream,
        tools: tools_to_chat(&request.tools),
        reasoning_effort: thinking
            .as_ref()
            .map(|_| effort_value(config.thinking.reasoning_effort)),
        thinking,
        max_tokens: config.max_tokens,
    })
}

fn thinking_body(config: &AgentConfig) -> Option<Value> {
    match config.speed_profile {
        SpeedProfile::Flash => None,
        SpeedProfile::FlashWithAutoThinking if config.thinking.enabled => {
            Some(json!({ "type": "enabled" }))
        }
        SpeedProfile::FlashWithAutoThinking => None,
    }
}

fn effort_value(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
    }
}

fn messages_to_chat(messages: &[AgentMessage]) -> Result<Vec<Value>, AgentError> {
    let mut out = Vec::new();
    for message in messages {
        match message {
            AgentMessage::User { content } => append_user_messages(&mut out, content)?,
            AgentMessage::Assistant { content } => out.push(assistant_message(content)?),
        }
    }
    Ok(out)
}

fn append_user_messages(out: &mut Vec<Value>, content: &[ContentBlock]) -> Result<(), AgentError> {
    let mut text = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text { text: value } => text.push(value.clone()),
            ContentBlock::ConversationDigest(digest) => {
                text.push(format!(
                    "CONVERSATION_DIGEST_JSON:\n{}",
                    serde_json::to_string(digest).map_err(|err| {
                        AgentError::Model(format!("failed to encode conversation digest: {err}"))
                    })?
                ));
            }
            ContentBlock::ToolResult(result) => {
                flush_user_text(out, &mut text);
                out.push(tool_result_message(result)?);
            }
            ContentBlock::Thinking { .. } | ContentBlock::ToolUse(_) => {
                return Err(AgentError::Model(
                    "user message contains assistant-only content block".to_string(),
                ));
            }
        }
    }
    flush_user_text(out, &mut text);
    Ok(())
}

fn flush_user_text(out: &mut Vec<Value>, text: &mut Vec<String>) {
    if text.is_empty() {
        return;
    }
    out.push(json!({ "role": "user", "content": text.join("\n") }));
    text.clear();
}

fn assistant_message(content: &[ContentBlock]) -> Result<Value, AgentError> {
    let mut text = Vec::new();
    let mut reasoning = Vec::new();
    let mut tool_calls = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text { text: value } => text.push(value.clone()),
            ContentBlock::Thinking { text: value, .. } => reasoning.push(value.clone()),
            ContentBlock::ToolUse(tool_use) => tool_calls.push(tool_call_to_chat(tool_use)?),
            ContentBlock::ConversationDigest(_) | ContentBlock::ToolResult(_) => {
                return Err(AgentError::Model(
                    "assistant message contains non-assistant content block".to_string(),
                ));
            }
        }
    }
    let mut value = Map::new();
    value.insert("role".to_string(), Value::String("assistant".to_string()));
    value.insert("content".to_string(), Value::String(text.join("")));
    if !reasoning.is_empty() {
        value.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning.join("")),
        );
    }
    if !tool_calls.is_empty() {
        value.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }
    Ok(Value::Object(value))
}

fn tool_call_to_chat(tool_use: &ToolUse) -> Result<Value, AgentError> {
    if let Some(raw) = tool_use.raw.as_ref() {
        return Ok(raw.clone());
    }
    Ok(json!({
        "id": tool_use.id,
        "type": "function",
        "function": {
            "name": tool_use.name,
            "arguments": encode_arguments(&tool_use.input)?
        }
    }))
}

fn tool_result_message(result: &ToolResult) -> Result<Value, AgentError> {
    Ok(json!({
        "role": "tool",
        "tool_call_id": result.tool_use_id,
        "content": tool_result_content(&result.content)?
    }))
}

fn tools_to_chat(tools: &[AgentToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": if tool.input_schema.is_null() {
                        json!({ "type": "object", "properties": {} })
                    } else {
                        tool.input_schema.clone()
                    }
                }
            })
        })
        .collect()
}

fn decode_choice(
    response_id: Option<String>,
    choice: ChatChoice,
    usage: Option<ChatUsage>,
    raw: Value,
) -> Result<ModelResponse, AgentError> {
    let mut content = Vec::new();
    if let Some(reasoning_content) = choice
        .message
        .reasoning_content
        .filter(|value| !value.is_empty())
    {
        content.push(ContentBlock::Thinking {
            text: reasoning_content,
            signature: None,
        });
    }
    if let Some(text) = choice.message.content.filter(|value| !value.is_empty()) {
        content.push(ContentBlock::Text { text });
    }
    for tool_call in choice.message.tool_calls {
        content.push(ContentBlock::ToolUse(tool_use_from_tool_call(tool_call)?));
    }
    Ok(ModelResponse {
        content,
        stop_reason: choice.finish_reason.map(stop_reason),
        response_id,
        raw: Some(raw),
        usage: usage.map(usage_from_chat),
        ..ModelResponse::default()
    })
}

async fn decode_response(response: reqwest::Response) -> Result<ModelResponse, AgentError> {
    let raw: Value = response
        .json()
        .await
        .map_err(|err| AgentError::Model(format!("model response was not JSON: {err}")))?;
    let decoded: ChatResponse = serde_json::from_value(raw.clone())
        .map_err(|err| AgentError::Model(format!("model response shape was invalid: {err}")))?;
    let choice =
        decoded.choices.into_iter().next().ok_or_else(|| {
            AgentError::Model("model response did not include a choice".to_string())
        })?;
    decode_choice(decoded.id, choice, decoded.usage, raw)
}

fn tool_use_from_tool_call(tool_call: ChatToolCall) -> Result<ToolUse, AgentError> {
    let raw = serde_json::to_value(&tool_call)
        .map_err(|err| AgentError::Model(format!("failed to preserve tool call: {err}")))?;
    Ok(ToolUse {
        id: tool_call
            .id
            .ok_or_else(|| AgentError::Model("tool call missing id".to_string()))?,
        name: tool_call
            .function
            .name
            .ok_or_else(|| AgentError::Model("tool call missing function name".to_string()))?,
        input: decode_arguments(tool_call.function.arguments.as_deref().unwrap_or("{}")),
        raw: Some(raw),
    })
}

pub(crate) fn usage_from_chat(usage: ChatUsage) -> Usage {
    Usage {
        input_tokens: usage.prompt_tokens.unwrap_or_default(),
        output_tokens: usage.completion_tokens.unwrap_or_default(),
    }
}

pub(crate) fn stop_reason(value: String) -> StopReason {
    match value.as_str() {
        "stop" => StopReason::EndTurn,
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        "content_filter" => StopReason::Refusal,
        other => StopReason::Other(other.to_string()),
    }
}

fn encode_arguments(input: &Value) -> Result<String, AgentError> {
    match input {
        Value::Null => Ok("{}".to_string()),
        other => serde_json::to_string(other)
            .map_err(|err| AgentError::Model(format!("failed to encode tool arguments: {err}"))),
    }
}

pub(crate) fn decode_arguments(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()))
}

fn tool_result_content(content: &Value) -> Result<String, AgentError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(value) => Ok(value.clone()),
        other => serde_json::to_string(other)
            .map_err(|err| AgentError::Model(format!("failed to encode tool result: {err}"))),
    }
}
