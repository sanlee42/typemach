use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::json;

use crate::deepseek::{ChatUsage, decode_arguments, stop_reason, usage_from_chat};
use crate::{AgentError, ContentBlock, ModelResponse, ModelStream, StopReason, ToolUse, Usage};

#[derive(Debug, Deserialize)]
struct StreamChunk {
    id: Option<String>,
    choices: Vec<StreamChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<StreamToolCallDelta>,
}

#[derive(Debug, Clone, Deserialize)]
struct StreamToolCallDelta {
    index: usize,
    id: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    function: Option<StreamFunctionDelta>,
}

#[derive(Debug, Clone, Deserialize)]
struct StreamFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct ToolCallBuilder {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[derive(Debug, Default)]
struct StreamAccumulator {
    response_id: Option<String>,
    stop_reason: Option<StopReason>,
    reasoning_content: String,
    tool_calls: Vec<ToolCallBuilder>,
    usage: Option<Usage>,
}

pub(crate) async fn decode_stream(
    response: reqwest::Response,
    stream: ModelStream,
) -> Result<ModelResponse, AgentError> {
    let mut bytes = response.bytes_stream();
    let mut pending = String::new();
    let mut acc = StreamAccumulator::default();
    while let Some(chunk) = bytes.next().await {
        let chunk =
            chunk.map_err(|err| AgentError::Model(format!("model stream failed: {err}")))?;
        pending.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(index) = pending.find('\n') {
            let line = pending[..index].trim_end_matches('\r').to_string();
            pending.drain(..=index);
            handle_stream_line(&line, &stream, &mut acc)?;
        }
    }
    if !pending.trim().is_empty() {
        handle_stream_line(pending.trim_end_matches('\r'), &stream, &mut acc)?;
    }
    let mut content = Vec::new();
    if !acc.reasoning_content.is_empty() {
        content.push(ContentBlock::Thinking {
            text: acc.reasoning_content,
            signature: None,
        });
    }
    for builder in acc.tool_calls {
        content.push(ContentBlock::ToolUse(tool_use_from_builder(builder)?));
    }
    Ok(ModelResponse {
        content,
        stop_reason: acc.stop_reason,
        response_id: acc.response_id,
        usage: acc.usage,
        ..ModelResponse::default()
    })
}

fn handle_stream_line(
    line: &str,
    stream: &ModelStream,
    acc: &mut StreamAccumulator,
) -> Result<(), AgentError> {
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(());
    };
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(());
    }
    let chunk: StreamChunk = serde_json::from_str(data)
        .map_err(|err| AgentError::Model(format!("model stream chunk was invalid: {err}")))?;
    if acc.response_id.is_none() {
        acc.response_id = chunk.id;
    }
    if let Some(usage) = chunk.usage {
        acc.usage = Some(usage_from_chat(usage));
    }
    for choice in chunk.choices {
        if let Some(reason) = choice.finish_reason {
            acc.stop_reason = Some(stop_reason(reason));
        }
        if let Some(delta) = choice.delta.reasoning_content {
            acc.reasoning_content.push_str(&delta);
        }
        if let Some(delta) = choice.delta.content {
            stream.delta(delta)?;
        }
        for tool_delta in choice.delta.tool_calls {
            apply_tool_delta(acc, tool_delta);
        }
    }
    Ok(())
}

fn apply_tool_delta(acc: &mut StreamAccumulator, delta: StreamToolCallDelta) {
    while acc.tool_calls.len() <= delta.index {
        acc.tool_calls.push(ToolCallBuilder::default());
    }
    let builder = &mut acc.tool_calls[delta.index];
    if delta.id.is_some() {
        builder.id = delta.id;
    }
    if let Some(function) = delta.function {
        if function.name.is_some() {
            builder.name = function.name;
        }
        if let Some(arguments) = function.arguments {
            builder.arguments.push_str(&arguments);
        }
    }
    let _ = delta.kind;
}

fn tool_use_from_builder(builder: ToolCallBuilder) -> Result<ToolUse, AgentError> {
    let id = builder
        .id
        .ok_or_else(|| AgentError::Model("streamed tool call missing id".to_string()))?;
    let name = builder
        .name
        .ok_or_else(|| AgentError::Model("streamed tool call missing function name".to_string()))?;
    Ok(ToolUse {
        id: id.clone(),
        name: name.clone(),
        input: decode_arguments(&builder.arguments),
        raw: Some(json!({
            "id": id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": builder.arguments
            }
        })),
    })
}
