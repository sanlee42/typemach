use std::collections::BTreeMap;

use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::Value;

use crate::responses::{model_response_from_value, tool_use_from_item, usage_from_value};
use crate::{AgentError, ContentBlock, ModelResponse, ModelStream, StopReason, ToolUse, Usage};

#[derive(Debug, Deserialize)]
struct Event {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    output_index: Option<usize>,
    #[serde(default)]
    item: Option<Value>,
    #[serde(default)]
    response: Option<Value>,
}

#[derive(Debug, Default)]
struct ToolCallBuilder {
    call_id: Option<String>,
    name: Option<String>,
    arguments: String,
    raw: Option<Value>,
}

#[derive(Debug, Default)]
struct Accumulator {
    response_id: Option<String>,
    status: Option<String>,
    reasoning_text: String,
    final_text: String,
    final_answer: bool,
    tool_calls: BTreeMap<usize, ToolCallBuilder>,
    usage: Option<Usage>,
    stop_reason: Option<StopReason>,
}

pub(crate) async fn decode_stream(
    response: reqwest::Response,
    stream: ModelStream,
) -> Result<ModelResponse, AgentError> {
    let mut bytes = response.bytes_stream();
    let mut pending = String::new();
    let mut acc = Accumulator::default();
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
    let stop_reason = acc.stop_reason.clone().or_else(|| inferred_stop(&acc));
    let mut content = Vec::new();
    if !acc.reasoning_text.is_empty() {
        content.push(ContentBlock::Thinking {
            text: acc.reasoning_text,
            signature: None,
        });
    }
    if !acc.final_text.is_empty() {
        content.push(ContentBlock::Text {
            text: acc.final_text,
        });
    }
    for builder in acc.tool_calls.into_values() {
        content.push(ContentBlock::ToolUse(tool_use_from_builder(builder)?));
    }
    Ok(ModelResponse {
        content,
        final_answer: acc.final_answer,
        stop_reason,
        response_id: acc.response_id,
        usage: acc.usage,
        ..ModelResponse::default()
    })
}

fn handle_stream_line(
    line: &str,
    stream: &ModelStream,
    acc: &mut Accumulator,
) -> Result<(), AgentError> {
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(());
    };
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(());
    }
    let event: Event = serde_json::from_str(data)
        .map_err(|err| AgentError::Model(format!("responses stream event was invalid: {err}")))?;
    match event.kind.as_str() {
        "response.output_text.delta" => {
            let Some(delta) = event.delta else {
                return Ok(());
            };
            acc.final_answer = true;
            acc.final_text.push_str(&delta);
            stream.final_delta(delta)?;
        }
        "response.reasoning_text.delta" => {
            if let Some(delta) = event.delta {
                acc.reasoning_text.push_str(&delta);
            }
        }
        "response.function_call_arguments.delta" => {
            let Some(index) = event.output_index else {
                return Ok(());
            };
            let builder = acc.tool_calls.entry(index).or_default();
            if let Some(delta) = event.delta {
                builder.arguments.push_str(&delta);
            }
        }
        "response.output_item.added" | "response.output_item.done" => {
            if let (Some(index), Some(item)) = (event.output_index, event.item) {
                merge_item(acc, index, item)?;
            }
        }
        "response.completed" | "response.incomplete" => {
            if let Some(response) = event.response {
                merge_response(acc, response)?;
            }
        }
        "response.failed" => {
            acc.stop_reason = Some(StopReason::Refusal);
            if let Some(response) = event.response {
                merge_response(acc, response)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn merge_item(acc: &mut Accumulator, index: usize, item: Value) -> Result<(), AgentError> {
    match item.get("type").and_then(Value::as_str) {
        Some("function_call") => {
            let builder = acc.tool_calls.entry(index).or_default();
            if let Some(call_id) = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(Value::as_str)
            {
                builder.call_id = Some(call_id.to_string());
            }
            if let Some(name) = item.get("name").and_then(Value::as_str) {
                builder.name = Some(name.to_string());
            }
            if let Some(arguments) = item.get("arguments").and_then(Value::as_str)
                && builder.arguments.is_empty()
            {
                builder.arguments.push_str(arguments);
            }
            builder.raw = Some(item);
        }
        Some("message") => {
            acc.final_answer = true;
        }
        _ => {}
    }
    Ok(())
}

fn merge_response(acc: &mut Accumulator, response: Value) -> Result<(), AgentError> {
    if acc.response_id.is_none() {
        acc.response_id = response
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string);
    }
    if acc.status.is_none() {
        acc.status = response
            .get("status")
            .and_then(Value::as_str)
            .map(ToString::to_string);
    }
    if acc.usage.is_none()
        && let Some(usage) = response.get("usage").cloned()
    {
        acc.usage = Some(usage_from_value(usage)?);
    }
    let decoded = model_response_from_value(response, false)?;
    if decoded.final_answer {
        acc.final_answer = true;
    }
    if let Some(reason) = decoded.stop_reason {
        acc.stop_reason = Some(reason);
    }
    for block in decoded.content {
        if let ContentBlock::ToolUse(tool) = block {
            upsert_tool(acc, tool);
        }
    }
    Ok(())
}

fn upsert_tool(acc: &mut Accumulator, tool: ToolUse) {
    if let Some((_, builder)) = acc
        .tool_calls
        .iter_mut()
        .find(|(_, builder)| builder.call_id.as_deref() == Some(tool.id.as_str()))
    {
        builder.arguments = arguments_from_tool(&tool);
        builder.call_id = Some(tool.id);
        builder.name = Some(tool.name);
        builder.raw = tool.raw;
        return;
    }
    let index = acc.tool_calls.keys().next_back().copied().unwrap_or(0) + 1;
    acc.tool_calls.insert(
        index,
        ToolCallBuilder {
            call_id: Some(tool.id.clone()),
            name: Some(tool.name.clone()),
            arguments: arguments_from_tool(&tool),
            raw: tool.raw,
        },
    );
}

fn arguments_from_tool(tool: &ToolUse) -> String {
    tool.raw
        .as_ref()
        .and_then(|raw| raw.get("arguments"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| tool.input.to_string())
}

fn tool_use_from_builder(builder: ToolCallBuilder) -> Result<ToolUse, AgentError> {
    if let Some(raw) = builder.raw.as_ref() {
        return tool_use_from_item(raw);
    }
    let call_id = builder
        .call_id
        .ok_or_else(|| AgentError::Model("function_call missing call_id".to_string()))?;
    let name = builder
        .name
        .ok_or_else(|| AgentError::Model("function_call missing name".to_string()))?;
    Ok(ToolUse {
        id: call_id,
        name,
        input: serde_json::from_str(&builder.arguments).unwrap_or(Value::String(builder.arguments)),
        raw: None,
    })
}

fn inferred_stop(acc: &Accumulator) -> Option<StopReason> {
    if !acc.tool_calls.is_empty() {
        return Some(StopReason::ToolUse);
    }
    if acc.final_answer {
        return Some(StopReason::EndTurn);
    }
    acc.status
        .as_ref()
        .map(|status| StopReason::Other(status.clone()))
}
