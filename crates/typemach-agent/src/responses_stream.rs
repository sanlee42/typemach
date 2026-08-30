use std::collections::BTreeMap;

use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::Value;

use crate::responses::{model_response_shape, tool_use_from_item, usage_from_value};
use crate::{AgentError, ModelOutcome, ModelResponse, ModelStream, StopReason, ToolUse, Usage};

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
    output: OutputKind,
    completed_text: String,
    streamed_text: bool,
    usage: Option<Usage>,
    stop_reason: Option<StopReason>,
}

#[derive(Debug, Default)]
enum OutputKind {
    #[default]
    Undecided,
    Message,
    FunctionCalls(BTreeMap<usize, ToolCallBuilder>),
}

pub(crate) async fn decode_stream(
    response: reqwest::Response,
    stream: ModelStream,
) -> Result<ModelResponse, AgentError> {
    let mut bytes = response.bytes_stream();
    let mut pending = Vec::new();
    let mut acc = Accumulator::default();
    while let Some(chunk) = bytes.next().await {
        let chunk =
            chunk.map_err(|err| AgentError::Model(format!("model stream failed: {err}")))?;
        pending.extend_from_slice(&chunk);
        while let Some(index) = pending.iter().position(|byte| *byte == b'\n') {
            let line = pending.drain(..=index).collect::<Vec<_>>();
            handle_stream_line(decode_line(line)?, &stream, &mut acc)?;
        }
    }
    if pending.iter().any(|byte| !byte.is_ascii_whitespace()) {
        handle_stream_line(decode_line(pending)?, &stream, &mut acc)?;
    }
    let stop_reason = acc.stop_reason.clone().or_else(|| inferred_stop(&acc));
    if acc.status.is_none() {
        return Err(AgentError::Model(
            "responses stream ended without a terminal event".to_string(),
        ));
    }
    let response_id = acc.response_id.clone();
    let usage = acc.usage.clone();
    Ok(ModelResponse {
        outcome: outcome(acc)?,
        stop_reason,
        response_id,
        usage,
        ..ModelResponse::default()
    })
}

fn decode_line(mut line: Vec<u8>) -> Result<String, AgentError> {
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    String::from_utf8(line)
        .map_err(|err| AgentError::Model(format!("responses stream line was not UTF-8: {err}")))
}

fn handle_stream_line(
    line: String,
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
                return Err(AgentError::Model(
                    "output_text delta missing text".to_string(),
                ));
            };
            acc.message()?;
            acc.streamed_text = true;
            stream.output_text(delta)?;
        }
        "response.function_call_arguments.delta" => {
            let index = event.output_index.ok_or_else(|| {
                AgentError::Model("function_call delta missing output_index".to_string())
            })?;
            let builder = acc.function_call(index)?;
            if let Some(delta) = event.delta {
                builder.arguments.push_str(&delta);
            }
        }
        "response.output_item.added" | "response.output_item.done" => {
            let index = event.output_index.ok_or_else(|| {
                AgentError::Model("output item event missing output_index".to_string())
            })?;
            let item = event
                .item
                .ok_or_else(|| AgentError::Model("output item event missing item".to_string()))?;
            merge_item(acc, index, item)?;
        }
        "response.completed" | "response.incomplete" => {
            let response = event
                .response
                .ok_or_else(|| AgentError::Model("response event missing response".to_string()))?;
            merge_response(acc, response)?;
        }
        "response.failed" => {
            return Err(AgentError::Model("responses request failed".to_string()));
        }
        "response.refusal.delta" => {
            return Err(AgentError::Model("model returned a refusal".to_string()));
        }
        _ => {}
    }
    Ok(())
}

fn merge_item(acc: &mut Accumulator, index: usize, item: Value) -> Result<(), AgentError> {
    match item.get("type").and_then(Value::as_str) {
        Some("function_call") => {
            let builder = acc.function_call(index)?;
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
            let raw = serde_json::json!({ "output": [item] });
            model_response_shape(&raw, false)?;
            acc.message()?;
        }
        Some("reasoning") => {}
        Some(other) => {
            return Err(AgentError::Model(format!(
                "unsupported responses output item: {other}"
            )));
        }
        None => return Err(AgentError::Model("output item missing type".to_string())),
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
    let decoded = model_response_shape(&response, !acc.streamed_text)?;
    if let Some(reason) = response_stop(&response, &decoded) {
        acc.stop_reason = Some(reason);
    }
    match decoded.outcome {
        Some(ModelOutcome::FinalAnswer { text }) => {
            acc.message()?;
            if !acc.streamed_text {
                acc.completed_text.push_str(&text);
            }
        }
        Some(ModelOutcome::ToolCalls { calls }) => {
            for call in calls {
                upsert_tool(acc, call)?;
            }
        }
        None => {}
    }
    Ok(())
}

fn upsert_tool(acc: &mut Accumulator, tool: ToolUse) -> Result<(), AgentError> {
    let builders = acc.function_calls()?;
    if let Some((_, builder)) = builders
        .iter_mut()
        .find(|(_, builder)| builder.call_id.as_deref() == Some(tool.id.as_str()))
    {
        builder.arguments = arguments_from_tool(&tool);
        builder.call_id = Some(tool.id);
        builder.name = Some(tool.name);
        builder.raw = tool.raw;
        return Ok(());
    }
    let index = builders.keys().next_back().copied().unwrap_or(0) + 1;
    builders.insert(
        index,
        ToolCallBuilder {
            call_id: Some(tool.id.clone()),
            name: Some(tool.name.clone()),
            arguments: arguments_from_tool(&tool),
            raw: tool.raw,
        },
    );
    Ok(())
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

impl Accumulator {
    fn message(&mut self) -> Result<(), AgentError> {
        match &self.output {
            OutputKind::Undecided | OutputKind::Message => {
                self.output = OutputKind::Message;
                Ok(())
            }
            OutputKind::FunctionCalls(_) => Err(AgentError::Model(
                "responses stream mixed messages and function calls".to_string(),
            )),
        }
    }

    fn function_call(&mut self, index: usize) -> Result<&mut ToolCallBuilder, AgentError> {
        Ok(self.function_calls()?.entry(index).or_default())
    }

    fn function_calls(&mut self) -> Result<&mut BTreeMap<usize, ToolCallBuilder>, AgentError> {
        if matches!(self.output, OutputKind::Undecided) {
            self.output = OutputKind::FunctionCalls(BTreeMap::new());
        }
        if let OutputKind::FunctionCalls(builders) = &mut self.output {
            Ok(builders)
        } else {
            Err(AgentError::Model(
                "responses stream mixed messages and function calls".to_string(),
            ))
        }
    }
}

fn outcome(acc: Accumulator) -> Result<Option<ModelOutcome>, AgentError> {
    match acc.output {
        OutputKind::Undecided => Ok(None),
        OutputKind::Message => Ok(Some(ModelOutcome::FinalAnswer {
            text: acc.completed_text,
        })),
        OutputKind::FunctionCalls(builders) => builders
            .into_values()
            .map(tool_use_from_builder)
            .collect::<Result<Vec<_>, _>>()
            .map(|calls| Some(ModelOutcome::ToolCalls { calls })),
    }
}

fn response_stop(
    response: &Value,
    decoded: &crate::responses::DecodedOutput,
) -> Option<StopReason> {
    match response.get("status").and_then(Value::as_str) {
        Some("incomplete") => Some(incomplete_reason(response)),
        Some("completed") if decoded.message_seen => Some(StopReason::EndTurn),
        Some("completed") if decoded.tool_seen => Some(StopReason::ToolUse),
        Some(other) => Some(StopReason::Other(other.to_string())),
        None if decoded.message_seen => Some(StopReason::EndTurn),
        None if decoded.tool_seen => Some(StopReason::ToolUse),
        None => None,
    }
}

fn inferred_stop(acc: &Accumulator) -> Option<StopReason> {
    match acc.output {
        OutputKind::FunctionCalls(_) => Some(StopReason::ToolUse),
        OutputKind::Message => Some(StopReason::EndTurn),
        OutputKind::Undecided => acc
            .status
            .as_ref()
            .map(|status| StopReason::Other(status.clone())),
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
        None => StopReason::Other("incomplete".to_string()),
    }
}
