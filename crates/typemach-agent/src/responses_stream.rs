use std::collections::BTreeMap;

use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::Value;

use crate::responses::{
    DecodeFailure, assistant_message_from_item, assistant_message_phase, model_response_from_value,
    model_response_shape, tool_use_from_item,
};
use crate::{
    AgentError, AssistantMessageId, AssistantMessageItem, AssistantMessagePhase, AssistantTextPart,
    ModelResponse, ModelStream, ModelStreamEvent, ResponseContentIndex, ResponseOutputIndex,
    StopReason,
};

#[derive(Debug, Deserialize)]
struct Event {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    item_id: Option<String>,
    #[serde(default)]
    output_index: Option<usize>,
    #[serde(default)]
    content_index: Option<usize>,
    #[serde(default)]
    item: Option<Value>,
    #[serde(default)]
    part: Option<Value>,
    #[serde(default)]
    response: Option<Value>,
}

#[derive(Debug)]
struct MessageBuilder {
    id: AssistantMessageId,
    output_index: ResponseOutputIndex,
    phase: AssistantMessagePhase,
    content: BTreeMap<ResponseContentIndex, String>,
    next_delta_index: usize,
}

#[derive(Debug)]
struct ReasoningItem {
    id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalKind {
    Completed,
    Incomplete,
}

#[derive(Debug, Default)]
struct Accumulator {
    active_messages: BTreeMap<ResponseOutputIndex, MessageBuilder>,
    active_reasoning: BTreeMap<ResponseOutputIndex, ReasoningItem>,
    completed_messages: Vec<AssistantMessageItem>,
    response: Option<ModelResponse>,
    terminal: Option<TerminalKind>,
}

pub(crate) async fn decode_stream(
    response: reqwest::Response,
    stream: ModelStream,
) -> Result<ModelResponse, DecodeFailure> {
    let mut bytes = response.bytes_stream();
    let mut pending = Vec::new();
    let mut acc = Accumulator::default();
    while let Some(chunk) = bytes.next().await {
        let chunk = chunk.map_err(|err| DecodeFailure::body(err, "model stream failed"))?;
        pending.extend_from_slice(&chunk);
        while let Some(index) = pending.iter().position(|byte| *byte == b'\n') {
            let line = pending.drain(..=index).collect::<Vec<_>>();
            handle_stream_line(decode_line(line)?, &stream, &mut acc)?;
        }
    }
    if pending.iter().any(|byte| !byte.is_ascii_whitespace()) {
        handle_stream_line(decode_line(pending)?, &stream, &mut acc)?;
    }
    acc.finish()
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
) -> Result<(), DecodeFailure> {
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(());
    };
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(());
    }
    let event: Event = serde_json::from_str(data).map_err(|err| {
        DecodeFailure::protocol_message(format!("responses stream event was invalid: {err}"))
    })?;
    match event.kind.as_str() {
        "response.output_item.added" => add_item(event, stream, acc)?,
        "response.content_part.added" => add_content(event, stream, acc)?,
        "response.output_text.delta" => append_text(event, stream, acc)?,
        "response.output_text.done" => finish_text(event, acc)?,
        "response.content_part.done" => finish_content(event, acc)?,
        "response.output_item.done" => finish_item(event, stream, acc)?,
        "response.reasoning_text.delta" | "response.reasoning_text.done" => {
            active_reasoning(&event, acc)?;
        }
        "response.function_call_arguments.delta" => {}
        "response.completed" => finish_response(event, TerminalKind::Completed, acc)?,
        "response.incomplete" => finish_response(event, TerminalKind::Incomplete, acc)?,
        "response.failed" => {
            return Err(DecodeFailure::protocol_message(
                "responses request failed".to_string(),
            ));
        }
        "response.refusal.delta" => {
            return Err(DecodeFailure::protocol_message(
                "model returned a refusal".to_string(),
            ));
        }
        _ => {}
    }
    Ok(())
}

fn add_item(event: Event, stream: &ModelStream, acc: &mut Accumulator) -> Result<(), AgentError> {
    let output_index = event_output_index(&event)?;
    let item = event
        .item
        .as_ref()
        .ok_or_else(|| AgentError::Model("output item event missing item".to_string()))?;
    match item.get("type").and_then(Value::as_str) {
        Some("message") => {
            let id = message_id(item)?;
            let phase = assistant_message_phase(item)?;
            if acc.active_messages.contains_key(&output_index)
                || acc.active_reasoning.contains_key(&output_index)
                || acc
                    .completed_messages
                    .iter()
                    .any(|message| message.output_index == output_index)
            {
                return Err(AgentError::Model(format!(
                    "duplicate message output index {}",
                    output_index.get()
                )));
            }
            if acc.active_messages.values().any(|message| message.id == id)
                || acc
                    .completed_messages
                    .iter()
                    .any(|message| message.id == id)
            {
                return Err(AgentError::Model("duplicate message item id".to_string()));
            }
            acc.active_messages.insert(
                output_index,
                MessageBuilder {
                    id: id.clone(),
                    output_index,
                    phase,
                    content: BTreeMap::new(),
                    next_delta_index: 0,
                },
            );
            stream.emit(ModelStreamEvent::AssistantMessageStarted {
                message_id: id,
                output_index,
                phase,
            })?;
        }
        Some("function_call") => validate_function_item(item, false)?,
        Some("reasoning") => add_reasoning(item, output_index, acc)?,
        Some(other) => {
            return Err(AgentError::Model(format!(
                "unsupported responses output item: {other}"
            )));
        }
        None => return Err(AgentError::Model("output item missing type".to_string())),
    }
    Ok(())
}

fn add_content(
    event: Event,
    stream: &ModelStream,
    acc: &mut Accumulator,
) -> Result<(), AgentError> {
    let part = event
        .part
        .as_ref()
        .ok_or_else(|| AgentError::Model("content part event missing part".to_string()))?;
    match part.get("type").and_then(Value::as_str) {
        Some("output_text") => add_message_content(event, stream, acc),
        Some("reasoning_text") => active_reasoning(&event, acc),
        Some(other) => Err(AgentError::Model(format!(
            "unsupported output content part: {other}"
        ))),
        None => Err(AgentError::Model(
            "output content part missing type".to_string(),
        )),
    }
}

fn add_message_content(
    event: Event,
    stream: &ModelStream,
    acc: &mut Accumulator,
) -> Result<(), AgentError> {
    let (message, content_index) = active_content(&event, acc)?;
    let part = event
        .part
        .as_ref()
        .expect("content part was checked before message dispatch");
    let initial = part.get("text").and_then(Value::as_str).unwrap_or_default();
    if message
        .content
        .insert(content_index, String::new())
        .is_some()
    {
        return Err(AgentError::Model(format!(
            "duplicate message content index {}",
            content_index.get()
        )));
    }
    if !initial.is_empty() {
        let index = message.next_delta_index;
        message.next_delta_index += 1;
        message
            .content
            .get_mut(&content_index)
            .expect("inserted content part")
            .push_str(initial);
        stream.emit(ModelStreamEvent::AssistantMessageDelta {
            message_id: message.id.clone(),
            output_index: message.output_index,
            content_index,
            delta: initial.to_string(),
            index,
        })?;
    }
    Ok(())
}

fn append_text(
    event: Event,
    stream: &ModelStream,
    acc: &mut Accumulator,
) -> Result<(), AgentError> {
    let delta = event
        .delta
        .clone()
        .ok_or_else(|| AgentError::Model("output_text delta missing text".to_string()))?;
    let (message, content_index) = active_content(&event, acc)?;
    let part = message.content.get_mut(&content_index).ok_or_else(|| {
        AgentError::Model(format!(
            "output_text delta preceded content part {}",
            content_index.get()
        ))
    })?;
    part.push_str(&delta);
    let index = message.next_delta_index;
    message.next_delta_index += 1;
    stream.emit(ModelStreamEvent::AssistantMessageDelta {
        message_id: message.id.clone(),
        output_index: message.output_index,
        content_index,
        delta,
        index,
    })?;
    Ok(())
}

fn finish_text(event: Event, acc: &mut Accumulator) -> Result<(), AgentError> {
    let completed = event
        .text
        .clone()
        .ok_or_else(|| AgentError::Model("output_text done missing text".to_string()))?;
    let (message, content_index) = active_content(&event, acc)?;
    verify_part(message, content_index, &completed)
}

fn finish_content(event: Event, acc: &mut Accumulator) -> Result<(), AgentError> {
    let part = event
        .part
        .as_ref()
        .ok_or_else(|| AgentError::Model("content part event missing part".to_string()))?;
    match part.get("type").and_then(Value::as_str) {
        Some("output_text") => finish_message_content(event, acc),
        Some("reasoning_text") => active_reasoning(&event, acc),
        Some(other) => Err(AgentError::Model(format!(
            "unsupported output content part: {other}"
        ))),
        None => Err(AgentError::Model(
            "output content part missing type".to_string(),
        )),
    }
}

fn finish_message_content(event: Event, acc: &mut Accumulator) -> Result<(), AgentError> {
    let part = event
        .part
        .as_ref()
        .expect("content part was checked before message dispatch");
    let completed = part
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentError::Model("output_text content part missing text".to_string()))?;
    let (message, content_index) = active_content(&event, acc)?;
    verify_part(message, content_index, completed)
}

fn finish_item(
    event: Event,
    stream: &ModelStream,
    acc: &mut Accumulator,
) -> Result<(), AgentError> {
    let output_index = event_output_index(&event)?;
    let item = event
        .item
        .as_ref()
        .ok_or_else(|| AgentError::Model("output item event missing item".to_string()))?;
    match item.get("type").and_then(Value::as_str) {
        Some("message") => {
            if item.get("status").and_then(Value::as_str) != Some("completed") {
                return Ok(());
            }
            let message = assistant_message_from_item(item, output_index)?;
            let streamed = acc.active_messages.remove(&output_index).ok_or_else(|| {
                AgentError::Model(format!(
                    "message output {} completed before it started",
                    output_index.get()
                ))
            })?;
            verify_message(&streamed, &message)?;
            stream.emit(ModelStreamEvent::AssistantMessageDone {
                message: message.clone(),
            })?;
            acc.completed_messages.push(message);
        }
        Some("function_call") => validate_function_item(item, true)?,
        Some("reasoning") => finish_reasoning(item, output_index, acc)?,
        Some(other) => {
            return Err(AgentError::Model(format!(
                "unsupported responses output item: {other}"
            )));
        }
        None => return Err(AgentError::Model("output item missing type".to_string())),
    }
    Ok(())
}

fn finish_response(
    event: Event,
    terminal: TerminalKind,
    acc: &mut Accumulator,
) -> Result<(), AgentError> {
    if acc.terminal.is_some() {
        return Err(AgentError::Model(
            "responses stream returned multiple terminal events".to_string(),
        ));
    }
    let raw = event
        .response
        .ok_or_else(|| AgentError::Model("response event missing response".to_string()))?;
    acc.response = Some(model_response_from_value(raw)?);
    acc.terminal = Some(terminal);
    Ok(())
}

fn active_content<'a>(
    event: &Event,
    acc: &'a mut Accumulator,
) -> Result<(&'a mut MessageBuilder, ResponseContentIndex), AgentError> {
    let output_index = event_output_index(event)?;
    let content_index = ResponseContentIndex::new(event.content_index.ok_or_else(|| {
        AgentError::Model("message content event missing content_index".to_string())
    })?);
    let item_id = event
        .item_id
        .as_deref()
        .ok_or_else(|| AgentError::Model("message content event missing item_id".to_string()))?;
    let message = acc.active_messages.get_mut(&output_index).ok_or_else(|| {
        AgentError::Model(format!(
            "message output {} was not active",
            output_index.get()
        ))
    })?;
    if message.id.as_str() != item_id {
        return Err(AgentError::Model(format!(
            "message item id {} did not match active item {}",
            item_id, message.id
        )));
    }
    Ok((message, content_index))
}

fn add_reasoning(
    item: &Value,
    output_index: ResponseOutputIndex,
    acc: &mut Accumulator,
) -> Result<(), AgentError> {
    validate_shape(item)?;
    let id = reasoning_id(item)?;
    if acc.active_messages.contains_key(&output_index)
        || acc.active_reasoning.contains_key(&output_index)
    {
        return Err(AgentError::Model(format!(
            "duplicate output index {}",
            output_index.get()
        )));
    }
    acc.active_reasoning
        .insert(output_index, ReasoningItem { id });
    Ok(())
}

fn finish_reasoning(
    item: &Value,
    output_index: ResponseOutputIndex,
    acc: &mut Accumulator,
) -> Result<(), AgentError> {
    validate_shape(item)?;
    let id = reasoning_id(item)?;
    let active = acc.active_reasoning.remove(&output_index).ok_or_else(|| {
        AgentError::Model(format!(
            "reasoning output {} completed before it started",
            output_index.get()
        ))
    })?;
    if active.id != id {
        return Err(AgentError::Model(
            "reasoning output identity differed from active item".to_string(),
        ));
    }
    Ok(())
}

fn active_reasoning(event: &Event, acc: &Accumulator) -> Result<(), AgentError> {
    let output_index = event_output_index(event)?;
    let item_id = event
        .item_id
        .as_deref()
        .ok_or_else(|| AgentError::Model("reasoning content event missing item_id".to_string()))?;
    if event.content_index.is_none() {
        return Err(AgentError::Model(
            "reasoning content event missing content_index".to_string(),
        ));
    }
    let reasoning = acc.active_reasoning.get(&output_index).ok_or_else(|| {
        AgentError::Model(format!(
            "reasoning output {} was not active",
            output_index.get()
        ))
    })?;
    if reasoning.id != item_id {
        return Err(AgentError::Model(format!(
            "reasoning item id {} did not match active item {}",
            item_id, reasoning.id
        )));
    }
    Ok(())
}

fn event_output_index(event: &Event) -> Result<ResponseOutputIndex, AgentError> {
    event
        .output_index
        .map(ResponseOutputIndex::new)
        .ok_or_else(|| AgentError::Model("output item event missing output_index".to_string()))
}

fn message_id(item: &Value) -> Result<AssistantMessageId, AgentError> {
    if item.get("role").and_then(Value::as_str) != Some("assistant") {
        return Err(AgentError::Model(
            "message output role must be assistant".to_string(),
        ));
    }
    item.get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(AssistantMessageId::new)
        .ok_or_else(|| AgentError::Model("message output missing id".to_string()))
}

fn reasoning_id(item: &Value) -> Result<String, AgentError> {
    item.get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(ToString::to_string)
        .ok_or_else(|| AgentError::Model("reasoning output missing id".to_string()))
}

fn verify_part(
    message: &MessageBuilder,
    content_index: ResponseContentIndex,
    completed: &str,
) -> Result<(), AgentError> {
    let streamed = message.content.get(&content_index).ok_or_else(|| {
        AgentError::Model(format!(
            "message content {} completed before it started",
            content_index.get()
        ))
    })?;
    if streamed != completed {
        return Err(AgentError::Model(format!(
            "streamed message content {} differed from completed content",
            content_index.get()
        )));
    }
    Ok(())
}

fn verify_message(
    streamed: &MessageBuilder,
    completed: &AssistantMessageItem,
) -> Result<(), AgentError> {
    if streamed.id != completed.id
        || streamed.output_index != completed.output_index
        || streamed.phase != completed.phase
    {
        return Err(AgentError::Model(
            "completed message identity differed from active item".to_string(),
        ));
    }
    let content = streamed
        .content
        .iter()
        .map(|(index, text)| AssistantTextPart {
            index: *index,
            text: text.clone(),
        })
        .collect::<Vec<_>>();
    if content != completed.content {
        return Err(AgentError::Model(
            "streamed message bytes differed from completed item".to_string(),
        ));
    }
    Ok(())
}

fn validate_shape(item: &Value) -> Result<(), AgentError> {
    model_response_shape(&serde_json::json!({ "output": [item] })).map(|_| ())
}

fn validate_function_item(item: &Value, completed: bool) -> Result<(), AgentError> {
    if completed {
        tool_use_from_item(item).map(|_| ())
    } else {
        for field in ["call_id", "name", "arguments"] {
            if item.get(field).and_then(Value::as_str).is_none() {
                return Err(AgentError::Model(format!("function_call missing {field}")));
            }
        }
        Ok(())
    }
}

impl Accumulator {
    fn finish(mut self) -> Result<ModelResponse, DecodeFailure> {
        let terminal = self.terminal.ok_or_else(|| {
            DecodeFailure::protocol_message(
                "responses stream ended without a terminal event".to_string(),
            )
        })?;
        let response = self.response.take().ok_or_else(|| {
            DecodeFailure::protocol_message("responses terminal event had no response".to_string())
        })?;
        if terminal == TerminalKind::Completed {
            if !self.active_messages.is_empty() || !self.active_reasoning.is_empty() {
                return Err(DecodeFailure::protocol_message(
                    "responses completed with an active output item".to_string(),
                ));
            }
            self.completed_messages
                .sort_by_key(|message| message.output_index);
            if self.completed_messages != response.assistant_messages {
                return Err(DecodeFailure::protocol_message(
                    "completed message items differed from response snapshot".to_string(),
                ));
            }
        } else if !matches!(
            response.stop_reason,
            Some(StopReason::MaxTokens | StopReason::Refusal)
        ) {
            return Err(DecodeFailure::protocol_message(
                "incomplete response did not report an incomplete stop reason".to_string(),
            ));
        }
        Ok(response)
    }
}
