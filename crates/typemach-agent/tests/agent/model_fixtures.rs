use typemach_agent::{
    AgentError, AssistantMessageId, AssistantMessageItem, AssistantMessagePhase, AssistantTextPart,
    ModelResponse, ModelStream, ModelStreamEvent, ResponseContentIndex, ResponseOutputIndex,
    ToolUse,
};

fn response_message(phase: AssistantMessagePhase, text: impl Into<String>) -> AssistantMessageItem {
    message_item("message-0", 0, phase, text)
}

pub(super) fn message_item(
    id: &str,
    output_index: usize,
    phase: AssistantMessagePhase,
    text: impl Into<String>,
) -> AssistantMessageItem {
    AssistantMessageItem {
        id: AssistantMessageId::new(id),
        output_index: ResponseOutputIndex::new(output_index),
        phase,
        content: vec![AssistantTextPart {
            index: ResponseContentIndex::new(0),
            text: text.into(),
        }],
    }
}

pub(super) fn emit_message(
    stream: &ModelStream,
    message: &AssistantMessageItem,
    deltas: &[&str],
) -> Result<(), AgentError> {
    stream.emit(ModelStreamEvent::AssistantMessageStarted {
        message_id: message.id.clone(),
        output_index: message.output_index,
        phase: message.phase,
    })?;
    for (index, delta) in deltas.iter().enumerate() {
        stream.emit(ModelStreamEvent::AssistantMessageDelta {
            message_id: message.id.clone(),
            output_index: message.output_index,
            content_index: ResponseContentIndex::new(0),
            delta: (*delta).to_string(),
            index,
        })?;
    }
    stream.emit(ModelStreamEvent::AssistantMessageDone {
        message: message.clone(),
    })
}

pub(super) fn emit_pending(stream: &ModelStream, id: &str, text: &str) -> Result<(), AgentError> {
    let message = message_item(id, 0, AssistantMessagePhase::FinalAnswer, text);
    stream.emit(ModelStreamEvent::AssistantMessageStarted {
        message_id: message.id.clone(),
        output_index: message.output_index,
        phase: message.phase,
    })?;
    stream.emit(ModelStreamEvent::AssistantMessageDelta {
        message_id: message.id,
        output_index: message.output_index,
        content_index: ResponseContentIndex::new(0),
        delta: text.to_string(),
        index: 0,
    })
}

pub(super) fn final_response(text: impl Into<String>) -> ModelResponse {
    ModelResponse {
        assistant_messages: vec![response_message(AssistantMessagePhase::FinalAnswer, text)],
        ..ModelResponse::default()
    }
}

pub(super) fn tool_response(text: impl Into<String>, calls: Vec<ToolUse>) -> ModelResponse {
    let text = text.into();
    ModelResponse {
        assistant_messages: (!text.is_empty())
            .then(|| response_message(AssistantMessagePhase::Commentary, text))
            .into_iter()
            .collect(),
        tool_calls: calls,
        ..ModelResponse::default()
    }
}
