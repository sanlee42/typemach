use std::collections::BTreeMap;

use tokio::sync::mpsc;
use typemach::MachineError;

use crate::{
    AgentError, AgentMessage, AgentModel, AgentRunContext, AgentSignal, AgentState, AgentToolSpec,
    AssistantMessageId, AssistantMessageItem, AssistantMessagePhase, AssistantTextPart,
    ContentBlock, ModelRequest, ModelResponse, ModelStream, ModelStreamEvent, ResponseContentIndex,
    ResponseOutputIndex, StopReason, ToolChoice, ToolUse, context,
};

pub(super) struct Turn {
    pub(super) outcome: Option<TurnOutcome>,
    pub(super) stop_reason: Option<StopReason>,
}

pub(super) enum TurnOutcome {
    Message {
        content: Vec<ContentBlock>,
        text: String,
    },
    ToolCalls {
        content: Vec<ContentBlock>,
        calls: Vec<ToolUse>,
    },
}

#[derive(Default)]
enum TurnStream {
    #[default]
    CompletedOnly,
    Observed(ObservedLifecycle),
}

#[derive(Default)]
struct ObservedLifecycle {
    active: BTreeMap<ResponseOutputIndex, ActiveMessage>,
    completed: Vec<AssistantMessageItem>,
}

struct ActiveMessage {
    id: AssistantMessageId,
    output_index: ResponseOutputIndex,
    content: BTreeMap<ResponseContentIndex, String>,
    next_index: usize,
}

pub(super) async fn prepare(
    state: &mut AgentState,
    ctx: &AgentRunContext,
    messages: Vec<AgentMessage>,
    tools: Vec<AgentToolSpec>,
    system_suffix: Option<String>,
    tool_choice: Option<ToolChoice>,
    turn: u32,
) -> Result<ModelRequest, MachineError> {
    let prompt_window = context::prompt_window(&messages, &state.context_policy)
        .map_err(MachineError::transition)?;
    if let Some(digest) = prompt_window.digest.clone()
        && state.digest.as_ref() != Some(&digest)
    {
        state.digest = Some(digest.clone());
        ctx.emit(AgentSignal::DigestUpdated { digest }).await?;
    }
    if let Some(compaction) = prompt_window.compaction.clone() {
        ctx.emit(AgentSignal::ContextCompacted { compaction })
            .await?;
    }
    Ok(ModelRequest {
        messages: prompt_window.messages,
        tools,
        context: state.context.clone(),
        turn,
        system_suffix,
        tool_choice,
    })
}

pub(super) async fn invoke<M: AgentModel + ?Sized>(
    model: &M,
    state: &mut AgentState,
    ctx: &AgentRunContext,
    request: ModelRequest,
) -> Result<Turn, MachineError> {
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let response = model.next_step(request, ModelStream::new(event_tx));
    tokio::pin!(response);
    let mut streamed = TurnStream::default();
    let mut response = loop {
        tokio::select! {
            maybe_event = event_rx.recv() => {
                if let Some(event) = maybe_event {
                    streamed.apply(ctx, event).await?;
                }
            }
            response = &mut response => break response.map_err(MachineError::transition)?,
        }
    };
    while let Ok(event) = event_rx.try_recv() {
        streamed.apply(ctx, event).await?;
    }
    if let Some(usage) = response.usage.take() {
        state.usage.input_tokens += usage.input_tokens;
        state.usage.output_tokens += usage.output_tokens;
        ctx.emit(AgentSignal::Usage { usage }).await?;
    }
    finish_response(ctx, streamed, response).await
}

async fn finish_response(
    ctx: &AgentRunContext,
    mut streamed: TurnStream,
    mut response: ModelResponse,
) -> Result<Turn, MachineError> {
    response
        .assistant_messages
        .sort_by_key(|message| message.output_index);
    if aborted(response.stop_reason.as_ref()) {
        return Ok(Turn {
            outcome: None,
            stop_reason: response.stop_reason,
        });
    }
    streamed.complete(ctx, &response.assistant_messages).await?;

    validate_phase_order(
        &response.assistant_messages,
        !response.tool_calls.is_empty(),
    )?;
    let mut content = reasoning_blocks(response.reasoning);
    content.extend(
        response
            .assistant_messages
            .iter()
            .cloned()
            .map(ContentBlock::AssistantMessage),
    );
    let outcome = if response.tool_calls.is_empty() {
        let text = response
            .assistant_messages
            .iter()
            .filter(|message| message.phase == AssistantMessagePhase::FinalAnswer)
            .map(AssistantMessageItem::text)
            .collect::<String>();
        (!response.assistant_messages.is_empty()).then_some(TurnOutcome::Message { content, text })
    } else {
        content.extend(
            response
                .tool_calls
                .iter()
                .cloned()
                .map(ContentBlock::ToolUse),
        );
        Some(TurnOutcome::ToolCalls {
            content,
            calls: response.tool_calls,
        })
    };
    Ok(Turn {
        outcome,
        stop_reason: response.stop_reason,
    })
}

fn validate_phase_order(
    messages: &[AssistantMessageItem],
    has_tool_calls: bool,
) -> Result<(), MachineError> {
    if has_tool_calls {
        if messages
            .iter()
            .any(|message| message.phase != AssistantMessagePhase::Commentary)
        {
            return Err(AgentError::Model(
                "tool response contained a final-answer message".to_string(),
            )
            .machine());
        }
        return Ok(());
    }
    let mut final_seen = false;
    for message in messages {
        match message.phase {
            AssistantMessagePhase::Commentary if final_seen => {
                return Err(AgentError::Model(
                    "commentary message followed a final-answer message".to_string(),
                )
                .machine());
            }
            AssistantMessagePhase::Commentary => {}
            AssistantMessagePhase::FinalAnswer => final_seen = true,
        }
    }
    if !messages.is_empty() && !final_seen {
        return Err(AgentError::Model(
            "terminal response contained no final-answer message".to_string(),
        )
        .machine());
    }
    Ok(())
}

fn reasoning_blocks(reasoning: Vec<String>) -> Vec<ContentBlock> {
    reasoning
        .into_iter()
        .filter(|text| !text.is_empty())
        .map(|text| ContentBlock::Thinking {
            text,
            signature: None,
        })
        .collect()
}

fn aborted(reason: Option<&StopReason>) -> bool {
    matches!(
        reason,
        Some(StopReason::MaxTokens | StopReason::Refusal | StopReason::Other(_))
    )
}

impl TurnStream {
    async fn apply(
        &mut self,
        ctx: &AgentRunContext,
        event: ModelStreamEvent,
    ) -> Result<(), MachineError> {
        let lifecycle = match self {
            Self::CompletedOnly => {
                *self = Self::Observed(ObservedLifecycle::default());
                let Self::Observed(lifecycle) = self else {
                    unreachable!();
                };
                lifecycle
            }
            Self::Observed(lifecycle) => lifecycle,
        };
        match event {
            ModelStreamEvent::AssistantMessageStarted {
                message_id,
                output_index,
            } => lifecycle.start(ctx, message_id, output_index).await,
            ModelStreamEvent::AssistantMessageDelta {
                message_id,
                output_index,
                content_index,
                delta,
                index,
            } => {
                lifecycle
                    .append(ctx, message_id, output_index, content_index, delta, index)
                    .await
            }
            ModelStreamEvent::AssistantMessageDone { message } => {
                lifecycle.done(ctx, message).await
            }
        }
    }

    async fn complete(
        &mut self,
        ctx: &AgentRunContext,
        messages: &[AssistantMessageItem],
    ) -> Result<(), MachineError> {
        match self {
            Self::Observed(lifecycle) => lifecycle.verify_complete(messages),
            Self::CompletedOnly => {
                let mut lifecycle = ObservedLifecycle::default();
                lifecycle.publish_completed(ctx, messages).await?;
                *self = Self::Observed(lifecycle);
                Ok(())
            }
        }
    }
}

impl ObservedLifecycle {
    async fn start(
        &mut self,
        ctx: &AgentRunContext,
        id: AssistantMessageId,
        output_index: ResponseOutputIndex,
    ) -> Result<(), MachineError> {
        if self.active.contains_key(&output_index) {
            return Err(AgentError::Model(
                "assistant message output index was reused before completion".to_string(),
            )
            .machine());
        }
        if self.active.values().any(|message| message.id == id)
            || self.completed.iter().any(|message| message.id == id)
        {
            return Err(AgentError::Model("assistant message id was reused".to_string()).machine());
        }
        self.active.insert(
            output_index,
            ActiveMessage {
                id: id.clone(),
                output_index,
                content: BTreeMap::new(),
                next_index: 0,
            },
        );
        ctx.emit(AgentSignal::AssistantMessageStarted { message_id: id })
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn append(
        &mut self,
        ctx: &AgentRunContext,
        id: AssistantMessageId,
        output_index: ResponseOutputIndex,
        content_index: ResponseContentIndex,
        delta: String,
        index: usize,
    ) -> Result<(), MachineError> {
        let active = self.active.get_mut(&output_index).ok_or_else(|| {
            MachineError::transition(AgentError::Model(
                "assistant message delta arrived before item start".to_string(),
            ))
        })?;
        if active.id != id || active.output_index != output_index {
            return Err(AgentError::Model(
                "assistant message delta identity differed from active item".to_string(),
            )
            .machine());
        }
        if active.next_index != index {
            return Err(AgentError::Model(format!(
                "assistant message delta index {index} did not match expected {}",
                active.next_index
            ))
            .machine());
        }
        active.next_index += 1;
        active
            .content
            .entry(content_index)
            .or_default()
            .push_str(&delta);
        ctx.emit(AgentSignal::AssistantMessageDelta {
            message_id: id,
            delta,
            index,
        })
        .await
    }

    async fn done(
        &mut self,
        ctx: &AgentRunContext,
        message: AssistantMessageItem,
    ) -> Result<(), MachineError> {
        let active = self.active.remove(&message.output_index).ok_or_else(|| {
            MachineError::transition(AgentError::Model(
                "assistant message completed before item start".to_string(),
            ))
        })?;
        let streamed = active
            .content
            .into_iter()
            .map(|(index, text)| AssistantTextPart { index, text })
            .collect::<Vec<_>>();
        if active.id != message.id
            || active.output_index != message.output_index
            || streamed != message.content
        {
            return Err(AgentError::Model(
                "assistant message completion differed from streamed item".to_string(),
            )
            .machine());
        }
        ctx.emit(AgentSignal::AssistantMessageDone {
            message_id: message.id.clone(),
            phase: message.phase,
        })
        .await?;
        self.completed.push(message);
        Ok(())
    }

    fn verify_complete(&mut self, completed: &[AssistantMessageItem]) -> Result<(), MachineError> {
        if !self.active.is_empty() {
            return Err(AgentError::Model(
                "model response completed with an active assistant message".to_string(),
            )
            .machine());
        }
        self.completed.sort_by_key(|message| message.output_index);
        if self.completed != completed {
            return Err(AgentError::Model(
                "streamed assistant messages differed from model response".to_string(),
            )
            .machine());
        }
        Ok(())
    }

    async fn publish_completed(
        &mut self,
        ctx: &AgentRunContext,
        messages: &[AssistantMessageItem],
    ) -> Result<(), MachineError> {
        for message in messages {
            self.start(ctx, message.id.clone(), message.output_index)
                .await?;
            for (index, part) in message.content.iter().enumerate() {
                self.append(
                    ctx,
                    message.id.clone(),
                    message.output_index,
                    part.index,
                    part.text.clone(),
                    index,
                )
                .await?;
            }
            self.done(ctx, message.clone()).await?;
        }
        Ok(())
    }
}
