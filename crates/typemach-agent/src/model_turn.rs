use tokio::sync::mpsc;
use typemach::MachineError;

use crate::{
    AgentMessage, AgentModel, AgentRunContext, AgentSignal, AgentState, AgentToolSpec,
    AssistantMessageId, AssistantMessagePhase, ContentBlock, ModelOutcome, ModelRequest,
    ModelStream, OutputTextDelta, StopReason, ToolChoice, ToolUse, context,
};

pub(super) struct Turn {
    pub(super) outcome: Option<TurnOutcome>,
    pub(super) stop_reason: Option<StopReason>,
}

pub(super) enum TurnOutcome {
    FinalAnswer {
        content: Vec<ContentBlock>,
        text: String,
    },
    ToolCalls {
        content: Vec<ContentBlock>,
        calls: Vec<ToolUse>,
    },
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
    let (delta_tx, mut delta_rx) = mpsc::unbounded_channel();
    let message_id = assistant_message_id(ctx, request.turn);
    let response = model.next_step(request, ModelStream::new(delta_tx));
    tokio::pin!(response);
    let mut content = Vec::new();
    let mut message_text = String::new();
    let mut next_index = 0;
    let response = loop {
        tokio::select! {
            maybe_delta = delta_rx.recv() => {
                if let Some(delta) = maybe_delta {
                    append_delta(
                        ctx,
                        &message_id,
                        &mut content,
                        &mut message_text,
                        &mut next_index,
                        delta,
                    )
                    .await?;
                }
            }
            response = &mut response => break response.map_err(MachineError::transition)?,
        }
    };
    while let Ok(delta) = delta_rx.try_recv() {
        append_delta(
            ctx,
            &message_id,
            &mut content,
            &mut message_text,
            &mut next_index,
            delta,
        )
        .await?;
    }
    if let Some(usage) = response.usage {
        state.usage.input_tokens += usage.input_tokens;
        state.usage.output_tokens += usage.output_tokens;
        ctx.emit(AgentSignal::Usage { usage }).await?;
    }
    let reasoning = reasoning_blocks(response.reasoning);
    let outcome = match response.outcome {
        Some(ModelOutcome::FinalAnswer { text: completed }) => {
            let mut final_content = reasoning;
            final_content.extend(content);
            if terminal_text_allowed(response.stop_reason.as_ref()) {
                append_completed_text(
                    ctx,
                    &message_id,
                    &mut final_content,
                    &mut message_text,
                    &mut next_index,
                    completed,
                )
                .await?;
                ctx.emit(AgentSignal::AssistantMessageDone {
                    message_id,
                    phase: AssistantMessagePhase::FinalAnswer,
                })
                .await?;
            }
            Some(TurnOutcome::FinalAnswer {
                content: final_content,
                text: message_text,
            })
        }
        Some(ModelOutcome::ToolCalls {
            text: completed,
            calls,
        }) => {
            let mut call_content = reasoning;
            call_content.extend(content);
            if completed_tool_calls(response.stop_reason.as_ref()) {
                append_completed_text(
                    ctx,
                    &message_id,
                    &mut call_content,
                    &mut message_text,
                    &mut next_index,
                    completed,
                )
                .await?;
                ctx.emit(AgentSignal::AssistantMessageDone {
                    message_id,
                    phase: AssistantMessagePhase::Commentary,
                })
                .await?;
            }
            call_content.extend(calls.iter().cloned().map(ContentBlock::ToolUse));
            Some(TurnOutcome::ToolCalls {
                content: call_content,
                calls,
            })
        }
        None if content.is_empty() => None,
        None => {
            let mut final_content = reasoning;
            final_content.extend(content);
            if terminal_text_allowed(response.stop_reason.as_ref()) {
                ctx.emit(AgentSignal::AssistantMessageDone {
                    message_id,
                    phase: AssistantMessagePhase::FinalAnswer,
                })
                .await?;
            }
            Some(TurnOutcome::FinalAnswer {
                content: final_content,
                text: message_text,
            })
        }
    };
    Ok(Turn {
        outcome,
        stop_reason: response.stop_reason,
    })
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

fn terminal_text_allowed(reason: Option<&StopReason>) -> bool {
    matches!(
        reason,
        Some(StopReason::EndTurn | StopReason::StopSequence) | None
    )
}

fn completed_tool_calls(reason: Option<&StopReason>) -> bool {
    matches!(reason, Some(StopReason::ToolUse) | None)
}

async fn append_delta(
    ctx: &AgentRunContext,
    message_id: &AssistantMessageId,
    content: &mut Vec<ContentBlock>,
    text: &mut String,
    next_index: &mut usize,
    delta: OutputTextDelta,
) -> Result<(), MachineError> {
    append_text(ctx, message_id, content, text, next_index, delta.0).await
}

async fn append_text(
    ctx: &AgentRunContext,
    message_id: &AssistantMessageId,
    content: &mut Vec<ContentBlock>,
    text: &mut String,
    next_index: &mut usize,
    delta: String,
) -> Result<(), MachineError> {
    if delta.is_empty() {
        return Ok(());
    }
    text.push_str(&delta);
    ctx.emit(AgentSignal::AssistantMessageDelta {
        message_id: message_id.clone(),
        delta: delta.clone(),
        index: *next_index,
    })
    .await?;
    *next_index += 1;
    content.push(ContentBlock::Text { text: delta });
    Ok(())
}

async fn append_completed_text(
    ctx: &AgentRunContext,
    message_id: &AssistantMessageId,
    content: &mut Vec<ContentBlock>,
    text: &mut String,
    next_index: &mut usize,
    completed: String,
) -> Result<(), MachineError> {
    if text.is_empty() {
        append_text(ctx, message_id, content, text, next_index, completed).await?;
    }
    Ok(())
}

fn assistant_message_id(ctx: &AgentRunContext, turn: u32) -> AssistantMessageId {
    AssistantMessageId::new(format!("{}:turn-{turn}", ctx.run_id.as_str()))
}
