use tokio::sync::mpsc;
use typemach::MachineError;

use crate::{
    AgentMessage, AgentModel, AgentRunContext, AgentSignal, AgentState, AgentToolSpec,
    ContentBlock, ModelRequest, ModelStream, StopReason, ToolChoice, context,
};

#[derive(Clone, Copy)]
pub(super) enum TextKind {
    Planning,
    FinalAnswer,
}

pub(super) struct Turn {
    pub(super) content: Vec<ContentBlock>,
    pub(super) stop_reason: Option<StopReason>,
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
    kind: TextKind,
) -> Result<Turn, MachineError> {
    let (delta_tx, mut delta_rx) = mpsc::unbounded_channel();
    let response = model.next_step(request, ModelStream::new(delta_tx));
    tokio::pin!(response);
    let mut content = Vec::new();
    let response = loop {
        tokio::select! {
            maybe_delta = delta_rx.recv() => {
                if let Some(delta) = maybe_delta {
                    append_text(state, ctx, &mut content, delta, kind).await?;
                }
            }
            response = &mut response => break response.map_err(MachineError::transition)?,
        }
    };
    while let Ok(delta) = delta_rx.try_recv() {
        append_text(state, ctx, &mut content, delta, kind).await?;
    }
    for delta in response.deltas {
        append_text(state, ctx, &mut content, delta, kind).await?;
    }
    if let Some(usage) = response.usage {
        state.usage.input_tokens += usage.input_tokens;
        state.usage.output_tokens += usage.output_tokens;
        ctx.emit(AgentSignal::Usage { usage }).await?;
    }
    for block in response.content {
        record_block(state, ctx, &mut content, block, kind).await?;
    }
    content.extend(response.tool_uses.into_iter().map(ContentBlock::ToolUse));
    if let Some(text) = response.final_text
        && !text.is_empty()
        && !content
            .iter()
            .any(|block| matches!(block, ContentBlock::Text { .. }))
    {
        append_text(state, ctx, &mut content, text, kind).await?;
    }
    Ok(Turn {
        content,
        stop_reason: response.stop_reason,
    })
}

async fn append_text(
    state: &mut AgentState,
    ctx: &AgentRunContext,
    content: &mut Vec<ContentBlock>,
    delta: String,
    kind: TextKind,
) -> Result<(), MachineError> {
    if delta.is_empty() {
        return Ok(());
    }
    match kind {
        TextKind::Planning => {}
        TextKind::FinalAnswer => {
            state.answer.push_str(&delta);
            ctx.emit(AgentSignal::FinalAnswerDelta {
                delta: delta.clone(),
                index: state.next_final_delta_index,
            })
            .await?;
            state.next_final_delta_index += 1;
        }
    }
    content.push(ContentBlock::Text { text: delta });
    Ok(())
}

async fn record_block(
    state: &mut AgentState,
    ctx: &AgentRunContext,
    content: &mut Vec<ContentBlock>,
    block: ContentBlock,
    kind: TextKind,
) -> Result<(), MachineError> {
    match block {
        ContentBlock::Text { text } => append_text(state, ctx, content, text, kind).await,
        other => {
            content.push(other);
            Ok(())
        }
    }
}
