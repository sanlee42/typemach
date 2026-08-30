use tokio::sync::mpsc;
use typemach::MachineError;

use crate::{
    AgentMessage, AgentModel, AgentRunContext, AgentSignal, AgentState, AgentToolSpec,
    ContentBlock, ModelOutcome, ModelRequest, ModelStream, OutputTextDelta, StopReason, ToolChoice,
    ToolUse, context,
};

pub(super) struct Turn {
    pub(super) outcome: Option<TurnOutcome>,
    pub(super) stop_reason: Option<StopReason>,
}

pub(super) enum TurnOutcome {
    FinalAnswer(Vec<ContentBlock>),
    ToolCalls(Vec<ToolUse>),
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
    let response = model.next_step(request, ModelStream::new(delta_tx));
    tokio::pin!(response);
    let mut content = Vec::new();
    let response = loop {
        tokio::select! {
            maybe_delta = delta_rx.recv() => {
                if let Some(delta) = maybe_delta {
                    append_delta(state, ctx, &mut content, delta).await?;
                }
            }
            response = &mut response => break response.map_err(MachineError::transition)?,
        }
    };
    while let Ok(delta) = delta_rx.try_recv() {
        append_delta(state, ctx, &mut content, delta).await?;
    }
    if let Some(usage) = response.usage {
        state.usage.input_tokens += usage.input_tokens;
        state.usage.output_tokens += usage.output_tokens;
        ctx.emit(AgentSignal::Usage { usage }).await?;
    }
    let outcome = match response.outcome {
        Some(ModelOutcome::FinalAnswer { text }) => {
            append_text(state, ctx, &mut content, text).await?;
            Some(TurnOutcome::FinalAnswer(content))
        }
        Some(ModelOutcome::ToolCalls { calls }) => Some(TurnOutcome::ToolCalls(calls)),
        None if content.is_empty() => None,
        None => Some(TurnOutcome::FinalAnswer(content)),
    };
    Ok(Turn {
        outcome,
        stop_reason: response.stop_reason,
    })
}

async fn append_delta(
    state: &mut AgentState,
    ctx: &AgentRunContext,
    content: &mut Vec<ContentBlock>,
    delta: OutputTextDelta,
) -> Result<(), MachineError> {
    append_text(state, ctx, content, delta.0).await
}

async fn append_text(
    state: &mut AgentState,
    ctx: &AgentRunContext,
    content: &mut Vec<ContentBlock>,
    delta: String,
) -> Result<(), MachineError> {
    if delta.is_empty() {
        return Ok(());
    }
    state.answer.push_str(&delta);
    ctx.emit(AgentSignal::FinalAnswerDelta {
        delta: delta.clone(),
        index: state.next_final_delta_index,
    })
    .await?;
    state.next_final_delta_index += 1;
    content.push(ContentBlock::Text { text: delta });
    Ok(())
}
