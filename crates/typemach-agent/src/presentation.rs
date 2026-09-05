use serde::{Deserialize, Serialize};
use typemach::{MachineError, Transition};

use crate::{
    AgentError, AgentMessage, AgentRunContext, AgentRunOutput, AgentSignal, AgentState, AgentStep,
    AskUserQuestion, AssistantMessageId, AssistantMessagePhase, FinishReason, ToolResult,
    commit_answer,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDisposition {
    #[default]
    Continue,
    Present {
        receipt: String,
    },
}

impl ToolDisposition {
    pub(crate) fn is_continue(&self) -> bool {
        matches!(self, Self::Continue)
    }
}

pub(super) struct Presentation {
    tool_use_id: String,
    receipt: String,
}

pub(super) fn take(result: &mut ToolResult) -> Option<Presentation> {
    match std::mem::take(&mut result.disposition) {
        ToolDisposition::Continue => None,
        ToolDisposition::Present { receipt } => Some(Presentation {
            tool_use_id: result.tool_use_id.clone(),
            receipt,
        }),
    }
}

pub(super) fn merge(
    current: &mut Option<Presentation>,
    next: Presentation,
) -> Result<(), AgentError> {
    if current.is_some() {
        return Err(AgentError::InvalidToolResult(
            "a tool batch cannot present more than one final answer".to_string(),
        ));
    }
    *current = Some(next);
    Ok(())
}

pub(super) fn validate_batch(results: &[ToolResult]) -> Result<(), AgentError> {
    for result in results {
        result.validate()?;
    }
    if results
        .iter()
        .filter(|result| matches!(&result.disposition, ToolDisposition::Present { .. }))
        .count()
        > 1
    {
        return Err(AgentError::InvalidToolResult(
            "a concurrent tool batch cannot present more than one final answer".to_string(),
        ));
    }
    Ok(())
}

pub(super) async fn complete(
    state: &mut AgentState,
    ctx: &AgentRunContext,
    presentation: Presentation,
) -> Result<Transition<AgentStep, AskUserQuestion, AgentRunOutput>, MachineError> {
    let message_id = AssistantMessageId::new(format!(
        "{}:present-{}",
        ctx.run_id.as_str(),
        presentation.tool_use_id
    ));
    ctx.emit(AgentSignal::AssistantMessageStarted {
        message_id: message_id.clone(),
    })
    .await?;
    ctx.emit(AgentSignal::AssistantMessageDelta {
        message_id: message_id.clone(),
        delta: presentation.receipt.clone(),
        index: 0,
    })
    .await?;
    ctx.emit(AgentSignal::AssistantMessageDone {
        message_id,
        phase: AssistantMessagePhase::FinalAnswer,
    })
    .await?;
    state
        .messages
        .push(AgentMessage::assistant_text(presentation.receipt.clone()));
    let answer = commit_answer(state, presentation.receipt);
    Ok(Transition::Complete(
        state.output_with_answer(FinishReason::Stop, answer),
    ))
}
