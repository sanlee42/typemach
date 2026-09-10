use serde::{Deserialize, Serialize};
use serde_json::Value;
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
    /// Stop evidence collection and ask the configured model to write the
    /// final answer from this authoritative capsule.
    Synthesize {
        evidence: Value,
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

pub(super) enum Disposition {
    Present(Presentation),
    Synthesize(Value),
}

impl Disposition {
    pub(super) fn is_synthesis(&self) -> bool {
        matches!(self, Self::Synthesize(_))
    }
}

pub(super) fn take(result: &mut ToolResult) -> Option<Disposition> {
    match std::mem::take(&mut result.disposition) {
        ToolDisposition::Continue => None,
        ToolDisposition::Present { receipt } => Some(Disposition::Present(Presentation {
            tool_use_id: result.tool_use_id.clone(),
            receipt,
        })),
        ToolDisposition::Synthesize { evidence } => Some(Disposition::Synthesize(evidence)),
    }
}

pub(super) fn merge(
    current: &mut Option<Disposition>,
    next: Disposition,
) -> Result<(), AgentError> {
    if let Some(current) = current {
        let reason = match (&*current, &next) {
            (Disposition::Present(_), Disposition::Present(_)) => {
                "a tool batch cannot present more than one final answer"
            }
            (Disposition::Synthesize(_), Disposition::Synthesize(_)) => {
                "a tool batch cannot provide more than one evidence capsule"
            }
            _ => "a tool batch cannot both present an answer and request synthesis",
        };
        return Err(AgentError::InvalidToolResult(reason.to_string()));
    }
    *current = Some(next);
    Ok(())
}

pub(super) fn validate_batch(results: &[ToolResult]) -> Result<(), AgentError> {
    for result in results {
        result.validate()?;
    }
    let presentations = results
        .iter()
        .filter(|result| matches!(result.disposition, ToolDisposition::Present { .. }))
        .count();
    if presentations > 1 {
        return Err(AgentError::InvalidToolResult(
            "a concurrent tool batch cannot present more than one final answer".to_string(),
        ));
    }
    let syntheses = results
        .iter()
        .filter(|result| matches!(result.disposition, ToolDisposition::Synthesize { .. }))
        .count();
    if syntheses > 1 {
        return Err(AgentError::InvalidToolResult(
            "a concurrent tool batch cannot provide more than one evidence capsule".to_string(),
        ));
    }
    if presentations == 1 && syntheses == 1 {
        return Err(AgentError::InvalidToolResult(
            "a concurrent tool batch cannot both present an answer and request synthesis"
                .to_string(),
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
        phase: AssistantMessagePhase::FinalAnswer,
    })
    .await?;
    ctx.emit(AgentSignal::AssistantMessageDelta {
        message_id: message_id.clone(),
        phase: AssistantMessagePhase::FinalAnswer,
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
