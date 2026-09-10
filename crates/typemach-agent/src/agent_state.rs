use std::collections::{HashSet, VecDeque};

use crate::{
    AgentError, AgentMessage, AgentPhase, AgentRunInput, AgentRunOutput, AgentState, ContentBlock,
    ContextPolicy, FinishReason, ToolResult, Usage, retained_result,
};

impl AgentState {
    pub(super) fn fresh(
        input: &AgentRunInput,
        previous: Option<&Self>,
        context_policy: &ContextPolicy,
    ) -> Result<Self, AgentError> {
        retained_result::validate(&input.retained_results)?;
        if let Some(request) = &input.synthesis_request {
            validate_synthesis_request(request)?;
        }
        let mut messages = previous
            .map(|state| state.messages.clone())
            .unwrap_or_default();
        repair_dangling_tool_uses(&mut messages);
        messages.extend(input.messages.clone());
        Ok(Self {
            messages,
            synthesis_request: input.synthesis_request.clone(),
            synthesis_evidence: None,
            context: input.context.clone(),
            retained_results: input.retained_results.clone(),
            budget: input.budget.clone(),
            phase: AgentPhase::Evidence,
            context_policy: context_policy.clone(),
            system_suffix: input.system_suffix.clone(),
            model_turns: 0,
            tool_calls: 0,
            loaded_deferred_tools: Default::default(),
            pending_tools: VecDeque::new(),
            pending_human: None,
            human_input: input.human_input.clone(),
            answer: String::new(),
            usage: Usage::default(),
            artifacts: Vec::new(),
            terminal: None,
            digest: previous.and_then(|state| state.digest.clone()),
            tool_result_archives: previous
                .map(|state| state.tool_result_archives.clone())
                .unwrap_or_default(),
        })
    }

    pub(super) fn output_with_answer(
        &self,
        finish_reason: FinishReason,
        answer: String,
    ) -> AgentRunOutput {
        AgentRunOutput {
            messages: self.messages.clone(),
            answer,
            finish_reason,
            terminal: self.terminal.clone(),
            usage: self.usage.clone(),
            artifacts: self.artifacts.clone(),
            digest: self.digest.clone(),
            tool_result_archives: self.tool_result_archives.clone(),
        }
    }
}

pub(super) fn validate_synthesis_request(request: &AgentMessage) -> Result<(), AgentError> {
    let AgentMessage::User { content } = request else {
        return Err(AgentError::Config(
            "synthesis_request must be a user message".to_string(),
        ));
    };
    if content.is_empty()
        || content
            .iter()
            .any(|block| !matches!(block, ContentBlock::Text { text } if !text.trim().is_empty()))
    {
        return Err(AgentError::Config(
            "synthesis_request must contain only non-empty text blocks".to_string(),
        ));
    }
    Ok(())
}

/// A run started over an inherited transcript may find tool calls whose
/// results never arrived (abandoned ask_user, disconnect mid-dispatch).
/// Provider protocols reject such transcripts outright, so close every
/// dangling call with a synthetic error result.
fn repair_dangling_tool_uses(messages: &mut Vec<AgentMessage>) {
    let mut resulted = HashSet::new();
    for message in messages.iter() {
        let (AgentMessage::User { content } | AgentMessage::Assistant { content }) = message;
        for block in content {
            if let ContentBlock::ToolResult(result) = block {
                resulted.insert(result.tool_use_id.clone());
            }
        }
    }
    let mut dangling = Vec::new();
    for message in messages.iter() {
        if let AgentMessage::Assistant { content } = message {
            for block in content {
                if let ContentBlock::ToolUse(tool_use) = block
                    && !resulted.contains(&tool_use.id)
                {
                    dangling.push(tool_use.clone());
                }
            }
        }
    }
    for tool_use in dangling {
        messages.push(AgentMessage::tool_result(ToolResult::error(
            &tool_use,
            "interrupted before completion",
        )));
    }
}
