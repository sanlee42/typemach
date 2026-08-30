use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::future::join_all;
use serde_json::{Value, json};
use typemach::{
    CheckpointSaver, Machine, MachineError, ResumeAction, RunContext, RunEventReceiver, Runner,
    Transition,
};

mod builtins;
mod context;
pub use context::estimate_messages;
mod deepseek;
mod responses;
mod responses_stream;
pub use deepseek::ConfiguredModel;
mod model_turn;
mod phase;
mod stream;
pub use stream::{ModelStream, OutputTextDelta};

mod sandbox;
pub use sandbox::{
    ByteLimit, ExecChild, ExecLimits, ExecSpec, OpenFileLimit, PermissionProfile, SandboxError,
    helper_requested, run_sandbox_helper,
};

pub use typemach as core;

pub type AgentRunContext =
    RunContext<AgentRunInput, AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>;
pub type AgentRunner<M, T, P, S> = Runner<AgentMachine<M, T, P>, S>;
pub type AgentEventReceiver =
    RunEventReceiver<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>;

mod types;
pub use types::*;

use builtins::{
    agent_builtin, artifact_from_tool, ask_user_question, is_terminal_tool, terminal_action,
};

impl AgentState {
    fn fresh(
        input: &AgentRunInput,
        previous: Option<&Self>,
        context_policy: &ContextPolicy,
    ) -> Self {
        let mut messages = previous
            .map(|state| state.messages.clone())
            .unwrap_or_default();
        repair_dangling_tool_uses(&mut messages);
        messages.extend(input.messages.clone());
        Self {
            messages,
            context: input.context.clone(),
            budget: input.budget.clone(),
            context_policy: context_policy.clone(),
            system_suffix: input.system_suffix.clone(),
            model_turns: 0,
            tool_calls: 0,
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
        }
    }

    fn output(&self, finish_reason: FinishReason) -> AgentRunOutput {
        self.output_with_answer(finish_reason, String::new())
    }

    fn output_with_answer(&self, finish_reason: FinishReason, answer: String) -> AgentRunOutput {
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

impl AgentError {
    fn machine(self) -> MachineError {
        MachineError::transition(self)
    }
}

#[async_trait]
pub trait AgentModel: Send + Sync {
    async fn next_step(
        &self,
        request: ModelRequest,
        stream: ModelStream,
    ) -> Result<ModelResponse, AgentError>;
}

#[async_trait]
pub trait ToolRegistry: Send + Sync {
    async fn list_tools(&self, context: &Value) -> Result<Vec<AgentToolSpec>, AgentError>;
    async fn call_tool(&self, request: ToolCallRequest) -> Result<ToolResult, AgentError>;
}

pub trait ToolPermissionPolicy: Send + Sync {
    fn check(
        &self,
        tool: &ToolUse,
        spec: Option<&AgentToolSpec>,
        context: &Value,
    ) -> PermissionDecision;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionDecision {
    Allow,
    Deny(String),
}

#[derive(Debug, Clone, Default)]
pub struct AllowAllTools;

impl ToolPermissionPolicy for AllowAllTools {
    fn check(
        &self,
        _tool: &ToolUse,
        _spec: Option<&AgentToolSpec>,
        _context: &Value,
    ) -> PermissionDecision {
        PermissionDecision::Allow
    }
}

pub struct AgentMachine<M, T, P> {
    model: Arc<M>,
    tools: Arc<T>,
    policy: Arc<P>,
    context_policy: ContextPolicy,
}

impl<M, T, P> AgentMachine<M, T, P> {
    pub fn new(model: M, tools: T, policy: P) -> Self {
        Self {
            model: Arc::new(model),
            tools: Arc::new(tools),
            policy: Arc::new(policy),
            context_policy: ContextPolicy::default(),
        }
    }

    pub fn with_context_policy(mut self, context_policy: ContextPolicy) -> Self {
        self.context_policy = context_policy;
        self
    }
}

pub fn build_agent_runner<S, M, T, P>(
    checkpointer: S,
    model: M,
    tools: T,
    policy: P,
) -> AgentRunner<M, T, P, S>
where
    S: CheckpointSaver + 'static,
    M: AgentModel + 'static,
    T: ToolRegistry + 'static,
    P: ToolPermissionPolicy + 'static,
{
    Runner::new(
        AgentMachine::new(model, tools, policy),
        Arc::new(checkpointer),
    )
}

pub fn build_agent_runner_with_context_policy<S, M, T, P>(
    checkpointer: S,
    model: M,
    tools: T,
    policy: P,
    context_policy: ContextPolicy,
) -> AgentRunner<M, T, P, S>
where
    S: CheckpointSaver + 'static,
    M: AgentModel + 'static,
    T: ToolRegistry + 'static,
    P: ToolPermissionPolicy + 'static,
{
    Runner::new(
        AgentMachine::new(model, tools, policy).with_context_policy(context_policy),
        Arc::new(checkpointer),
    )
}

pub fn build_configured_agent_runner<S, T, P>(
    checkpointer: S,
    config: AgentConfig,
    tools: T,
    policy: P,
) -> Result<AgentRunner<ConfiguredModel, T, P, S>, AgentError>
where
    S: CheckpointSaver + 'static,
    T: ToolRegistry + 'static,
    P: ToolPermissionPolicy + 'static,
{
    let context_policy = config.context_policy.clone();
    let model = ConfiguredModel::new(config)?;
    Ok(build_agent_runner_with_context_policy(
        checkpointer,
        model,
        tools,
        policy,
        context_policy,
    ))
}

#[async_trait]
impl<M, T, P> Machine for AgentMachine<M, T, P>
where
    M: AgentModel + 'static,
    T: ToolRegistry + 'static,
    P: ToolPermissionPolicy + 'static,
{
    type Step = AgentStep;
    type State = AgentState;
    type Input = AgentRunInput;
    type Signal = AgentSignal;
    type Output = AgentRunOutput;
    type Interrupt = AskUserQuestion;

    fn start_step(&self) -> Self::Step {
        AgentStep::PrepareTurn
    }

    fn resume_action(&self, _interrupt: &Self::Interrupt) -> ResumeAction<Self::Step> {
        ResumeAction::JumpTo(AgentStep::DispatchTools)
    }

    fn new_state(
        &self,
        input: &Self::Input,
        previous: Option<&Self::State>,
        _snapshot: Option<&Value>,
    ) -> Result<Self::State, MachineError> {
        Ok(AgentState::fresh(input, previous, &self.context_policy))
    }

    fn apply_resume_input(
        &self,
        state: &mut Self::State,
        input: &Self::Input,
    ) -> Result<(), MachineError> {
        state.human_input = input.human_input.clone();
        state.context = input.context.clone();
        state.system_suffix = input.system_suffix.clone();
        if state.human_input.is_some()
            && let Some(tool_use) = state.pending_human.take()
        {
            state.pending_tools.push_front(tool_use);
        }
        Ok(())
    }

    async fn transition(
        &self,
        step: Self::Step,
        state: &mut Self::State,
        ctx: &AgentRunContext,
    ) -> Result<Transition<Self::Step, Self::Interrupt, Self::Output>, MachineError> {
        match step {
            AgentStep::PrepareTurn => Ok(Transition::Next(AgentStep::ModelStep)),
            AgentStep::ModelStep => self.planning_step(state, ctx).await,
            AgentStep::DispatchTools => self.dispatch_tools(state, ctx).await,
            AgentStep::FinalAnswer => self.final_answer_step(state, ctx).await,
        }
    }
}

impl<M, T, P> AgentMachine<M, T, P>
where
    M: AgentModel + 'static,
    T: ToolRegistry + 'static,
    P: ToolPermissionPolicy + 'static,
{
    async fn planning_step(
        &self,
        state: &mut AgentState,
        ctx: &AgentRunContext,
    ) -> Result<Transition<AgentStep, AskUserQuestion, AgentRunOutput>, MachineError> {
        if planning_budget_exhausted(state) {
            enter_final_answer(state);
            return Ok(Transition::Next(AgentStep::FinalAnswer));
        }
        let tools = self
            .tools
            .list_tools(&state.context)
            .await
            .map_err(AgentError::machine)?;
        let messages = state.messages.clone();
        let suffix = state.system_suffix.clone();
        state.model_turns += 1;
        let planning_turn = state.model_turns;
        let request = model_turn::prepare(
            state,
            ctx,
            messages,
            tools.clone(),
            suffix,
            Some(ToolChoice::Auto),
            planning_turn,
        )
        .await?;
        let turn = model_turn::invoke(self.model.as_ref(), state, ctx, request).await?;
        match turn.outcome {
            Some(model_turn::TurnOutcome::FinalAnswer { content, text }) => {
                let reason = finish_reason(turn.stop_reason.as_ref())?;
                if !content.is_empty() {
                    state.messages.push(AgentMessage::Assistant { content });
                }
                let answer = commit_answer(state, reason.clone(), text);
                Ok(Transition::Complete(
                    state.output_with_answer(reason, answer),
                ))
            }
            Some(model_turn::TurnOutcome::ToolCalls {
                content,
                calls: tool_uses,
            }) => {
                if !matches!(turn.stop_reason, Some(StopReason::ToolUse) | None) {
                    return Err(AgentError::Model(
                        "planning stopped before completing tool calls".to_string(),
                    )
                    .machine());
                }
                if tool_uses.is_empty() {
                    return Err(AgentError::Model(
                        "model returned an empty tool call set".to_string(),
                    )
                    .machine());
                }
                let remaining =
                    state.budget.max_tool_calls.saturating_sub(state.tool_calls) as usize;
                if tool_uses.len() > remaining {
                    enter_final_answer(state);
                    return Ok(Transition::Next(AgentStep::FinalAnswer));
                }
                state
                    .pending_tools
                    .extend(tool_uses.clone().into_iter().map(|tool_use| {
                        let spec = tools.iter().find(|spec| spec.name == tool_use.name);
                        PendingToolCall::new(tool_use, spec.cloned())
                    }));
                state.messages.push(AgentMessage::Assistant { content });
                Ok(Transition::Next(AgentStep::DispatchTools))
            }
            None if turn.stop_reason == Some(StopReason::MaxTokens) => {
                state.pending_tools.clear();
                Ok(Transition::Complete(state.output(FinishReason::MaxTokens)))
            }
            None => Err(no_outcome_error(turn.stop_reason).machine()),
        }
    }

    async fn final_answer_step(
        &self,
        state: &mut AgentState,
        ctx: &AgentRunContext,
    ) -> Result<Transition<AgentStep, AskUserQuestion, AgentRunOutput>, MachineError> {
        let messages = phase::final_messages(&state.messages);
        let suffix = phase::final_system_suffix(state.system_suffix.as_deref());
        let final_turn = state.model_turns.saturating_add(1);
        let request = model_turn::prepare(
            state,
            ctx,
            messages.clone(),
            Vec::new(),
            Some(suffix),
            Some(ToolChoice::None),
            final_turn,
        )
        .await?;
        let turn = model_turn::invoke(self.model.as_ref(), state, ctx, request).await?;
        let (content, text) = match turn.outcome {
            Some(model_turn::TurnOutcome::FinalAnswer { content, text }) => (content, text),
            Some(model_turn::TurnOutcome::ToolCalls { .. }) => {
                return Err(AgentError::Model(
                    "final answer step returned a tool call even though tools were disabled"
                        .to_string(),
                )
                .machine());
            }
            None if turn.stop_reason == Some(StopReason::MaxTokens) => (Vec::new(), String::new()),
            None => return Err(no_outcome_error(turn.stop_reason).machine()),
        };
        let reason = finish_reason(turn.stop_reason.as_ref())?;
        if !content.is_empty() {
            state.messages.push(AgentMessage::Assistant { content });
        }
        let answer = commit_answer(state, reason.clone(), text);
        Ok(Transition::Complete(
            state.output_with_answer(reason, answer),
        ))
    }

    async fn dispatch_tools(
        &self,
        state: &mut AgentState,
        ctx: &AgentRunContext,
    ) -> Result<Transition<AgentStep, AskUserQuestion, AgentRunOutput>, MachineError> {
        let remaining = state.budget.max_tool_calls.saturating_sub(state.tool_calls) as usize;
        if state.pending_tools.len() > remaining {
            state.pending_tools.clear();
            enter_final_answer(state);
            return Ok(Transition::Next(AgentStep::FinalAnswer));
        }
        if self.dispatch_concurrent_read_only(state, ctx).await? {
            return if planning_budget_exhausted(state) {
                enter_final_answer(state);
                Ok(Transition::Next(AgentStep::FinalAnswer))
            } else {
                Ok(Transition::Next(AgentStep::ModelStep))
            };
        }
        while let Some(tool_use) = state.pending_tools.pop_front() {
            let spec = tool_use.spec.as_ref();
            let built_in_error = if tool_use.tool_use.name == "ask_user" {
                if let Some(result) = self
                    .consume_human_answer(state, &tool_use.tool_use, ctx)
                    .await?
                {
                    state.messages.push(AgentMessage::tool_result(result));
                    continue;
                }
                match ask_user_question(&tool_use.tool_use) {
                    Ok(question) => {
                        state.pending_human = Some(tool_use);
                        return Ok(Transition::Interrupt(question));
                    }
                    Err(err) => Some(ToolResult::error(&tool_use.tool_use, err.to_string())),
                }
            } else {
                None
            };
            let terminal = is_terminal_tool(&tool_use.tool_use, spec)
                && tool_use.tool_use.name != "emit_artifact";
            if terminal && built_in_error.is_none() {
                let action = terminal_action(&tool_use.tool_use);
                ctx.emit(AgentSignal::Terminal {
                    action: action.clone(),
                })
                .await?;
                state.terminal = Some(action);
                return Ok(Transition::Complete(
                    state.output_with_answer(FinishReason::Terminal, state.answer.clone()),
                ));
            }
            state.tool_calls += 1;
            ctx.emit(AgentSignal::ToolStarted {
                tool_use_id: tool_use.tool_use.id.clone(),
                name: tool_use.tool_use.name.clone(),
            })
            .await?;
            let result = if let Some(result) = built_in_error {
                result
            } else if tool_use.tool_use.name == "emit_artifact" {
                match artifact_from_tool(&tool_use.tool_use) {
                    Ok(artifact) => {
                        self.emit_artifact(state, ctx, &tool_use.tool_use, artifact)
                            .await?
                    }
                    Err(err) => ToolResult::error(&tool_use.tool_use, err.to_string()),
                }
            } else {
                match self.policy.check(&tool_use.tool_use, spec, &state.context) {
                    PermissionDecision::Allow => self
                        .tools
                        .call_tool(ToolCallRequest {
                            tool_use: tool_use.tool_use.clone(),
                            context: state.context.clone(),
                        })
                        .await
                        .unwrap_or_else(|err| {
                            ToolResult::error(&tool_use.tool_use, err.to_string())
                        }),
                    PermissionDecision::Deny(reason) => {
                        ToolResult::error(&tool_use.tool_use, reason)
                    }
                }
            };
            record_tool_result(state, ctx, result).await?;
        }
        if planning_budget_exhausted(state) {
            enter_final_answer(state);
            Ok(Transition::Next(AgentStep::FinalAnswer))
        } else {
            Ok(Transition::Next(AgentStep::ModelStep))
        }
    }

    async fn consume_human_answer(
        &self,
        state: &mut AgentState,
        tool_use: &ToolUse,
        ctx: &AgentRunContext,
    ) -> Result<Option<ToolResult>, MachineError> {
        let Some(answer) = state.human_input.take() else {
            return Ok(None);
        };
        if answer.tool_use_id != tool_use.id {
            state.pending_human = None;
            return Ok(Some(ToolResult::error(
                tool_use,
                format!(
                    "human answer targets {}, expected {}",
                    answer.tool_use_id, tool_use.id
                ),
            )));
        }
        state.tool_calls += 1;
        ctx.emit(AgentSignal::ToolStarted {
            tool_use_id: tool_use.id.clone(),
            name: tool_use.name.clone(),
        })
        .await?;
        let result = ToolResult::ok(tool_use, json!({ "answer": answer.answer }));
        ctx.emit(AgentSignal::ToolResult {
            tool_use_id: result.tool_use_id.clone(),
            name: result.name.clone(),
            content: result.content.clone(),
            is_error: false,
        })
        .await?;
        ctx.emit(AgentSignal::ToolCompleted {
            tool_use_id: result.tool_use_id.clone(),
            name: result.name.clone(),
            is_error: false,
        })
        .await?;
        state.pending_human = None;
        Ok(Some(result))
    }

    async fn emit_artifact(
        &self,
        state: &mut AgentState,
        ctx: &AgentRunContext,
        tool_use: &ToolUse,
        artifact: Artifact,
    ) -> Result<ToolResult, MachineError> {
        state.artifacts.push(artifact.clone());
        ctx.emit(AgentSignal::Artifact { artifact }).await?;
        Ok(ToolResult::ok(tool_use, json!({ "ok": true })))
    }

    async fn dispatch_concurrent_read_only(
        &self,
        state: &mut AgentState,
        ctx: &AgentRunContext,
    ) -> Result<bool, MachineError> {
        if state.pending_tools.is_empty() || !self.concurrent_batch_ready(state) {
            return Ok(false);
        }
        let batch = state.pending_tools.drain(..).collect::<Vec<_>>();
        state.tool_calls += batch.len() as u32;
        for pending in &batch {
            ctx.emit(AgentSignal::ToolStarted {
                tool_use_id: pending.tool_use.id.clone(),
                name: pending.tool_use.name.clone(),
            })
            .await?;
        }
        let context = state.context.clone();
        let calls = batch.iter().map(|pending| {
            let tools = Arc::clone(&self.tools);
            let context = context.clone();
            let tool_use = pending.tool_use.clone();
            async move {
                tools
                    .call_tool(ToolCallRequest {
                        tool_use: tool_use.clone(),
                        context,
                    })
                    .await
                    .unwrap_or_else(|err| ToolResult::error(&tool_use, err.to_string()))
            }
        });
        for result in join_all(calls).await {
            record_tool_result(state, ctx, result).await?;
        }
        Ok(true)
    }

    fn concurrent_batch_ready(&self, state: &AgentState) -> bool {
        state.pending_tools.iter().all(|pending| {
            let Some(spec) = pending.spec.as_ref() else {
                return false;
            };
            spec.annotations.read_only
                && !spec.annotations.destructive
                && !spec.annotations.open_world
                && !spec.annotations.terminal
                && !agent_builtin(&pending.tool_use)
                && !is_terminal_tool(&pending.tool_use, Some(spec))
                && self
                    .policy
                    .check(&pending.tool_use, Some(spec), &state.context)
                    == PermissionDecision::Allow
        })
    }
}

/// A run started over an inherited transcript may find tool calls whose
/// results never arrived (abandoned ask_user, disconnect mid-dispatch).
/// Provider protocols reject such transcripts outright, so close every
/// dangling call with a synthetic error result.
fn repair_dangling_tool_uses(messages: &mut Vec<AgentMessage>) {
    let mut resulted = std::collections::HashSet::new();
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

fn enter_final_answer(state: &mut AgentState) {
    state.pending_tools.clear();
    state.pending_human = None;
    state.human_input = None;
}

fn commit_answer(state: &mut AgentState, reason: FinishReason, text: String) -> String {
    if reason == FinishReason::Stop {
        state.answer = text;
        state.answer.clone()
    } else {
        String::new()
    }
}

fn planning_budget_exhausted(state: &AgentState) -> bool {
    state.model_turns >= state.budget.max_model_turns
        || state.tool_calls >= state.budget.max_tool_calls
}

fn finish_reason(reason: Option<&StopReason>) -> Result<FinishReason, MachineError> {
    match reason {
        Some(StopReason::EndTurn | StopReason::StopSequence) | None => Ok(FinishReason::Stop),
        Some(StopReason::MaxTokens) => Ok(FinishReason::MaxTokens),
        Some(StopReason::Refusal) => {
            Err(AgentError::Model("final answer was refused".to_string()).machine())
        }
        Some(StopReason::ToolUse) => Err(AgentError::Model(
            "final answer stopped for a tool call without returning one".to_string(),
        )
        .machine()),
        Some(StopReason::Other(reason)) => {
            Err(AgentError::Model(format!("final answer stopped unexpectedly: {reason}")).machine())
        }
    }
}

fn no_outcome_error(reason: Option<StopReason>) -> AgentError {
    match reason {
        Some(StopReason::EndTurn) => AgentError::Model(
            "planning ended without a typed final answer or tool calls".to_string(),
        ),
        Some(reason) => AgentError::Model(format!(
            "planning stopped without a tool or natural completion: {reason:?}"
        )),
        None => AgentError::Model("planning stopped without a tool or stop reason".to_string()),
    }
}

async fn record_tool_result(
    state: &mut AgentState,
    ctx: &AgentRunContext,
    result: ToolResult,
) -> Result<(), MachineError> {
    ctx.emit(AgentSignal::ToolResult {
        tool_use_id: result.tool_use_id.clone(),
        name: result.name.clone(),
        content: result.content.clone(),
        is_error: result.is_error,
    })
    .await?;
    let (prompt_result, archive) =
        context::maybe_archive_tool_result(&result, &state.context_policy)
            .map_err(AgentError::machine)?;
    if let Some(archive) = archive {
        state.tool_result_archives.push(archive.clone());
        ctx.emit(AgentSignal::ToolResultArchived { archive })
            .await?;
    }
    ctx.emit(AgentSignal::ToolCompleted {
        tool_use_id: result.tool_use_id.clone(),
        name: result.name.clone(),
        is_error: result.is_error,
    })
    .await?;
    state
        .messages
        .push(AgentMessage::tool_result(prompt_result));
    Ok(())
}
