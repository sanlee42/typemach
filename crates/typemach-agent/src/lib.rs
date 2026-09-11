use std::sync::Arc;

use async_trait::async_trait;
use futures_util::future::join_all;
use serde_json::{Value, json};
use typemach::{
    CheckpointSaver, Machine, MachineError, ResumeAction, RunContext, RunEventReceiver, Runner,
    Transition,
};

mod agent_state;
mod builtins;
mod context;
pub use context::estimate_messages;
mod deepseek;
mod deferred_tools;
pub use deferred_tools::DeferredToolName;
mod message_item;
mod responses;
mod responses_stream;
pub use deepseek::ConfiguredModel;
pub use message_item::*;
mod model_turn;
mod pending_tool;
pub use pending_tool::PendingToolCall;
mod presentation;
pub use presentation::ToolDisposition;
mod retained_result;
pub use retained_result::{ResultId, RetainedResult};
mod stream;
pub use stream::{ModelStream, ModelStreamEvent};

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
    /// Tools available immediately for the current context.
    async fn list_tools(&self, context: &Value) -> Result<Vec<AgentToolSpec>, AgentError>;
    /// Authorized tools discoverable through the agent's built-in `tool_search`.
    /// Names must be unique across both catalogs and cannot be `tool_search`.
    /// Discovered names persist until the next fresh turn and are revalidated
    /// against this catalog before model calls and tool dispatch.
    async fn list_deferred_tools(
        &self,
        _context: &Value,
    ) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok(Vec::new())
    }
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

enum ToolDispatch {
    Continue,
    Disposition(presentation::Presentation),
    Transition(Box<Transition<AgentStep, AskUserQuestion, AgentRunOutput>>),
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
        AgentState::fresh(input, previous, &self.context_policy).map_err(AgentError::machine)
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
            AgentStep::ModelStep => self.model_step(state, ctx).await,
            AgentStep::DispatchTools => self.dispatch_tools(state, ctx).await,
        }
    }
}

impl<M, T, P> AgentMachine<M, T, P>
where
    M: AgentModel + 'static,
    T: ToolRegistry + 'static,
    P: ToolPermissionPolicy + 'static,
{
    async fn model_step(
        &self,
        state: &mut AgentState,
        ctx: &AgentRunContext,
    ) -> Result<Transition<AgentStep, AskUserQuestion, AgentRunOutput>, MachineError> {
        check_budget(state).map_err(|reason| AgentError::Incomplete(reason).machine())?;
        let tools = deferred_tools::ToolCatalog::read(
            self.tools.as_ref(),
            &state.context,
            &mut state.loaded_deferred_tools,
        )
        .await
        .map_err(AgentError::machine)?
        .visible;
        state.model_turns += 1;
        let turn_number = state.model_turns;
        let request = model_turn::prepare(
            state,
            ctx,
            state.messages.clone(),
            tools.clone(),
            state.system_suffix.clone(),
            Some(ToolChoice::Auto),
            turn_number,
        )
        .await?;
        let turn = model_turn::invoke(self.model.as_ref(), state, ctx, request).await?;
        match turn.outcome {
            Some(model_turn::TurnOutcome::Message { content, text }) => {
                let reason = finish_reason(turn.stop_reason.as_ref())?;
                if reason == FinishReason::MaxTokens {
                    return Err(AgentError::Incomplete(reason).machine());
                }
                if text.trim().is_empty() {
                    return Err(
                        AgentError::Model("assistant message was empty".to_string()).machine()
                    );
                }
                if !content.is_empty() {
                    state.messages.push(AgentMessage::Assistant { content });
                }
                let answer = commit_answer(state, text);
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
                        "model stopped before completing tool calls".to_string(),
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
                state.messages.push(AgentMessage::Assistant { content });
                if tool_uses.len() > remaining {
                    close_budget_exhausted_batch(state, &tool_uses)?;
                    return Ok(Transition::Next(AgentStep::ModelStep));
                }
                state
                    .pending_tools
                    .extend(tool_uses.clone().into_iter().map(|tool_use| {
                        let spec = tools.iter().find(|spec| spec.name == tool_use.name);
                        PendingToolCall::new(tool_use, spec.cloned())
                    }));
                Ok(Transition::Next(AgentStep::DispatchTools))
            }
            None if turn.stop_reason == Some(StopReason::MaxTokens) => {
                state.pending_tools.clear();
                Err(AgentError::Incomplete(FinishReason::MaxTokens).machine())
            }
            None => Err(no_outcome_error(turn.stop_reason).machine()),
        }
    }

    async fn dispatch_tools(
        &self,
        state: &mut AgentState,
        ctx: &AgentRunContext,
    ) -> Result<Transition<AgentStep, AskUserQuestion, AgentRunOutput>, MachineError> {
        let mut disposition = None;
        let mut catalog = deferred_tools::ToolCatalog::read(
            self.tools.as_ref(),
            &state.context,
            &mut state.loaded_deferred_tools,
        )
        .await
        .map_err(AgentError::machine)?;
        for pending in &mut state.pending_tools {
            catalog.refresh_pending(pending, &mut state.loaded_deferred_tools);
        }
        let remaining = state.budget.max_tool_calls.saturating_sub(state.tool_calls) as usize;
        if state.pending_tools.len() > remaining {
            let tool_uses = state
                .pending_tools
                .drain(..)
                .map(|pending| pending.tool_use)
                .collect::<Vec<_>>();
            close_budget_exhausted_batch(state, &tool_uses)?;
            return Ok(Transition::Next(AgentStep::ModelStep));
        }
        if self.concurrent_batch_ready(state) {
            let checked = state
                .pending_tools
                .drain(..)
                .map(|pending| {
                    let permission =
                        deferred_tools::permission(self.policy.as_ref(), &pending, &state.context);
                    (pending, permission)
                })
                .collect::<Vec<_>>();
            if checked
                .iter()
                .all(|(_, permission)| *permission == PermissionDecision::Allow)
            {
                let batch = checked.into_iter().map(|(pending, _)| pending).collect();
                disposition = self
                    .dispatch_concurrent_read_only(state, ctx, batch)
                    .await?;
            } else {
                for (pending, permission) in checked {
                    match self
                        .dispatch_checked_tool(state, ctx, pending, permission, &catalog)
                        .await?
                    {
                        ToolDispatch::Continue => {}
                        ToolDispatch::Disposition(next) => {
                            presentation::merge(&mut disposition, next)
                                .map_err(AgentError::machine)?;
                        }
                        ToolDispatch::Transition(transition) => return Ok(*transition),
                    }
                }
            }
        } else {
            while let Some(pending) = state.pending_tools.pop_front() {
                let permission =
                    deferred_tools::permission(self.policy.as_ref(), &pending, &state.context);
                match self
                    .dispatch_checked_tool(state, ctx, pending, permission, &catalog)
                    .await?
                {
                    ToolDispatch::Continue => {}
                    ToolDispatch::Disposition(next) => {
                        presentation::merge(&mut disposition, next).map_err(AgentError::machine)?;
                    }
                    ToolDispatch::Transition(transition) => return Ok(*transition),
                }
            }
        }
        if let Some(presentation) = disposition {
            return presentation::complete(state, ctx, presentation).await;
        }
        Ok(Transition::Next(AgentStep::ModelStep))
    }

    async fn dispatch_checked_tool(
        &self,
        state: &mut AgentState,
        ctx: &AgentRunContext,
        pending: PendingToolCall,
        permission: PermissionDecision,
        catalog: &deferred_tools::ToolCatalog,
    ) -> Result<ToolDispatch, MachineError> {
        let tool_use = &pending.tool_use;
        if let PermissionDecision::Deny(reason) = permission {
            if tool_use.name == "ask_user" {
                state.pending_human = None;
                state.human_input = None;
            }
            state.tool_calls += 1;
            ctx.emit(AgentSignal::ToolStarted {
                tool_use_id: tool_use.id.clone(),
                name: tool_use.name.clone(),
            })
            .await?;
            let _ = record_tool_result(state, ctx, ToolResult::error(tool_use, reason)).await?;
            return Ok(ToolDispatch::Continue);
        }
        let spec = pending.spec();
        let built_in_error = if tool_use.name == "ask_user" {
            if let Some(result) = self.consume_human_answer(state, tool_use, ctx).await? {
                state.messages.push(AgentMessage::tool_result(result));
                return Ok(ToolDispatch::Continue);
            }
            match ask_user_question(tool_use) {
                Ok(question) => {
                    state.pending_human = Some(pending);
                    return Ok(ToolDispatch::Transition(Box::new(Transition::Interrupt(
                        question,
                    ))));
                }
                Err(err) => Some(ToolResult::error(tool_use, err.to_string())),
            }
        } else {
            None
        };
        let terminal = is_terminal_tool(tool_use, spec) && tool_use.name != "emit_artifact";
        if terminal && built_in_error.is_none() {
            let action = terminal_action(tool_use);
            ctx.emit(AgentSignal::Terminal {
                action: action.clone(),
            })
            .await?;
            state.terminal = Some(action);
            return Ok(ToolDispatch::Transition(Box::new(Transition::Complete(
                state.output_with_answer(FinishReason::Terminal, state.answer.clone()),
            ))));
        }
        state.tool_calls += 1;
        ctx.emit(AgentSignal::ToolStarted {
            tool_use_id: tool_use.id.clone(),
            name: tool_use.name.clone(),
        })
        .await?;
        let result = if let Some(result) = built_in_error {
            result
        } else if tool_use.name == deferred_tools::SEARCH_NAME {
            catalog.search(tool_use, &mut state.loaded_deferred_tools)
        } else if tool_use.name == "emit_artifact" {
            match artifact_from_tool(tool_use) {
                Ok(artifact) => self.emit_artifact(state, ctx, tool_use, artifact).await?,
                Err(err) => ToolResult::error(tool_use, err.to_string()),
            }
        } else {
            self.tools
                .call_tool(ToolCallRequest {
                    tool_use: tool_use.clone(),
                    context: state.context.clone(),
                    retained_results: state.retained_results.clone(),
                })
                .await
                .unwrap_or_else(|err| ToolResult::error(tool_use, err.to_string()))
        };
        Ok(match record_tool_result(state, ctx, result).await? {
            Some(disposition) => ToolDispatch::Disposition(disposition),
            None => ToolDispatch::Continue,
        })
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
            retained_authorization: None,
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
        batch: Vec<PendingToolCall>,
    ) -> Result<Option<presentation::Presentation>, MachineError> {
        state.tool_calls += batch.len() as u32;
        for pending in &batch {
            ctx.emit(AgentSignal::ToolStarted {
                tool_use_id: pending.tool_use.id.clone(),
                name: pending.tool_use.name.clone(),
            })
            .await?;
        }
        let context = state.context.clone();
        let retained_results = state.retained_results.clone();
        let calls = batch.iter().map(|pending| {
            let tools = Arc::clone(&self.tools);
            let context = context.clone();
            let retained_results = retained_results.clone();
            let tool_use = pending.tool_use.clone();
            async move {
                tools
                    .call_tool(ToolCallRequest {
                        tool_use: tool_use.clone(),
                        context,
                        retained_results,
                    })
                    .await
                    .unwrap_or_else(|err| ToolResult::error(&tool_use, err.to_string()))
            }
        });
        let results = join_all(calls).await;
        presentation::validate_batch(&results).map_err(AgentError::machine)?;
        let mut disposition = None;
        for result in results {
            if let Some(next) = record_tool_result(state, ctx, result).await? {
                disposition = Some(next);
            }
        }
        Ok(disposition)
    }

    fn concurrent_batch_ready(&self, state: &AgentState) -> bool {
        !state.pending_tools.is_empty()
            && state.pending_tools.iter().all(|pending| {
                let Some(spec) = pending.spec() else {
                    return false;
                };
                spec.annotations.read_only
                    && !spec.annotations.destructive
                    && !spec.annotations.open_world
                    && !spec.annotations.terminal
                    && !agent_builtin(&pending.tool_use)
                    && !is_terminal_tool(&pending.tool_use, Some(spec))
            })
    }
}

fn commit_answer(state: &mut AgentState, text: String) -> String {
    state.answer = text;
    state.answer.clone()
}

fn check_budget(state: &AgentState) -> Result<(), FinishReason> {
    if state.model_turns >= state.budget.max_model_turns {
        return Err(FinishReason::MaxModelTurns);
    }
    if state.tool_calls >= state.budget.max_tool_calls {
        return Err(FinishReason::MaxToolCalls);
    }
    Ok(())
}

fn close_budget_exhausted_batch(
    state: &mut AgentState,
    tool_uses: &[ToolUse],
) -> Result<(), MachineError> {
    for tool_use in tool_uses {
        let mut result = ToolResult::error(
            tool_use,
            "tool batch was not executed because it exceeds the remaining tool-call budget",
        );
        result.content = json!({
            "error": {
                "code": "budget_exhausted",
                "message": "The entire tool batch was not executed because it exceeds the remaining tool-call budget."
            }
        });
        result.validate().map_err(AgentError::machine)?;
        state.messages.push(AgentMessage::tool_result(result));
    }
    Ok(())
}

fn finish_reason(reason: Option<&StopReason>) -> Result<FinishReason, MachineError> {
    match reason {
        Some(StopReason::EndTurn | StopReason::StopSequence) | None => Ok(FinishReason::Stop),
        Some(StopReason::MaxTokens) => Ok(FinishReason::MaxTokens),
        Some(StopReason::Refusal) => {
            Err(AgentError::Model("assistant message was refused".to_string()).machine())
        }
        Some(StopReason::ToolUse) => Err(AgentError::Model(
            "assistant message stopped for a tool call without returning one".to_string(),
        )
        .machine()),
        Some(StopReason::Other(reason)) => Err(AgentError::Model(format!(
            "assistant message stopped unexpectedly: {reason}"
        ))
        .machine()),
    }
}

fn no_outcome_error(reason: Option<StopReason>) -> AgentError {
    match reason {
        Some(StopReason::EndTurn) => {
            AgentError::Model("model ended without an assistant message or tool calls".to_string())
        }
        Some(reason) => AgentError::Model(format!(
            "model stopped without a tool or natural completion: {reason:?}"
        )),
        None => AgentError::Model("model stopped without a tool or stop reason".to_string()),
    }
}

async fn record_tool_result(
    state: &mut AgentState,
    ctx: &AgentRunContext,
    mut result: ToolResult,
) -> Result<Option<presentation::Presentation>, MachineError> {
    result.validate().map_err(AgentError::machine)?;
    let retained_authorization = result
        .retained
        .as_ref()
        .map(|retained| retained.authorization().clone());
    if let Some(retained) = result.retained.take() {
        retained_result::push(&mut state.retained_results, retained)
            .map_err(AgentError::machine)?;
    }
    let disposition = presentation::take(&mut result);
    let artifacts = std::mem::take(&mut result.artifacts);
    ctx.emit(AgentSignal::ToolResult {
        tool_use_id: result.tool_use_id.clone(),
        name: result.name.clone(),
        content: result.content.clone(),
        is_error: result.is_error,
        retained_authorization,
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
    for artifact in artifacts {
        state.artifacts.push(artifact.clone());
        ctx.emit(AgentSignal::Artifact { artifact }).await?;
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
    Ok(disposition)
}
