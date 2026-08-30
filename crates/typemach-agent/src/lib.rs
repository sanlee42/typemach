use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use typemach::{
    CheckpointSaver, Machine, MachineError, ResumeAction, RunContext, RunEventReceiver, Runner,
    Transition,
};

mod context;
pub use context::estimate_messages;
mod deepseek;
mod deepseek_stream;
pub use deepseek::ConfiguredModel;
mod model_turn;
mod phase;

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
            next_delta_index: 0,
            next_final_delta_index: 0,
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

#[derive(Clone)]
pub struct ModelStream {
    tx: mpsc::UnboundedSender<String>,
    emitted: Arc<std::sync::atomic::AtomicUsize>,
}

impl ModelStream {
    fn new(tx: mpsc::UnboundedSender<String>) -> Self {
        Self {
            tx,
            emitted: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    pub fn channel() -> (Self, mpsc::UnboundedReceiver<String>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self::new(tx), rx)
    }

    pub fn delta(&self, delta: impl Into<String>) -> Result<(), AgentError> {
        self.tx
            .send(delta.into())
            .map_err(|_| AgentError::Model("model delta stream closed".to_string()))?;
        self.emitted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Number of deltas already delivered downstream. Retry logic uses this
    /// to refuse re-sending a response the user has partially seen.
    pub(crate) fn emitted(&self) -> usize {
        self.emitted.load(std::sync::atomic::Ordering::Relaxed)
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
            tools,
            suffix,
            Some(ToolChoice::Auto),
            planning_turn,
        )
        .await?;
        let turn = model_turn::invoke(
            self.model.as_ref(),
            state,
            ctx,
            request,
            model_turn::TextKind::Planning,
        )
        .await?;
        if turn.stop_reason == Some(StopReason::MaxTokens) {
            state.pending_tools.clear();
            return Ok(Transition::Complete(state.output(FinishReason::MaxTokens)));
        }
        let tool_uses = turn
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse(tool) => Some(tool.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if tool_uses.is_empty() {
            return match turn.stop_reason {
                Some(StopReason::EndTurn) => {
                    enter_final_answer(state);
                    Ok(Transition::Next(AgentStep::FinalAnswer))
                }
                Some(reason) => Err(AgentError::Model(format!(
                    "planning stopped without a tool or natural completion: {reason:?}"
                ))
                .machine()),
                None => Err(AgentError::Model(
                    "planning stopped without a tool or stop reason".to_string(),
                )
                .machine()),
            };
        }
        state.pending_tools.extend(tool_uses);
        if !turn.content.is_empty() {
            state.messages.push(AgentMessage::Assistant {
                content: turn.content,
            });
        }
        Ok(Transition::Next(AgentStep::DispatchTools))
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
        let turn = model_turn::invoke(
            self.model.as_ref(),
            state,
            ctx,
            request,
            model_turn::TextKind::FinalAnswer,
        )
        .await?;
        if turn
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse(_)))
        {
            return Err(AgentError::Model(
                "final answer step returned a tool call even though tools were disabled"
                    .to_string(),
            )
            .machine());
        }
        let reason = match turn.stop_reason.as_ref() {
            Some(StopReason::EndTurn | StopReason::StopSequence) | None => FinishReason::Stop,
            Some(StopReason::MaxTokens) => FinishReason::MaxTokens,
            Some(StopReason::Refusal) => {
                return Err(AgentError::Model("final answer was refused".to_string()).machine());
            }
            Some(StopReason::ToolUse) => {
                return Err(AgentError::Model(
                    "final answer stopped for a tool call without returning one".to_string(),
                )
                .machine());
            }
            Some(StopReason::Other(reason)) => {
                return Err(AgentError::Model(format!(
                    "final answer stopped unexpectedly: {reason}"
                ))
                .machine());
            }
        };
        state.messages = messages;
        let content = turn
            .content
            .into_iter()
            .filter(|block| matches!(block, ContentBlock::Text { .. }))
            .collect::<Vec<_>>();
        if !content.is_empty() {
            state.messages.push(AgentMessage::Assistant { content });
        }
        Ok(Transition::Complete(
            state.output_with_answer(reason, state.answer.clone()),
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
        let specs = self
            .tools
            .list_tools(&state.context)
            .await
            .map_err(AgentError::machine)?;
        while let Some(tool_use) = state.pending_tools.pop_front() {
            let spec = specs.iter().find(|spec| spec.name == tool_use.name);
            let built_in_error = if tool_use.name == "ask_user" {
                if let Some(result) = self.consume_human_answer(state, &tool_use, ctx).await? {
                    state.messages.push(AgentMessage::tool_result(result));
                    continue;
                }
                match ask_user_question(&tool_use) {
                    Ok(question) => {
                        state.pending_human = Some(tool_use);
                        return Ok(Transition::Interrupt(question));
                    }
                    Err(err) => Some(ToolResult::error(&tool_use, err.to_string())),
                }
            } else {
                None
            };
            let terminal = is_terminal_tool(&tool_use, spec) && tool_use.name != "emit_artifact";
            if terminal && built_in_error.is_none() {
                let action = terminal_action(&tool_use);
                ctx.emit(AgentSignal::Terminal {
                    action: action.clone(),
                })
                .await?;
                state.terminal = Some(action);
                return Ok(Transition::Complete(state.output(FinishReason::Terminal)));
            }
            state.tool_calls += 1;
            ctx.emit(AgentSignal::ToolStarted {
                tool_use_id: tool_use.id.clone(),
                name: tool_use.name.clone(),
            })
            .await?;
            let result = if let Some(result) = built_in_error {
                result
            } else if tool_use.name == "emit_artifact" {
                match artifact_from_tool(&tool_use) {
                    Ok(artifact) => self.emit_artifact(state, ctx, &tool_use, artifact).await?,
                    Err(err) => ToolResult::error(&tool_use, err.to_string()),
                }
            } else {
                match self.policy.check(&tool_use, spec, &state.context) {
                    PermissionDecision::Allow => self
                        .tools
                        .call_tool(ToolCallRequest {
                            tool_use: tool_use.clone(),
                            context: state.context.clone(),
                        })
                        .await
                        .unwrap_or_else(|err| ToolResult::error(&tool_use, err.to_string())),
                    PermissionDecision::Deny(reason) => ToolResult::error(&tool_use, reason),
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
}

/// A run started over an inherited transcript may find tool calls whose
/// results never arrived (abandoned ask_user, disconnect mid-dispatch).
/// Chat completion providers reject such transcripts outright, so close
/// every dangling call with a synthetic error result.
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
    state.answer.clear();
    state.next_final_delta_index = 0;
    state.pending_tools.clear();
    state.pending_human = None;
    state.human_input = None;
}

fn planning_budget_exhausted(state: &AgentState) -> bool {
    state.model_turns >= state.budget.max_model_turns
        || state.tool_calls >= state.budget.max_tool_calls
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

fn ask_user_question(tool_use: &ToolUse) -> Result<AskUserQuestion, AgentError> {
    let question = tool_use
        .input
        .get("question")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            AgentError::InvalidBuiltInTool("ask_user requires non-empty question".to_string())
        })?
        .to_string();
    Ok(AskUserQuestion {
        tool_use_id: tool_use.id.clone(),
        question,
        fields: tool_use.input.get("fields").cloned().unwrap_or(Value::Null),
    })
}

fn is_terminal_tool(tool_use: &ToolUse, spec: Option<&AgentToolSpec>) -> bool {
    spec.is_some_and(|spec| spec.annotations.terminal)
        || matches!(
            tool_use.name.as_str(),
            "report" | "reject" | "terminal" | "planner.report" | "planner.reject"
        )
}

fn terminal_action(tool_use: &ToolUse) -> TerminalAction {
    TerminalAction {
        tool_use_id: tool_use.id.clone(),
        name: tool_use.name.clone(),
        input: tool_use.input.clone(),
    }
}

fn artifact_from_tool(tool_use: &ToolUse) -> Result<Artifact, AgentError> {
    let title = required_string(&tool_use.input, "title")?;
    let content = required_string(&tool_use.input, "content")?;
    let kind = required_string(&tool_use.input, "type")?;
    if !matches!(kind.as_str(), "markdown" | "table") {
        return Err(AgentError::InvalidBuiltInTool(
            "type must be markdown or table".to_string(),
        ));
    }
    Ok(Artifact {
        tool_use_id: tool_use.id.clone(),
        title,
        kind,
        content,
        source: optional_source(&tool_use.input)?,
        window: optional_string(&tool_use.input, "window"),
        updated_at: optional_string(&tool_use.input, "updated_at"),
    })
}

fn optional_string(input: &Value, name: &str) -> Option<String> {
    input
        .get(name)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn optional_source(input: &Value) -> Result<Option<String>, AgentError> {
    let Some(source) = input.get("source") else {
        return Ok(None);
    };
    source
        .as_str()
        .map(str::trim)
        .filter(|source| !source.is_empty())
        .map(str::to_owned)
        .map(Some)
        .ok_or_else(|| {
            AgentError::InvalidBuiltInTool("source must be a non-empty string".to_string())
        })
}

fn required_string(input: &Value, name: &str) -> Result<String, AgentError> {
    input
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| AgentError::InvalidBuiltInTool(format!("missing non-empty {name}")))
}
