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
            model_turns: 0,
            tool_calls: 0,
            next_delta_index: 0,
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
        AgentRunOutput {
            messages: self.messages.clone(),
            answer: self.answer.clone(),
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
}

impl ModelStream {
    fn new(tx: mpsc::UnboundedSender<String>) -> Self {
        Self { tx }
    }

    pub fn channel() -> (Self, mpsc::UnboundedReceiver<String>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self::new(tx), rx)
    }

    pub fn delta(&self, delta: impl Into<String>) -> Result<(), AgentError> {
        self.tx
            .send(delta.into())
            .map_err(|_| AgentError::Model("model delta stream closed".to_string()))
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
        if state.model_turns >= state.budget.max_model_turns {
            return Ok(Transition::Complete(
                state.output(FinishReason::MaxModelTurns),
            ));
        }
        state.model_turns += 1;
        let tools = self
            .tools
            .list_tools(&state.context)
            .await
            .map_err(AgentError::machine)?;
        let prompt_window = context::prompt_window(&state.messages, &state.context_policy)
            .map_err(AgentError::machine)?;
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
        let request = ModelRequest {
            messages: prompt_window.messages,
            tools,
            context: state.context.clone(),
            turn: state.model_turns,
        };
        let (delta_tx, mut delta_rx) = mpsc::unbounded_channel();
        let response = self.model.next_step(request, ModelStream::new(delta_tx));
        tokio::pin!(response);

        let mut assistant_content = Vec::new();
        let response = loop {
            tokio::select! {
                maybe_delta = delta_rx.recv() => {
                    if let Some(delta) = maybe_delta {
                        append_delta(state, ctx, &mut assistant_content, delta).await?;
                    }
                }
                response = &mut response => {
                    break response.map_err(AgentError::machine)?;
                }
            }
        };
        while let Ok(delta) = delta_rx.try_recv() {
            append_delta(state, ctx, &mut assistant_content, delta).await?;
        }
        for delta in response.deltas {
            append_delta(state, ctx, &mut assistant_content, delta).await?;
        }
        if let Some(usage) = response.usage {
            state.usage.input_tokens += usage.input_tokens;
            state.usage.output_tokens += usage.output_tokens;
            ctx.emit(AgentSignal::Usage { usage }).await?;
        }
        for block in response.content {
            record_assistant_block(state, ctx, &mut assistant_content, block).await?;
        }
        for tool_use in response.tool_uses {
            assistant_content.push(ContentBlock::ToolUse(tool_use.clone()));
            state.pending_tools.push_back(tool_use);
        }
        if let Some(final_text) = response.final_text {
            if !final_text.is_empty() && !assistant_content.iter().any(is_text_block) {
                state.answer.push_str(&final_text);
                ctx.emit(AgentSignal::AssistantDelta {
                    delta: final_text.clone(),
                    index: state.next_delta_index,
                })
                .await?;
                state.next_delta_index += 1;
                assistant_content.push(ContentBlock::Text { text: final_text });
            }
            if !assistant_content.is_empty() {
                state.messages.push(AgentMessage::Assistant {
                    content: assistant_content,
                });
            }
            return Ok(Transition::Complete(state.output(FinishReason::Stop)));
        }
        if !assistant_content.is_empty() {
            state.messages.push(AgentMessage::Assistant {
                content: assistant_content,
            });
        }
        if state.pending_tools.is_empty() {
            return Ok(Transition::Complete(state.output(FinishReason::Stop)));
        }
        Ok(Transition::Next(AgentStep::DispatchTools))
    }

    async fn dispatch_tools(
        &self,
        state: &mut AgentState,
        ctx: &AgentRunContext,
    ) -> Result<Transition<AgentStep, AskUserQuestion, AgentRunOutput>, MachineError> {
        let specs = self
            .tools
            .list_tools(&state.context)
            .await
            .map_err(AgentError::machine)?;
        while let Some(tool_use) = state.pending_tools.pop_front() {
            if state.tool_calls >= state.budget.max_tool_calls {
                return Ok(Transition::Complete(
                    state.output(FinishReason::MaxToolCalls),
                ));
            }
            let spec = specs.iter().find(|spec| spec.name == tool_use.name);
            if tool_use.name == "ask_user" {
                if let Some(result) = self.consume_human_answer(state, &tool_use, ctx).await? {
                    state.messages.push(AgentMessage::tool_result(result));
                    continue;
                }
                let question = ask_user_question(&tool_use)?;
                state.pending_human = Some(tool_use);
                return Ok(Transition::Interrupt(question));
            }
            if is_terminal_tool(&tool_use, spec) {
                let action = terminal_action(&tool_use);
                if state.answer.is_empty()
                    && let Some(message) = terminal_message(&tool_use)
                {
                    append_delta(state, ctx, &mut Vec::new(), message).await?;
                }
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
            let result = if tool_use.name == "emit_artifact" {
                self.emit_artifact(state, ctx, &tool_use).await?
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
        }
        Ok(Transition::Next(AgentStep::ModelStep))
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
    ) -> Result<ToolResult, MachineError> {
        let artifact = artifact_from_tool(tool_use)?;
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

async fn append_delta(
    state: &mut AgentState,
    ctx: &AgentRunContext,
    assistant_content: &mut Vec<ContentBlock>,
    delta: String,
) -> Result<(), MachineError> {
    if delta.is_empty() {
        return Ok(());
    }
    state.answer.push_str(&delta);
    ctx.emit(AgentSignal::AssistantDelta {
        delta: delta.clone(),
        index: state.next_delta_index,
    })
    .await?;
    state.next_delta_index += 1;
    assistant_content.push(ContentBlock::Text { text: delta });
    Ok(())
}

async fn record_assistant_block(
    state: &mut AgentState,
    ctx: &AgentRunContext,
    assistant_content: &mut Vec<ContentBlock>,
    block: ContentBlock,
) -> Result<(), MachineError> {
    match block {
        ContentBlock::Text { text } => append_delta(state, ctx, assistant_content, text).await,
        ContentBlock::ToolUse(tool_use) => {
            assistant_content.push(ContentBlock::ToolUse(tool_use.clone()));
            state.pending_tools.push_back(tool_use);
            Ok(())
        }
        other => {
            assistant_content.push(other);
            Ok(())
        }
    }
}

fn ask_user_question(tool_use: &ToolUse) -> Result<AskUserQuestion, MachineError> {
    let question = tool_use
        .input
        .get("question")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            AgentError::InvalidBuiltInTool("ask_user requires non-empty question".to_string())
                .machine()
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

fn terminal_message(tool_use: &ToolUse) -> Option<String> {
    ["message", "reason", "answer"]
        .iter()
        .find_map(|key| tool_use.input.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn artifact_from_tool(tool_use: &ToolUse) -> Result<Artifact, MachineError> {
    let title = required_string(&tool_use.input, "title")?;
    let content = required_string(&tool_use.input, "content")?;
    let kind = tool_use
        .input
        .get("type")
        .or_else(|| tool_use.input.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or("markdown")
        .to_string();
    Ok(Artifact {
        tool_use_id: tool_use.id.clone(),
        title,
        kind,
        content,
    })
}

fn is_text_block(block: &ContentBlock) -> bool {
    matches!(block, ContentBlock::Text { .. })
}

fn required_string(input: &Value, name: &str) -> Result<String, MachineError> {
    input
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            AgentError::InvalidBuiltInTool(format!("missing non-empty {name}")).machine()
        })
}
