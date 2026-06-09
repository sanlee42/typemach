use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use typemach::{
    CheckpointSaver, Machine, MachineError, ResumeAction, RunContext, RunEventReceiver, Runner,
    Transition,
};

pub use typemach as core;

pub type AgentRunContext =
    RunContext<AgentRunInput, AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>;
pub type AgentRunner<M, T, P, S> = Runner<AgentMachine<M, T, P>, S>;
pub type AgentEventReceiver =
    RunEventReceiver<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStep {
    PrepareTurn,
    ModelStep,
    DispatchTools,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum AgentMessage {
    User { content: Vec<ContentBlock> },
    Assistant { content: Vec<ContentBlock> },
}

impl AgentMessage {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self::User {
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn tool_result(result: ToolResult) -> Self {
        Self::User {
            content: vec![ContentBlock::ToolResult(result)],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    ToolUse(ToolUse),
    ToolResult(ToolResult),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub input: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub name: String,
    #[serde(default)]
    pub content: Value,
    #[serde(default)]
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(tool_use: &ToolUse, content: Value) -> Self {
        Self {
            tool_use_id: tool_use.id.clone(),
            name: tool_use.name.clone(),
            content,
            is_error: false,
        }
    }

    pub fn error(tool_use: &ToolUse, message: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use.id.clone(),
            name: tool_use.name.clone(),
            content: json!({ "error": message.into() }),
            is_error: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentToolSpec {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub input_schema: Value,
    #[serde(default)]
    pub output_schema: Value,
    #[serde(default)]
    pub annotations: ToolAnnotations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolAnnotations {
    pub read_only: bool,
    pub destructive: bool,
    pub open_world: bool,
}

impl Default for ToolAnnotations {
    fn default() -> Self {
        Self {
            read_only: true,
            destructive: false,
            open_world: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<AgentToolSpec>,
    #[serde(default)]
    pub context: Value,
    pub turn: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ModelResponse {
    #[serde(default)]
    pub deltas: Vec<String>,
    #[serde(default)]
    pub tool_uses: Vec<ToolUse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRequest {
    pub tool_use: ToolUse,
    #[serde(default)]
    pub context: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AskUserQuestion {
    pub tool_use_id: String,
    pub question: String,
    #[serde(default)]
    pub fields: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanInputAnswer {
    pub tool_use_id: String,
    pub answer: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub tool_use_id: String,
    pub title: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentSignal {
    AssistantDelta {
        delta: String,
        index: usize,
    },
    ToolStarted {
        tool_use_id: String,
        name: String,
    },
    ToolCompleted {
        tool_use_id: String,
        name: String,
        is_error: bool,
    },
    ToolResult {
        tool_use_id: String,
        name: String,
        content: Value,
        is_error: bool,
    },
    Artifact {
        artifact: Artifact,
    },
    Usage {
        usage: Usage,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentBudget {
    pub max_model_turns: u32,
    pub max_tool_calls: u32,
}

impl Default for AgentBudget {
    fn default() -> Self {
        Self {
            max_model_turns: 16,
            max_tool_calls: 32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRunInput {
    pub messages: Vec<AgentMessage>,
    #[serde(default)]
    pub context: Value,
    #[serde(default)]
    pub budget: AgentBudget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_input: Option<HumanInputAnswer>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRunOutput {
    pub messages: Vec<AgentMessage>,
    pub answer: String,
    pub finish_reason: FinishReason,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    MaxModelTurns,
    MaxToolCalls,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub messages: Vec<AgentMessage>,
    pub context: Value,
    pub budget: AgentBudget,
    pub model_turns: u32,
    pub tool_calls: u32,
    pub next_delta_index: usize,
    pub pending_tools: VecDeque<ToolUse>,
    pub pending_human: Option<ToolUse>,
    pub human_input: Option<HumanInputAnswer>,
    pub answer: String,
    pub usage: Usage,
    pub artifacts: Vec<Artifact>,
}

impl AgentState {
    fn fresh(input: &AgentRunInput, previous: Option<&Self>) -> Self {
        let mut messages = previous
            .map(|state| state.messages.clone())
            .unwrap_or_default();
        messages.extend(input.messages.clone());
        Self {
            messages,
            context: input.context.clone(),
            budget: input.budget.clone(),
            model_turns: 0,
            tool_calls: 0,
            next_delta_index: 0,
            pending_tools: VecDeque::new(),
            pending_human: None,
            human_input: input.human_input.clone(),
            answer: String::new(),
            usage: Usage::default(),
            artifacts: Vec::new(),
        }
    }

    fn output(&self, finish_reason: FinishReason) -> AgentRunOutput {
        AgentRunOutput {
            messages: self.messages.clone(),
            answer: self.answer.clone(),
            finish_reason,
            usage: self.usage.clone(),
            artifacts: self.artifacts.clone(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("model failed: {0}")]
    Model(String),
    #[error("tool failed: {0}")]
    Tool(String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("invalid built-in tool arguments: {0}")]
    InvalidBuiltInTool(String),
}

impl AgentError {
    fn machine(self) -> MachineError {
        MachineError::transition(self)
    }
}

#[async_trait]
pub trait AgentModel: Send + Sync {
    async fn next_step(&self, request: ModelRequest) -> Result<ModelResponse, AgentError>;
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
}

impl<M, T, P> AgentMachine<M, T, P> {
    pub fn new(model: M, tools: T, policy: P) -> Self {
        Self {
            model: Arc::new(model),
            tools: Arc::new(tools),
            policy: Arc::new(policy),
        }
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
        Ok(AgentState::fresh(input, previous))
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
        let response = self
            .model
            .next_step(ModelRequest {
                messages: state.messages.clone(),
                tools,
                context: state.context.clone(),
                turn: state.model_turns,
            })
            .await
            .map_err(AgentError::machine)?;

        let mut assistant_content = Vec::new();
        for delta in response.deltas {
            state.answer.push_str(&delta);
            ctx.emit(AgentSignal::AssistantDelta {
                delta: delta.clone(),
                index: state.next_delta_index,
            })
            .await?;
            state.next_delta_index += 1;
            assistant_content.push(ContentBlock::Text { text: delta });
        }
        if let Some(usage) = response.usage {
            state.usage.input_tokens += usage.input_tokens;
            state.usage.output_tokens += usage.output_tokens;
            ctx.emit(AgentSignal::Usage { usage }).await?;
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
            if tool_use.name == "ask_user" {
                if let Some(result) = self.consume_human_answer(state, &tool_use, ctx).await? {
                    state.messages.push(AgentMessage::tool_result(result));
                    continue;
                }
                let question = ask_user_question(&tool_use)?;
                state.pending_human = Some(tool_use);
                return Ok(Transition::Interrupt(question));
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
                let spec = specs.iter().find(|spec| spec.name == tool_use.name);
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
            ctx.emit(AgentSignal::ToolCompleted {
                tool_use_id: result.tool_use_id.clone(),
                name: result.name.clone(),
                is_error: result.is_error,
            })
            .await?;
            state.messages.push(AgentMessage::tool_result(result));
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use typemach::{
        MemorySaver, RunCommand, RunId, RunRequest, RunStreamEvent, RuntimeLimits, SessionId,
        StreamConfig, ThreadId,
    };

    #[derive(Clone, Default)]
    struct ScriptedModel {
        responses: Arc<Mutex<VecDeque<ModelResponse>>>,
        requests: Arc<Mutex<Vec<ModelRequest>>>,
    }

    impl ScriptedModel {
        fn new(responses: impl IntoIterator<Item = ModelResponse>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses.into_iter().collect())),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn requests(&self) -> Vec<ModelRequest> {
            self.requests.lock().expect("requests lock").clone()
        }
    }

    #[async_trait]
    impl AgentModel for ScriptedModel {
        async fn next_step(&self, request: ModelRequest) -> Result<ModelResponse, AgentError> {
            self.requests.lock().expect("requests lock").push(request);
            self.responses
                .lock()
                .expect("responses lock")
                .pop_front()
                .ok_or_else(|| AgentError::Model("script exhausted".to_string()))
        }
    }

    #[derive(Default)]
    struct FakeTools;

    #[async_trait]
    impl ToolRegistry for FakeTools {
        async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
            Ok(vec![
                AgentToolSpec {
                    name: "metric_point".to_string(),
                    description: "read metric point".to_string(),
                    input_schema: json!({ "type": "object" }),
                    output_schema: Value::Null,
                    annotations: ToolAnnotations::default(),
                },
                AgentToolSpec {
                    name: "ask_user".to_string(),
                    description: "ask user".to_string(),
                    input_schema: json!({ "type": "object" }),
                    output_schema: Value::Null,
                    annotations: ToolAnnotations::default(),
                },
                AgentToolSpec {
                    name: "emit_artifact".to_string(),
                    description: "emit artifact".to_string(),
                    input_schema: json!({ "type": "object" }),
                    output_schema: Value::Null,
                    annotations: ToolAnnotations::default(),
                },
            ])
        }

        async fn call_tool(&self, request: ToolCallRequest) -> Result<ToolResult, AgentError> {
            Ok(ToolResult::ok(
                &request.tool_use,
                json!({ "value": 42, "ds": "2026-06-08" }),
            ))
        }
    }

    fn request(input: AgentRunInput) -> RunRequest<AgentRunInput> {
        RunRequest {
            run_id: RunId::from("run-1"),
            session_id: SessionId::from("session-1"),
            thread_id: ThreadId::from("thread-1"),
            command: RunCommand::Start,
            input,
            snapshot: None,
            runtime_limits: RuntimeLimits::new(32),
        }
    }

    async fn collect(
        mut rx: AgentEventReceiver,
    ) -> Vec<RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>> {
        let mut events = Vec::new();
        while let Some(event) = rx.next_event().await {
            let terminal = matches!(
                event,
                RunStreamEvent::Completed { .. }
                    | RunStreamEvent::Interrupted { .. }
                    | RunStreamEvent::Failed { .. }
                    | RunStreamEvent::Cancelled
            );
            events.push(event);
            if terminal {
                break;
            }
        }
        events
    }

    #[tokio::test]
    async fn model_tool_result_model_loop_completes() {
        let model = ScriptedModel::new([
            ModelResponse {
                tool_uses: vec![ToolUse {
                    id: "tool-1".to_string(),
                    name: "metric_point".to_string(),
                    input: json!({ "metric_id": "paid_order_count", "ds": "2026-06-08" }),
                }],
                ..ModelResponse::default()
            },
            ModelResponse {
                deltas: vec!["订单量是 42。".to_string()],
                final_text: Some(String::new()),
                ..ModelResponse::default()
            },
        ]);
        let runner = build_agent_runner(
            MemorySaver::default(),
            model.clone(),
            FakeTools,
            AllowAllTools,
        );
        let rx = runner.stream(
            request(AgentRunInput {
                messages: vec![AgentMessage::user_text("昨天订单量")],
                context: Value::Null,
                budget: AgentBudget::default(),
                human_input: None,
            }),
            StreamConfig::default(),
        );
        let events = collect(rx).await;
        assert!(events.iter().any(|event| matches!(
          event,
          RunStreamEvent::Signal {
            signal: AgentSignal::ToolStarted { name, .. },
          } if name == "metric_point"
        )));
        assert!(events.iter().any(|event| matches!(
          event,
          RunStreamEvent::Signal {
            signal: AgentSignal::ToolResult { name, content, .. },
          } if name == "metric_point" && content["value"] == 42
        )));
        let completed = events
            .iter()
            .find_map(|event| match event {
                RunStreamEvent::Completed { output, .. } => Some(output),
                _ => None,
            })
            .expect("completed output");
        assert_eq!(completed.answer, "订单量是 42。");
        assert_eq!(completed.finish_reason, FinishReason::Stop);
    }

    #[tokio::test]
    async fn final_text_without_stream_deltas_is_emitted_and_persisted() {
        let model = ScriptedModel::new([ModelResponse {
            final_text: Some("订单量是 42。".to_string()),
            ..ModelResponse::default()
        }]);
        let runner = build_agent_runner(MemorySaver::default(), model, FakeTools, AllowAllTools);
        let events = collect(runner.stream(
            request(AgentRunInput {
                messages: vec![AgentMessage::user_text("昨天订单量")],
                context: Value::Null,
                budget: AgentBudget::default(),
                human_input: None,
            }),
            StreamConfig::default(),
        ))
        .await;
        assert!(events.iter().any(|event| matches!(
          event,
          RunStreamEvent::Signal {
            signal: AgentSignal::AssistantDelta { delta, .. },
          } if delta == "订单量是 42。"
        )));
        let completed = events
            .iter()
            .find_map(|event| match event {
                RunStreamEvent::Completed { output, .. } => Some(output),
                _ => None,
            })
            .expect("completed output");
        assert_eq!(completed.answer, "订单量是 42。");
        assert!(matches!(
            completed.messages.last(),
            Some(AgentMessage::Assistant { content })
                if content == &vec![ContentBlock::Text {
                    text: "订单量是 42。".to_string()
                }]
        ));
    }

    #[tokio::test]
    async fn ask_user_interrupts_and_resume_continues() {
        let model = ScriptedModel::new([
            ModelResponse {
                tool_uses: vec![ToolUse {
                    id: "ask-1".to_string(),
                    name: "ask_user".to_string(),
                    input: json!({ "question": "看哪个日期？", "fields": { "type": "choice" } }),
                }],
                ..ModelResponse::default()
            },
            ModelResponse {
                deltas: vec!["按 2026-06-08 查看，订单量是 42。".to_string()],
                final_text: Some(String::new()),
                ..ModelResponse::default()
            },
        ]);
        let runner = build_agent_runner(
            MemorySaver::default(),
            model.clone(),
            FakeTools,
            AllowAllTools,
        );
        let mut first = runner.stream(
            request(AgentRunInput {
                messages: vec![AgentMessage::user_text("订单量")],
                context: Value::Null,
                budget: AgentBudget::default(),
                human_input: None,
            }),
            StreamConfig::default(),
        );
        let interrupted = loop {
            match first.next_event().await.expect("event") {
                RunStreamEvent::Interrupted { interrupt, .. } => break interrupt,
                RunStreamEvent::Failed { error } => panic!("failed: {error}"),
                _ => {}
            }
        };
        assert_eq!(interrupted.question, "看哪个日期？");

        let resume = RunRequest {
            command: RunCommand::Resume,
            input: AgentRunInput {
                messages: Vec::new(),
                context: Value::Null,
                budget: AgentBudget::default(),
                human_input: Some(HumanInputAnswer {
                    tool_use_id: "ask-1".to_string(),
                    answer: "2026-06-08".to_string(),
                }),
            },
            ..request(AgentRunInput {
                messages: Vec::new(),
                context: Value::Null,
                budget: AgentBudget::default(),
                human_input: None,
            })
        };
        let events = collect(runner.stream(resume, StreamConfig::default())).await;
        let completed = events
            .iter()
            .find_map(|event| match event {
                RunStreamEvent::Completed { output, .. } => Some(output),
                _ => None,
            })
            .expect("completed output");
        assert_eq!(completed.answer, "按 2026-06-08 查看，订单量是 42。");
        let requests = model.requests();
        let second = requests.get(1).expect("second model request");
        assert!(second.messages.iter().any(|message| matches!(
            message,
            AgentMessage::User { content }
                if content.iter().any(|block| matches!(
                    block,
                    ContentBlock::ToolResult(result)
                        if result.tool_use_id == "ask-1"
                            && result.content == json!({ "answer": "2026-06-08" })
                ))
        )));
    }

    #[tokio::test]
    async fn emit_artifact_is_signalled_and_not_required_in_answer() {
        let model = ScriptedModel::new([
            ModelResponse {
                tool_uses: vec![ToolUse {
                    id: "artifact-1".to_string(),
                    name: "emit_artifact".to_string(),
                    input: json!({
                      "title": "经营复盘",
                      "type": "markdown",
                      "content": "# 经营复盘"
                    }),
                }],
                ..ModelResponse::default()
            },
            ModelResponse {
                deltas: vec!["复盘已生成。".to_string()],
                final_text: Some(String::new()),
                ..ModelResponse::default()
            },
        ]);
        let runner = build_agent_runner(MemorySaver::default(), model, FakeTools, AllowAllTools);
        let events = collect(runner.stream(
            request(AgentRunInput {
                messages: vec![AgentMessage::user_text("生成复盘")],
                context: Value::Null,
                budget: AgentBudget::default(),
                human_input: None,
            }),
            StreamConfig::default(),
        ))
        .await;
        assert!(events.iter().any(|event| matches!(
          event,
          RunStreamEvent::Signal {
            signal: AgentSignal::Artifact { artifact },
          } if artifact.title == "经营复盘"
        )));
        let completed = events
            .iter()
            .find_map(|event| match event {
                RunStreamEvent::Completed { output, .. } => Some(output),
                _ => None,
            })
            .expect("completed output");
        assert_eq!(completed.answer, "复盘已生成。");
        assert_eq!(completed.artifacts.len(), 1);
    }
}
