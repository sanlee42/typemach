use std::collections::{BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PendingToolCall;
use crate::presentation::ToolDisposition;
use crate::{AssistantMessageId, AssistantMessageItem, AssistantMessagePhase};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentStep {
    PrepareTurn,
    /// Automatic tool selection or the final assistant answer.
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
    Text {
        text: String,
    },
    AssistantMessage(AssistantMessageItem),
    ConversationDigest(ConversationDigest),
    Thinking {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ToolUse(ToolUse),
    ToolResult(ToolResult),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub input: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub name: String,
    #[serde(default)]
    pub content: Value,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "ToolDisposition::is_continue")]
    pub disposition: ToolDisposition,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

impl ToolResult {
    pub fn ok(tool_use: &ToolUse, content: Value) -> Self {
        Self {
            tool_use_id: tool_use.id.clone(),
            name: tool_use.name.clone(),
            content,
            is_error: false,
            disposition: ToolDisposition::Continue,
            artifacts: Vec::new(),
            raw: None,
        }
    }

    pub fn error(tool_use: &ToolUse, message: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use.id.clone(),
            name: tool_use.name.clone(),
            content: json!({ "error": message.into() }),
            is_error: true,
            disposition: ToolDisposition::Continue,
            artifacts: Vec::new(),
            raw: None,
        }
    }

    pub fn present(mut self, receipt: impl Into<String>) -> Result<Self, AgentError> {
        self.disposition = ToolDisposition::Present {
            receipt: receipt.into(),
        };
        self.validate()?;
        Ok(self)
    }

    pub fn with_artifacts(mut self, artifacts: Vec<Artifact>) -> Result<Self, AgentError> {
        self.artifacts = artifacts;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn validate(&self) -> Result<(), AgentError> {
        if let ToolDisposition::Present { receipt } = &self.disposition {
            if self.is_error {
                return Err(AgentError::InvalidToolResult(
                    "an error result cannot present the final answer".to_string(),
                ));
            }
            if receipt.trim().is_empty() {
                return Err(AgentError::InvalidToolResult(
                    "present requires a non-empty receipt".to_string(),
                ));
            }
        }
        for (index, artifact) in self.artifacts.iter().enumerate() {
            artifact.validate_for(&self.tool_use_id).map_err(|reason| {
                AgentError::InvalidToolResult(format!("artifact {index}: {reason}"))
            })?;
        }
        Ok(())
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
    pub metadata: Value,
    #[serde(default)]
    pub annotations: ToolAnnotations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolAnnotations {
    #[serde(default = "default_true")]
    pub read_only: bool,
    #[serde(default)]
    pub destructive: bool,
    #[serde(default)]
    pub open_world: bool,
    #[serde(default)]
    pub terminal: bool,
}

impl Default for ToolAnnotations {
    fn default() -> Self {
        Self {
            read_only: true,
            destructive: false,
            open_world: false,
            terminal: false,
        }
    }
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<AgentToolSpec>,
    #[serde(default)]
    pub context: Value,
    pub turn: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_suffix: Option<String>,
    /// Per-call policy. Agent lifecycle phases override the configured
    /// transport default without changing the provider/model settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ModelResponse {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assistant_messages: Vec<AssistantMessageItem>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolUse>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Refusal,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SpeedProfile {
    #[default]
    Flash,
    FlashWithAutoThinking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    #[default]
    Low,
    Medium,
    High,
}

/// How the model is allowed to pick tools each turn. `Required` forces a tool
/// call (no bare-prose turns) which protocol-style agents that must always act
/// via tools rely on to avoid stalls on smaller models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Required,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingConfig {
    pub enabled: bool,
    pub reasoning_effort: ReasoningEffort,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            reasoning_effort: ReasoningEffort::Low,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextPolicy {
    pub max_input_tokens: u64,
    pub compact_at_tokens: u64,
    /// Window size in messages (not conversation turns).
    #[serde(alias = "recent_turns")]
    pub recent_messages: usize,
    pub max_tool_result_bytes: usize,
    pub background_digest: bool,
}

impl Default for ContextPolicy {
    fn default() -> Self {
        Self {
            max_input_tokens: 128_000,
            compact_at_tokens: 96_000,
            recent_messages: 8,
            max_tool_result_bytes: 24 * 1024,
            background_digest: false,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(skip_serializing)]
    pub api_key: String,
    pub model: String,
    #[serde(default = "default_deepseek_base_url")]
    pub base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default)]
    pub speed_profile: SpeedProfile,
    #[serde(default)]
    pub thinking: ThinkingConfig,
    #[serde(default)]
    pub context_policy: ContextPolicy,
    #[serde(default = "default_true")]
    pub stream: bool,
    /// Total request timeout for non-streaming calls; idle gap between
    /// chunks for streaming calls (a healthy long stream never trips it).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Extra attempts after a transient failure (429/5xx/connect/timeout)
    /// when no answer delta has been streamed yet.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Optional tool-choice policy sent to the model (e.g. `Required` to force a
    /// tool call every turn). `None` leaves it to the provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

impl std::fmt::Debug for AgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentConfig")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("system", &self.system)
            .field("speed_profile", &self.speed_profile)
            .field("thinking", &self.thinking)
            .field("context_policy", &self.context_policy)
            .field("stream", &self.stream)
            .field("request_timeout_secs", &self.request_timeout_secs)
            .field("max_tokens", &self.max_tokens)
            .field("max_retries", &self.max_retries)
            .field("tool_choice", &self.tool_choice)
            .finish()
    }
}

impl AgentConfig {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: default_deepseek_base_url(),
            system: None,
            speed_profile: SpeedProfile::default(),
            thinking: ThinkingConfig::default(),
            context_policy: ContextPolicy::default(),
            stream: true,
            request_timeout_secs: default_request_timeout_secs(),
            max_tokens: None,
            max_retries: default_max_retries(),
            tool_choice: None,
        }
    }
}

fn default_deepseek_base_url() -> String {
    "https://api.deepseek.com".to_string()
}

const fn default_request_timeout_secs() -> u64 {
    120
}

const fn default_max_retries() -> u32 {
    2
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationDigest {
    pub omitted_message_count: usize,
    pub user_goals: Vec<String>,
    pub confirmed_facts: Vec<String>,
    pub decisions: Vec<String>,
    pub open_questions: Vec<String>,
    pub evidence_refs: Vec<String>,
    pub risks: Vec<String>,
    pub current_state: String,
}

impl ConversationDigest {
    pub fn compacted_window(omitted_message_count: usize) -> Self {
        Self {
            omitted_message_count,
            user_goals: Vec::new(),
            confirmed_facts: Vec::new(),
            decisions: Vec::new(),
            open_questions: Vec::new(),
            evidence_refs: Vec::new(),
            risks: Vec::new(),
            current_state: "Older closed conversation turns were omitted from the active prompt window. Use recent turns and tool evidence refs for the current answer.".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptEstimate {
    pub message_count: usize,
    pub estimated_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptArchive {
    pub message_count: usize,
    pub byte_count: usize,
    pub estimated_tokens: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCompaction {
    pub before: PromptEstimate,
    pub after: PromptEstimate,
    pub archive: TranscriptArchive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultArchive {
    pub tool_use_id: String,
    pub name: String,
    pub byte_count: usize,
    pub sha256: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

impl Artifact {
    fn validate_for(&self, tool_use_id: &str) -> Result<(), String> {
        if self.tool_use_id != tool_use_id {
            return Err(format!(
                "tool_use_id {} does not match {tool_use_id}",
                self.tool_use_id
            ));
        }
        if self.title.trim().is_empty() {
            return Err("missing non-empty title".to_string());
        }
        if self.content.trim().is_empty() {
            return Err("missing non-empty content".to_string());
        }
        if !matches!(self.kind.as_str(), "markdown" | "table") {
            return Err("type must be markdown or table".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerminalAction {
    pub tool_use_id: String,
    pub name: String,
    #[serde(default)]
    pub input: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentSignal {
    AssistantMessageStarted {
        message_id: AssistantMessageId,
        phase: AssistantMessagePhase,
    },
    AssistantMessageDelta {
        message_id: AssistantMessageId,
        phase: AssistantMessagePhase,
        delta: String,
        index: usize,
    },
    AssistantMessageDone {
        message_id: AssistantMessageId,
        phase: AssistantMessagePhase,
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
    Terminal {
        action: TerminalAction,
    },
    Usage {
        usage: Usage,
    },
    ToolResultArchived {
        archive: ToolResultArchive,
    },
    DigestUpdated {
        digest: ConversationDigest,
    },
    ContextCompacted {
        compaction: ContextCompaction,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentBudget {
    /// Maximum model calls, including the terminal answer call.
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
    /// Per-run system prompt addition (e.g. shop scope), appended after the
    /// model's static system prompt. Re-supplied on every Start and Resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_suffix: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRunOutput {
    pub messages: Vec<AgentMessage>,
    pub answer: String,
    pub finish_reason: FinishReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<TerminalAction>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<ConversationDigest>,
    #[serde(default)]
    pub tool_result_archives: Vec<ToolResultArchive>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Terminal,
    MaxModelTurns,
    MaxToolCalls,
    /// The model hit its output token limit; partial text is not committed.
    MaxTokens,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub messages: Vec<AgentMessage>,
    pub context: Value,
    pub budget: AgentBudget,
    #[serde(default)]
    pub context_policy: ContextPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_suffix: Option<String>,
    pub model_turns: u32,
    pub tool_calls: u32,
    /// Loaded deferred names for this run; revalidated against the current context.
    #[serde(default)]
    pub loaded_deferred_tools: BTreeSet<crate::DeferredToolName>,
    #[serde(default)]
    pub pending_tools: VecDeque<PendingToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_human: Option<PendingToolCall>,
    pub human_input: Option<HumanInputAnswer>,
    pub answer: String,
    pub usage: Usage,
    pub artifacts: Vec<Artifact>,
    pub terminal: Option<TerminalAction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<ConversationDigest>,
    #[serde(default)]
    pub tool_result_archives: Vec<ToolResultArchive>,
}

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("registry tool uses reserved name: {0}")]
    ReservedToolName(String),
    #[error("registry catalogs contain duplicate tool name: {0}")]
    DuplicateToolName(String),
    #[error("invalid agent configuration: {0}")]
    Config(String),
    #[error("model failed: {0}")]
    Model(String),
    #[error("tool failed: {0}")]
    Tool(String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("invalid built-in tool arguments: {0}")]
    InvalidBuiltInTool(String),
    #[error("invalid tool result: {0}")]
    InvalidToolResult(String),
    #[error("agent stopped before completing an answer: {0:?}")]
    Incomplete(FinishReason),
}

#[cfg(test)]
mod type_contracts {
    use super::*;

    #[test]
    fn assistant_lifecycle_signals_include_phase() {
        let started = AgentSignal::AssistantMessageStarted {
            message_id: AssistantMessageId::new("run-1:turn-1"),
            phase: AssistantMessagePhase::Commentary,
        };
        let signal = AgentSignal::AssistantMessageDelta {
            message_id: AssistantMessageId::new("run-1:turn-1"),
            phase: AssistantMessagePhase::Commentary,
            delta: "Checking data.".to_string(),
            index: 0,
        };

        assert_eq!(
            serde_json::to_value(started).expect("serialize signal"),
            json!({
                "type": "assistant_message_started",
                "message_id": "run-1:turn-1",
                "phase": "commentary"
            })
        );

        assert_eq!(
            serde_json::to_value(signal).expect("serialize signal"),
            json!({
                "type": "assistant_message_delta",
                "message_id": "run-1:turn-1",
                "phase": "commentary",
                "delta": "Checking data.",
                "index": 0
            })
        );
    }

    #[test]
    fn old_state_defaults_pending_specs_and_step_remains_planning() {
        let state: AgentState = serde_json::from_value(json!({
            "messages": [],
            "context": null,
            "budget": { "max_model_turns": 4, "max_tool_calls": 8 },
            "model_turns": 1,
            "tool_calls": 0,
            "pending_tools": [{
                "id": "tool-1",
                "name": "metric_point",
                "input": {}
            }],
            "pending_human": null,
            "human_input": null,
            "answer": "",
            "usage": { "input_tokens": 0, "output_tokens": 0 },
            "artifacts": [],
            "terminal": null
        }))
        .expect("deserialize legacy state");
        let step: AgentStep =
            serde_json::from_value(json!("model_step")).expect("deserialize legacy planning step");

        assert_eq!(state.pending_tools.len(), 1);
        assert!(state.pending_tools[0].spec().is_none());
        assert!(state.loaded_deferred_tools.is_empty());
        assert_eq!(state.pending_tools[0].tool_use.id, "tool-1");
        assert_eq!(step, AgentStep::ModelStep);
    }
}
