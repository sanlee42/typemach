use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

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
    Text {
        text: String,
    },
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
            raw: None,
        }
    }

    pub fn error(tool_use: &ToolUse, message: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use.id.clone(),
            name: tool_use.name.clone(),
            content: json!({ "error": message.into() }),
            is_error: true,
            raw: None,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ModelResponse {
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    #[serde(default)]
    pub deltas: Vec<String>,
    #[serde(default)]
    pub tool_uses: Vec<ToolUse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_text: Option<String>,
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
    pub recent_turns: usize,
    pub max_tool_result_bytes: usize,
    pub background_digest: bool,
}

impl Default for ContextPolicy {
    fn default() -> Self {
        Self {
            max_input_tokens: 128_000,
            compact_at_tokens: 96_000,
            recent_turns: 8,
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
    pub next_delta_index: usize,
    pub pending_tools: VecDeque<ToolUse>,
    pub pending_human: Option<ToolUse>,
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
}
