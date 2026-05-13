use async_trait::async_trait;
use serde_json::Value;

use crate::error::MachineError;
use crate::run::{RunId, SessionId};

#[derive(Debug, Clone)]
pub struct RunStart {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub agent_kind: String,
    pub model: String,
    pub client_run_key: Option<String>,
    pub parent_run_id: Option<RunId>,
    pub retry_of_run_id: Option<RunId>,
    pub scope: Value,
}

#[derive(Debug, Clone)]
pub struct RunFinish {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub status: &'static str,
    pub finish_reason: &'static str,
    pub error_code: Option<&'static str>,
    pub event_json: Value,
    pub snapshot_json: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct RunLookup {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub status: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FinishRunResult {
    pub won: bool,
    pub terminal_event_json: Value,
}

#[derive(Debug, Clone)]
pub enum StartRunResult {
    Created,
    Existing(RunLookup),
}

#[async_trait]
pub trait RunStore: Send + Sync {
    async fn ensure_session(
        &self,
        session_id: Option<SessionId>,
        scope: &Value,
    ) -> Result<SessionId, MachineError>;

    async fn start_run(&self, run: &RunStart) -> Result<StartRunResult, MachineError>;

    async fn finish_run(&self, finish: &RunFinish) -> Result<FinishRunResult, MachineError>;

    async fn find_idempotent_run(
        &self,
        scope: &Value,
        session_id: &SessionId,
        key: &str,
    ) -> Result<Option<RunLookup>, MachineError>;

    async fn mark_cancelled(&self, run_id: &RunId) -> Result<(), MachineError>;

    async fn record_event(&self, run_id: &RunId, event_json: &Value) -> Result<bool, MachineError>;

    async fn list_events(
        &self,
        run_id: &RunId,
        scope: &Value,
        after_seq: i64,
    ) -> Result<Vec<Value>, MachineError>;
}
