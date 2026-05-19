use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::checkpoint::CheckpointRecord;
use crate::error::MachineError;
use crate::run::{LeaseId, RunId, SessionId, ThreadId, WorkerId};

pub trait RunEvent: Clone + Send + Sync + 'static {
    fn run_id(&self) -> &RunId;
    fn session_id(&self) -> &SessionId;
    fn seq(&self) -> i64;
    fn is_terminal(&self) -> bool;
}

pub trait RunEventPayload: Clone + Send + Sync + 'static {
    fn is_terminal(&self) -> bool;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunEventEnvelope<P> {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub seq: i64,
    pub payload: P,
}

impl<P> RunEventEnvelope<P> {
    pub fn new(run_id: RunId, session_id: SessionId, seq: i64, payload: P) -> Self {
        Self {
            run_id,
            session_id,
            seq,
            payload,
        }
    }

    pub fn into_payload(self) -> P {
        self.payload
    }
}

impl<P> RunEvent for RunEventEnvelope<P>
where
    P: RunEventPayload,
{
    fn run_id(&self) -> &RunId {
        &self.run_id
    }

    fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    fn seq(&self) -> i64 {
        self.seq
    }

    fn is_terminal(&self) -> bool {
        self.payload.is_terminal()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Completed,
    Interrupted,
    Cancelled,
    Error,
}

impl RunStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Interrupted => "interrupted",
            Self::Cancelled => "cancelled",
            Self::Error => "error",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "running" => Some(Self::Running),
            "completed" => Some(Self::Completed),
            "interrupted" => Some(Self::Interrupted),
            "cancelled" => Some(Self::Cancelled),
            "error" => Some(Self::Error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lease {
    pub run: RunId,
    pub owner: WorkerId,
    pub id: LeaseId,
}

impl Lease {
    pub fn new(run: RunId, owner: WorkerId, id: LeaseId) -> Self {
        Self { run, owner, id }
    }
}

#[derive(Debug, Clone)]
pub struct LeaseClaim {
    pub owner: WorkerId,
    pub id: LeaseId,
    pub ttl: Duration,
}

impl LeaseClaim {
    pub fn new(owner: WorkerId, id: LeaseId, ttl: Duration) -> Self {
        Self { owner, id, ttl }
    }
}

#[derive(Debug, Clone)]
pub struct RunStart<Scope = Value> {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub agent_kind: String,
    pub model: Option<String>,
    pub client_run_key: Option<String>,
    pub parent_run_id: Option<RunId>,
    pub retry_of_run_id: Option<RunId>,
    pub scope: Scope,
    pub metadata: Value,
    pub lease: Option<LeaseClaim>,
}

#[derive(Debug, Clone)]
pub struct RunFinishRecord<E: RunEvent, Data = (), Scope = Value> {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub scope: Scope,
    pub status: RunStatus,
    pub finish_reason: String,
    pub error_code: Option<String>,
    pub terminal_event: E,
    pub data: Data,
}

#[derive(Debug, Clone)]
pub struct RunFinish<Data = (), Scope = Value> {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub scope: Scope,
    pub status: RunStatus,
    pub finish_reason: String,
    pub error_code: Option<String>,
    pub data: Data,
}

impl<Data, Scope> RunFinish<Data, Scope> {
    pub fn into_record<E>(self, terminal_event: E) -> RunFinishRecord<E, Data, Scope>
    where
        E: RunEvent,
    {
        RunFinishRecord {
            run_id: self.run_id,
            session_id: self.session_id,
            scope: self.scope,
            status: self.status,
            finish_reason: self.finish_reason,
            error_code: self.error_code,
            terminal_event,
            data: self.data,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointWrite {
    pub thread_id: ThreadId,
    pub record: CheckpointRecord,
}

impl CheckpointWrite {
    pub fn new(thread_id: ThreadId, record: CheckpointRecord) -> Self {
        Self { thread_id, record }
    }
}

#[derive(Debug, Clone)]
pub struct RunCommit<E: RunEvent, Data = (), Scope = Value> {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub scope: Scope,
    pub lease: Option<LeaseId>,
    pub checkpoint: Option<CheckpointWrite>,
    pub events: Vec<E>,
    pub finish: Option<RunFinish<Data, Scope>>,
}

#[derive(Debug, Clone)]
pub struct CommitPlan<Data = (), Scope = Value> {
    pub lease: Option<LeaseId>,
    pub checkpoint: Option<CheckpointWrite>,
    pub event_count: usize,
    pub finish: Option<RunFinish<Data, Scope>>,
}

#[derive(Debug, Clone)]
pub enum RunCommitResult<E: RunEvent> {
    Recorded(Vec<E>),
    Finished {
        events: Vec<E>,
        result: FinishRunResult<E>,
    },
    Skipped,
}

impl<E: RunEvent> RunCommitResult<E> {
    pub fn is_skipped(&self) -> bool {
        matches!(self, Self::Skipped)
    }

    pub fn events(&self) -> &[E] {
        match self {
            Self::Recorded(events) | Self::Finished { events, .. } => events,
            Self::Skipped => &[],
        }
    }

    pub fn finish_result(&self) -> Option<&FinishRunResult<E>> {
        match self {
            Self::Finished { result, .. } => Some(result),
            Self::Recorded(_) | Self::Skipped => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunLookup {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub status: RunStatus,
    pub finish_reason: Option<String>,
    pub cancel_requested: bool,
    pub owner: Option<WorkerId>,
}

#[derive(Debug, Clone)]
pub enum FinishRunResult<E: RunEvent> {
    Finished(E),
    AlreadyFinished(E),
}

impl<E: RunEvent> FinishRunResult<E> {
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Finished(_))
    }

    pub fn is_already_finished(&self) -> bool {
        matches!(self, Self::AlreadyFinished(_))
    }

    pub fn terminal_event(&self) -> &E {
        match self {
            Self::Finished(event) | Self::AlreadyFinished(event) => event,
        }
    }

    pub fn into_terminal_event(self) -> E {
        match self {
            Self::Finished(event) | Self::AlreadyFinished(event) => event,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StoreStartResult {
    Created,
    Existing(RunLookup),
}

#[async_trait]
pub trait RunStore<E>: Send + Sync
where
    E: RunEvent,
{
    type Scope: Clone + Send + Sync + 'static;
    type FinishData: Clone + Send + Sync + 'static;

    async fn ensure_session(
        &self,
        session_id: Option<SessionId>,
        scope: &Self::Scope,
    ) -> Result<SessionId, MachineError>;

    async fn start_run(
        &self,
        run: &RunStart<Self::Scope>,
    ) -> Result<StoreStartResult, MachineError>;

    async fn lookup_run(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
    ) -> Result<Option<RunLookup>, MachineError>;

    async fn finish_run(
        &self,
        finish: &RunFinishRecord<E, Self::FinishData, Self::Scope>,
    ) -> Result<FinishRunResult<E>, MachineError>;

    async fn terminal_event(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
    ) -> Result<Option<E>, MachineError>;

    async fn find_idempotent_run(
        &self,
        scope: &Self::Scope,
        session_id: &SessionId,
        key: &str,
    ) -> Result<Option<RunLookup>, MachineError>;

    async fn mark_cancelled(&self, run_id: &RunId, scope: &Self::Scope)
    -> Result<(), MachineError>;

    async fn record_event(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
        event: &E,
    ) -> Result<bool, MachineError>;

    async fn list_events(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
        after_seq: i64,
    ) -> Result<Vec<E>, MachineError>;
}

#[async_trait]
pub trait RunTx<E>: RunStore<E>
where
    E: RunEvent,
{
    async fn commit_run(
        &self,
        commit: &RunCommit<E, Self::FinishData, Self::Scope>,
    ) -> Result<RunCommitResult<E>, MachineError>;
}

#[async_trait]
pub trait RunLease<E>: Send + Sync
where
    E: RunEvent,
{
    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<bool, MachineError>;

    async fn release(&self, lease: &Lease) -> Result<(), MachineError>;

    async fn reap_stale<F>(
        &self,
        owner: &WorkerId,
        limit: usize,
        build_event: F,
    ) -> Result<Vec<RunLookup>, MachineError>
    where
        F: FnMut(&RunLookup, i64) -> E + Send;
}

mod memory;
pub use memory::MemoryRunStore;
#[cfg(test)]
mod memory_tests;
