use super::*;
use crate::op::Page;
use crate::run::{LeaseId, ThreadId, WorkerId};
use crate::store::{
    LeaseClaim, MemoryRunStore, RunFinishRecord, RunStart, RunStore, StoreStartResult,
};
use async_rt::sync::Notify;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
struct TestPayload {
    terminal: bool,
    name: &'static str,
}

impl RunEventPayload for TestPayload {
    fn is_terminal(&self) -> bool {
        self.terminal
    }
}

type TestEvent = RunEventEnvelope<TestPayload>;

fn lifecycle() -> RunLifecycle<TestEvent, MemoryRunStore<TestEvent>> {
    RunLifecycle::new(
        RunRegistry::new(),
        Arc::new(MemoryRunStore::<TestEvent>::new()),
        8,
    )
}

fn run_start(run_id: &str, key: Option<&str>) -> RunStart {
    RunStart {
        run_id: RunId::from(run_id),
        session_id: SessionId::from("session-a"),
        thread_id: ThreadId::from(format!("thread-{run_id}")),
        agent_kind: "test".to_string(),
        model: None,
        client_run_key: key.map(str::to_string),
        parent_run_id: None,
        retry_of_run_id: None,
        scope: scope(),
        metadata: serde_json::json!({}),
        lease: None,
    }
}

fn leased_run_start(run_id: &str, owner: &str) -> RunStart {
    let mut run = run_start(run_id, None);
    run.lease = Some(LeaseClaim::new(
        WorkerId::from(owner),
        LeaseId::from("lease-a"),
        Duration::from_secs(30),
    ));
    run
}

fn scope() -> Value {
    serde_json::json!({"tenant": "demo"})
}

fn payload(terminal: bool) -> TestPayload {
    TestPayload {
        terminal,
        name: if terminal { "terminal" } else { "event" },
    }
}

fn event(run_id: &str, seq: i64, terminal: bool) -> TestEvent {
    RunEventEnvelope::new(
        RunId::from(run_id),
        SessionId::from("session-a"),
        seq,
        payload(terminal),
    )
}

fn finish_request(run_id: &str, status: RunStatus) -> RunFinish {
    RunFinish {
        run_id: RunId::from(run_id),
        session_id: SessionId::from("session-a"),
        scope: scope(),
        status,
        finish_reason: "stop".to_string(),
        error_code: None,
        data: (),
    }
}

#[path = "lifecycle_tests/control.rs"]
mod control;
#[path = "lifecycle_tests/events.rs"]
mod events;

#[derive(Debug, Clone)]
struct BlockingRecordStore {
    inner: MemoryRunStore<TestEvent>,
    block_seq: i64,
    blocked: Arc<Notify>,
    release: Arc<Notify>,
}

impl BlockingRecordStore {
    fn new(block_seq: i64) -> Self {
        Self {
            inner: MemoryRunStore::new(),
            block_seq,
            blocked: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        }
    }
}

#[async_trait]
impl RunStore<TestEvent> for BlockingRecordStore {
    type Scope = Value;
    type FinishData = ();

    async fn ensure_session(
        &self,
        session_id: Option<SessionId>,
        scope: &Value,
    ) -> Result<SessionId, MachineError> {
        self.inner.ensure_session(session_id, scope).await
    }

    async fn start_run(&self, run: &RunStart<Value>) -> Result<StoreStartResult, MachineError> {
        self.inner.start_run(run).await
    }

    async fn lookup_run(
        &self,
        run_id: &RunId,
        scope: &Value,
    ) -> Result<Option<RunLookup>, MachineError> {
        self.inner.lookup_run(run_id, scope).await
    }

    async fn finish_run(
        &self,
        finish: &RunFinishRecord<TestEvent, (), Value>,
    ) -> Result<FinishRunResult<TestEvent>, MachineError> {
        self.inner.finish_run(finish).await
    }

    async fn terminal_event(
        &self,
        run_id: &RunId,
        scope: &Value,
    ) -> Result<Option<TestEvent>, MachineError> {
        self.inner.terminal_event(run_id, scope).await
    }

    async fn find_idempotent_run(
        &self,
        scope: &Value,
        session_id: &SessionId,
        key: &str,
    ) -> Result<Option<RunLookup>, MachineError> {
        self.inner.find_idempotent_run(scope, session_id, key).await
    }

    async fn mark_cancelled(&self, run_id: &RunId, scope: &Value) -> Result<(), MachineError> {
        self.inner.mark_cancelled(run_id, scope).await
    }

    async fn record_event(
        &self,
        run_id: &RunId,
        scope: &Value,
        event: &TestEvent,
    ) -> Result<bool, MachineError> {
        if event.seq() == self.block_seq {
            self.blocked.notify_one();
            self.release.notified().await;
        }
        self.inner.record_event(run_id, scope, event).await
    }

    async fn list_events(
        &self,
        run_id: &RunId,
        scope: &Value,
        after_seq: i64,
        limit: usize,
    ) -> Result<Page<TestEvent>, MachineError> {
        self.inner
            .list_events(run_id, scope, after_seq, limit)
            .await
    }
}

fn block_on<F>(future: F) -> F::Output
where
    F: std::future::Future,
{
    async_rt::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
        .block_on(future)
}
