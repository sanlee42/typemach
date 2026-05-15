use std::collections::HashMap;
use std::sync::Arc;

use async_rt::sync::Mutex;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::MachineError;
use crate::run::{RunId, SessionId};

pub trait RunEvent: Clone + Send + Sync + 'static {
    fn run_id(&self) -> &RunId;
    fn session_id(&self) -> &SessionId;
    fn seq(&self) -> i64;
    fn is_terminal(&self) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Completed,
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
            Self::Cancelled => "cancelled",
            Self::Error => "error",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunStart {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub agent_kind: String,
    pub model: Option<String>,
    pub client_run_key: Option<String>,
    pub parent_run_id: Option<RunId>,
    pub retry_of_run_id: Option<RunId>,
    pub scope: Value,
    pub metadata: Value,
}

#[derive(Debug, Clone)]
pub struct RunFinish<E: RunEvent> {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub status: RunStatus,
    pub finish_reason: String,
    pub error_code: Option<String>,
    pub terminal_event: E,
    pub snapshot_json: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct RunFinishRequest {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub status: RunStatus,
    pub finish_reason: String,
    pub error_code: Option<String>,
    pub snapshot_json: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct RunLookup {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub status: RunStatus,
    pub finish_reason: Option<String>,
    pub cancel_requested: bool,
}

#[derive(Debug, Clone)]
pub struct FinishRunResult<E: RunEvent> {
    pub won: bool,
    pub terminal_event: E,
}

#[derive(Debug, Clone)]
pub enum StartRunResult {
    Created,
    Existing(RunLookup),
}

#[async_trait]
pub trait RunStore<E>: Send + Sync
where
    E: RunEvent,
{
    async fn ensure_session(
        &self,
        session_id: Option<SessionId>,
        scope: &Value,
    ) -> Result<SessionId, MachineError>;

    async fn start_run(&self, run: &RunStart) -> Result<StartRunResult, MachineError>;

    async fn lookup_run(
        &self,
        run_id: &RunId,
        scope: &Value,
    ) -> Result<Option<RunLookup>, MachineError>;

    async fn finish_run(&self, finish: &RunFinish<E>) -> Result<FinishRunResult<E>, MachineError>;

    async fn terminal_event(&self, run_id: &RunId) -> Result<Option<E>, MachineError>;

    async fn find_idempotent_run(
        &self,
        scope: &Value,
        session_id: &SessionId,
        key: &str,
    ) -> Result<Option<RunLookup>, MachineError>;

    async fn mark_cancelled(&self, run_id: &RunId) -> Result<(), MachineError>;

    async fn record_event(&self, run_id: &RunId, event: &E) -> Result<bool, MachineError>;

    async fn list_events(
        &self,
        run_id: &RunId,
        scope: &Value,
        after_seq: i64,
    ) -> Result<Vec<E>, MachineError>;
}

#[derive(Debug)]
pub struct MemoryRunStore<E: RunEvent> {
    inner: Arc<Mutex<MemoryRunStoreInner<E>>>,
}

impl<E: RunEvent> Clone for MemoryRunStore<E> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<E: RunEvent> Default for MemoryRunStore<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: RunEvent> MemoryRunStore<E> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MemoryRunStoreInner::default())),
        }
    }
}

#[derive(Debug)]
struct MemoryRunStoreInner<E: RunEvent> {
    next_session: u64,
    sessions: HashMap<SessionId, Value>,
    runs: HashMap<RunId, MemoryRun<E>>,
    idempotency: HashMap<(String, SessionId, String), RunId>,
}

impl<E: RunEvent> Default for MemoryRunStoreInner<E> {
    fn default() -> Self {
        Self {
            next_session: 0,
            sessions: HashMap::new(),
            runs: HashMap::new(),
            idempotency: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct MemoryRun<E: RunEvent> {
    start: RunStart,
    status: RunStatus,
    finish_reason: Option<String>,
    cancel_requested: bool,
    terminal_event: Option<E>,
    events: Vec<E>,
}

#[async_trait]
impl<E> RunStore<E> for MemoryRunStore<E>
where
    E: RunEvent,
{
    async fn ensure_session(
        &self,
        session_id: Option<SessionId>,
        scope: &Value,
    ) -> Result<SessionId, MachineError> {
        let mut inner = self.inner.lock().await;
        let session_id = match session_id {
            Some(session_id) => session_id,
            None => loop {
                inner.next_session += 1;
                let candidate = SessionId::from(format!("session-{}", inner.next_session));
                if !inner.sessions.contains_key(&candidate) {
                    break candidate;
                }
            },
        };
        inner
            .sessions
            .entry(session_id.clone())
            .or_insert_with(|| scope.clone());
        Ok(session_id)
    }

    async fn start_run(&self, run: &RunStart) -> Result<StartRunResult, MachineError> {
        let mut inner = self.inner.lock().await;
        inner
            .sessions
            .entry(run.session_id.clone())
            .or_insert_with(|| run.scope.clone());
        if let Some(existing) = inner.runs.get(&run.run_id) {
            return Ok(StartRunResult::Existing(run_lookup(existing)));
        }
        if let Some(client_key) = &run.client_run_key {
            let idempotency_key = (
                scope_key(&run.scope)?,
                run.session_id.clone(),
                client_key.clone(),
            );
            if let Some(existing_run_id) = inner.idempotency.get(&idempotency_key)
                && let Some(existing) = inner.runs.get(existing_run_id)
            {
                return Ok(StartRunResult::Existing(run_lookup(existing)));
            }
            inner
                .idempotency
                .insert(idempotency_key, run.run_id.clone());
        }
        inner.runs.insert(
            run.run_id.clone(),
            MemoryRun {
                start: run.clone(),
                status: RunStatus::Running,
                finish_reason: None,
                cancel_requested: false,
                terminal_event: None,
                events: Vec::new(),
            },
        );
        Ok(StartRunResult::Created)
    }

    async fn lookup_run(
        &self,
        run_id: &RunId,
        scope: &Value,
    ) -> Result<Option<RunLookup>, MachineError> {
        let inner = self.inner.lock().await;
        Ok(inner
            .runs
            .get(run_id)
            .filter(|run| run.start.scope == *scope)
            .map(run_lookup))
    }

    async fn finish_run(&self, finish: &RunFinish<E>) -> Result<FinishRunResult<E>, MachineError> {
        validate_event_run(&finish.run_id, &finish.session_id, &finish.terminal_event)?;
        if !finish.terminal_event.is_terminal() {
            return Err(MachineError::InvalidRunEvent {
                reason: "finish_run requires a terminal event".to_string(),
            });
        }
        if !finish.status.is_terminal() {
            return Err(MachineError::InvalidRunEvent {
                reason: "finish_run requires a terminal status".to_string(),
            });
        }
        let mut inner = self.inner.lock().await;
        let run = inner
            .runs
            .get_mut(&finish.run_id)
            .ok_or(MachineError::RunNotFound)?;
        if finish.session_id != run.start.session_id {
            return Err(MachineError::InvalidRunEvent {
                reason: "finish session_id does not match target run".to_string(),
            });
        }
        if run.status.is_terminal() {
            let terminal_event = run
                .terminal_event
                .clone()
                .ok_or(MachineError::RunNotFound)?;
            return Ok(FinishRunResult {
                won: false,
                terminal_event,
            });
        }
        validate_next_seq(run, &finish.terminal_event)?;
        run.status = finish.status;
        run.finish_reason = Some(finish.finish_reason.clone());
        run.terminal_event = Some(finish.terminal_event.clone());
        run.events.push(finish.terminal_event.clone());
        Ok(FinishRunResult {
            won: true,
            terminal_event: finish.terminal_event.clone(),
        })
    }

    async fn terminal_event(&self, run_id: &RunId) -> Result<Option<E>, MachineError> {
        let inner = self.inner.lock().await;
        Ok(inner
            .runs
            .get(run_id)
            .and_then(|run| run.terminal_event.clone()))
    }

    async fn find_idempotent_run(
        &self,
        scope: &Value,
        session_id: &SessionId,
        key: &str,
    ) -> Result<Option<RunLookup>, MachineError> {
        let inner = self.inner.lock().await;
        let idempotency_key = (scope_key(scope)?, session_id.clone(), key.to_string());
        Ok(inner
            .idempotency
            .get(&idempotency_key)
            .and_then(|run_id| inner.runs.get(run_id))
            .map(run_lookup))
    }

    async fn mark_cancelled(&self, run_id: &RunId) -> Result<(), MachineError> {
        let mut inner = self.inner.lock().await;
        let run = inner
            .runs
            .get_mut(run_id)
            .ok_or(MachineError::RunNotFound)?;
        run.cancel_requested = true;
        Ok(())
    }

    async fn record_event(&self, run_id: &RunId, event: &E) -> Result<bool, MachineError> {
        if event.is_terminal() {
            return Err(MachineError::InvalidRunEvent {
                reason: "record_event does not accept terminal events".to_string(),
            });
        }
        if event.run_id() != run_id {
            return Err(MachineError::InvalidRunEvent {
                reason: "event run_id does not match target run".to_string(),
            });
        }
        let mut inner = self.inner.lock().await;
        let Some(run) = inner.runs.get_mut(run_id) else {
            return Ok(false);
        };
        if run.status.is_terminal() {
            return Ok(false);
        }
        if event.session_id() != &run.start.session_id {
            return Err(MachineError::InvalidRunEvent {
                reason: "event session_id does not match target run".to_string(),
            });
        }
        validate_next_seq(run, event)?;
        run.events.push(event.clone());
        Ok(true)
    }

    async fn list_events(
        &self,
        run_id: &RunId,
        scope: &Value,
        after_seq: i64,
    ) -> Result<Vec<E>, MachineError> {
        let inner = self.inner.lock().await;
        Ok(inner
            .runs
            .get(run_id)
            .filter(|run| run.start.scope == *scope)
            .map(|run| {
                let mut events = run
                    .events
                    .iter()
                    .filter(|event| event.seq() > after_seq)
                    .cloned()
                    .collect::<Vec<_>>();
                events.sort_by_key(|event| event.seq());
                events
            })
            .unwrap_or_default())
    }
}

fn run_lookup<E: RunEvent>(run: &MemoryRun<E>) -> RunLookup {
    RunLookup {
        run_id: run.start.run_id.clone(),
        session_id: run.start.session_id.clone(),
        status: run.status,
        finish_reason: run.finish_reason.clone(),
        cancel_requested: run.cancel_requested,
    }
}

fn scope_key(scope: &Value) -> Result<String, MachineError> {
    serde_json::to_string(scope).map_err(MachineError::Serialization)
}

fn validate_event_run<E: RunEvent>(
    run_id: &RunId,
    session_id: &SessionId,
    event: &E,
) -> Result<(), MachineError> {
    if event.run_id() != run_id {
        return Err(MachineError::InvalidRunEvent {
            reason: "event run_id does not match target run".to_string(),
        });
    }
    if event.session_id() != session_id {
        return Err(MachineError::InvalidRunEvent {
            reason: "event session_id does not match target run".to_string(),
        });
    }
    Ok(())
}

fn validate_next_seq<E: RunEvent>(run: &MemoryRun<E>, event: &E) -> Result<(), MachineError> {
    if event.seq() <= 0 {
        return Err(MachineError::InvalidRunEvent {
            reason: "event seq must be positive".to_string(),
        });
    }
    if let Some(last) = run.events.last()
        && event.seq() <= last.seq()
    {
        return Err(MachineError::InvalidRunEvent {
            reason: "event seq must increase monotonically".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestEvent {
        run_id: RunId,
        session_id: SessionId,
        seq: i64,
        terminal: bool,
        name: &'static str,
    }

    impl RunEvent for TestEvent {
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
            self.terminal
        }
    }

    fn run_start(run_id: &str, session_id: &str, key: Option<&str>) -> RunStart {
        RunStart {
            run_id: RunId::from(run_id),
            session_id: SessionId::from(session_id),
            agent_kind: "test".to_string(),
            model: None,
            client_run_key: key.map(str::to_string),
            parent_run_id: None,
            retry_of_run_id: None,
            scope: serde_json::json!({"tenant": "demo"}),
            metadata: serde_json::json!({}),
        }
    }

    fn event(run_id: &str, session_id: &str, seq: i64, terminal: bool) -> TestEvent {
        TestEvent {
            run_id: RunId::from(run_id),
            session_id: SessionId::from(session_id),
            seq,
            terminal,
            name: if terminal { "terminal" } else { "event" },
        }
    }

    #[test]
    fn memory_store_idempotent_start_returns_existing_run() {
        block_on(async {
            let store = MemoryRunStore::<TestEvent>::new();
            let first = run_start("run-a", "session-a", Some("client-key"));
            let second = run_start("run-b", "session-a", Some("client-key"));
            assert!(matches!(
                store.start_run(&first).await.expect("start"),
                StartRunResult::Created
            ));
            match store.start_run(&second).await.expect("idempotent") {
                StartRunResult::Existing(existing) => {
                    assert_eq!(existing.run_id, RunId::from("run-a"));
                    assert_eq!(existing.status, RunStatus::Running);
                    assert!(!existing.cancel_requested);
                }
                StartRunResult::Created => panic!("expected existing run"),
            }
        });
    }

    #[test]
    fn memory_store_records_running_events_and_skips_after_terminal() {
        block_on(async {
            let store = MemoryRunStore::<TestEvent>::new();
            let start = run_start("run-a", "session-a", None);
            store.start_run(&start).await.expect("start");

            assert!(
                store
                    .record_event(
                        &RunId::from("run-a"),
                        &event("run-a", "session-a", 1, false)
                    )
                    .await
                    .expect("record")
            );
            let terminal = event("run-a", "session-a", 2, true);
            let finish = RunFinish {
                run_id: RunId::from("run-a"),
                session_id: SessionId::from("session-a"),
                status: RunStatus::Completed,
                finish_reason: "stop".to_string(),
                error_code: None,
                terminal_event: terminal.clone(),
                snapshot_json: None,
            };
            let result = store.finish_run(&finish).await.expect("finish");
            assert!(result.won);
            assert_eq!(
                store
                    .terminal_event(&RunId::from("run-a"))
                    .await
                    .expect("terminal event"),
                Some(terminal.clone())
            );
            assert!(
                !store
                    .record_event(
                        &RunId::from("run-a"),
                        &event("run-a", "session-a", 3, false)
                    )
                    .await
                    .expect("post-terminal record")
            );

            let events = store
                .list_events(&RunId::from("run-a"), &start.scope, 0)
                .await
                .expect("events");
            assert_eq!(events.len(), 2);
            assert_eq!(events[0].seq, 1);
            assert_eq!(events[1], terminal);
        });
    }

    #[test]
    fn memory_store_terminal_competes_once() {
        block_on(async {
            let store = MemoryRunStore::<TestEvent>::new();
            let start = run_start("run-a", "session-a", None);
            store.start_run(&start).await.expect("start");
            let first_terminal = event("run-a", "session-a", 1, true);
            let second_terminal = event("run-a", "session-a", 2, true);
            let first = RunFinish {
                run_id: RunId::from("run-a"),
                session_id: SessionId::from("session-a"),
                status: RunStatus::Completed,
                finish_reason: "stop".to_string(),
                error_code: None,
                terminal_event: first_terminal.clone(),
                snapshot_json: None,
            };
            let mut second = first.clone();
            second.terminal_event = second_terminal;
            second.status = RunStatus::Error;
            second.finish_reason = "runtime_failed".to_string();

            assert!(store.finish_run(&first).await.expect("first").won);
            let result = store.finish_run(&second).await.expect("second");
            assert!(!result.won);
            assert_eq!(result.terminal_event, first_terminal);
        });
    }

    #[test]
    fn memory_store_marks_cancel_requested() {
        block_on(async {
            let store = MemoryRunStore::<TestEvent>::new();
            let start = run_start("run-a", "session-a", None);
            store.start_run(&start).await.expect("start");

            store
                .mark_cancelled(&RunId::from("run-a"))
                .await
                .expect("cancel");

            let lookup = store
                .lookup_run(&RunId::from("run-a"), &start.scope)
                .await
                .expect("lookup")
                .expect("run");
            assert!(lookup.cancel_requested);
            assert_eq!(lookup.status, RunStatus::Running);
        });
    }

    #[test]
    fn memory_store_rejects_non_increasing_event_seq() {
        block_on(async {
            let store = MemoryRunStore::<TestEvent>::new();
            let start = run_start("run-a", "session-a", None);
            store.start_run(&start).await.expect("start");
            store
                .record_event(
                    &RunId::from("run-a"),
                    &event("run-a", "session-a", 1, false),
                )
                .await
                .expect("record first");

            let err = store
                .record_event(
                    &RunId::from("run-a"),
                    &event("run-a", "session-a", 1, false),
                )
                .await
                .expect_err("duplicate seq should fail");
            assert!(matches!(err, MachineError::InvalidRunEvent { .. }));
        });
    }

    #[test]
    fn memory_store_lists_events_after_cursor() {
        block_on(async {
            let store = MemoryRunStore::<TestEvent>::new();
            let start = run_start("run-a", "session-a", None);
            store.start_run(&start).await.expect("start");
            for seq in 1..=3 {
                store
                    .record_event(
                        &RunId::from("run-a"),
                        &event("run-a", "session-a", seq, false),
                    )
                    .await
                    .expect("record");
            }
            let events = store
                .list_events(&RunId::from("run-a"), &start.scope, 1)
                .await
                .expect("events");
            assert_eq!(
                events.iter().map(|event| event.seq).collect::<Vec<_>>(),
                vec![2, 3]
            );
        });
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
}
