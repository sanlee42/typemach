use std::collections::HashMap;
use std::sync::Arc;

use async_rt::sync::{Mutex, mpsc};
use serde_json::Value;

use crate::error::MachineError;
use crate::registry::{RunHandle, RunRegistry};
use crate::run::{RunId, SessionId};
use crate::store::{
    FinishRunResult, RunEvent, RunEventEnvelope, RunEventPayload, RunFinish, RunFinishRecord,
    RunLookup, RunStart, RunStatus, RunStore, StartRunResult,
};

#[derive(Debug)]
pub enum AppendEventResult<E: RunEvent> {
    Recorded(E),
    Skipped,
}

#[derive(Debug)]
pub enum RunSubscription<E: RunEvent> {
    Active { replay: Vec<E>, tail: RunTail<E> },
    Inactive { status: RunStatus, replay: Vec<E> },
    Missing,
}

#[derive(Debug)]
pub struct RunTail<E: RunEvent> {
    receiver: mpsc::UnboundedReceiver<E>,
    after_seq: i64,
}

impl<E: RunEvent> RunTail<E> {
    fn new(receiver: mpsc::UnboundedReceiver<E>, after_seq: i64) -> Self {
        Self {
            receiver,
            after_seq,
        }
    }

    pub fn cursor(&self) -> i64 {
        self.after_seq
    }

    pub async fn next_event(&mut self) -> Option<E> {
        while let Some(event) = self.receiver.recv().await {
            if event.seq() <= self.after_seq {
                continue;
            }
            self.after_seq = event.seq();
            return Some(event);
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct RunLifecycle<E, S>
where
    E: RunEvent,
    S: RunStore<E>,
{
    registry: RunRegistry<E>,
    store: Arc<S>,
    event_locks: Arc<Mutex<HashMap<RunId, Arc<Mutex<()>>>>>,
    max_in_flight: usize,
}

impl<E, S> RunLifecycle<E, S>
where
    E: RunEvent,
    S: RunStore<E>,
{
    pub fn new(registry: RunRegistry<E>, store: Arc<S>, max_in_flight: usize) -> Self {
        Self {
            registry,
            store,
            event_locks: Arc::new(Mutex::new(HashMap::new())),
            max_in_flight,
        }
    }

    pub fn registry(&self) -> &RunRegistry<E> {
        &self.registry
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub async fn ensure_session(
        &self,
        session_id: Option<SessionId>,
        scope: &Value,
    ) -> Result<SessionId, MachineError> {
        self.store.ensure_session(session_id, scope).await
    }

    pub async fn start_run(
        &self,
        run: RunStart,
        handle: RunHandle,
        stream_sender: Option<mpsc::UnboundedSender<E>>,
    ) -> Result<StartRunResult, MachineError> {
        if let Some(key) = run.client_run_key.as_deref()
            && let Some(existing) = self
                .store
                .find_idempotent_run(&run.scope, &run.session_id, key)
                .await?
        {
            return Ok(StartRunResult::Existing(existing));
        }

        self.registry.try_insert(
            run.run_id.clone(),
            handle,
            stream_sender,
            self.max_in_flight,
        )?;
        match self.store.start_run(&run).await {
            Ok(StartRunResult::Created) => Ok(StartRunResult::Created),
            Ok(StartRunResult::Existing(existing)) => {
                self.registry.remove(&run.run_id);
                Ok(StartRunResult::Existing(existing))
            }
            Err(err) => {
                self.registry.remove(&run.run_id);
                Err(err)
            }
        }
    }

    async fn event_lock(&self, run_id: &RunId) -> Arc<Mutex<()>> {
        let mut locks = self.event_locks.lock().await;
        Arc::clone(
            locks
                .entry(run_id.clone())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }

    async fn remove_event_lock(&self, run_id: &RunId) {
        self.event_locks.lock().await.remove(run_id);
    }

    pub async fn append_with<F>(
        &self,
        run_id: &RunId,
        build_event: F,
    ) -> Result<AppendEventResult<E>, MachineError>
    where
        F: FnOnce(i64) -> E,
    {
        let lock = self.event_lock(run_id).await;
        let guard = lock.lock().await;
        let Some(seq) = self.registry.next_seq(run_id) else {
            drop(guard);
            self.remove_event_lock(run_id).await;
            return Err(MachineError::RunNotFound);
        };
        let event = build_event(seq);
        if event.run_id() != run_id {
            return Err(MachineError::InvalidRunEvent {
                reason: "event run_id does not match target run".to_string(),
            });
        }
        if event.seq() != seq {
            return Err(MachineError::InvalidRunEvent {
                reason: "event seq does not match allocated seq".to_string(),
            });
        }
        if event.is_terminal() {
            return Err(MachineError::InvalidRunEvent {
                reason: "append_event does not accept terminal events".to_string(),
            });
        }
        let recorded = self.store.record_event(run_id, &event).await?;
        if recorded {
            self.registry.publish(run_id, event.clone());
            Ok(AppendEventResult::Recorded(event))
        } else {
            self.registry.remove(run_id);
            drop(guard);
            self.remove_event_lock(run_id).await;
            Ok(AppendEventResult::Skipped)
        }
    }

    pub async fn finish_with<F>(
        &self,
        finish: RunFinish,
        build_event: F,
    ) -> Result<FinishRunResult<E>, MachineError>
    where
        F: FnOnce(i64) -> E,
    {
        if !finish.status.is_terminal() {
            return Err(MachineError::InvalidRunEvent {
                reason: "finish_run requires a terminal status".to_string(),
            });
        }

        let lock = self.event_lock(&finish.run_id).await;
        let guard = lock.lock().await;
        let Some(seq) = self.registry.next_seq(&finish.run_id) else {
            if let Some(terminal_event) = self.store.terminal_event(&finish.run_id).await? {
                drop(guard);
                self.remove_event_lock(&finish.run_id).await;
                return Ok(FinishRunResult::AlreadyFinished(terminal_event));
            }
            drop(guard);
            self.remove_event_lock(&finish.run_id).await;
            return Err(MachineError::RunNotFound);
        };

        let terminal_event = build_event(seq);
        if terminal_event.run_id() != &finish.run_id {
            return Err(MachineError::InvalidRunEvent {
                reason: "event run_id does not match target run".to_string(),
            });
        }
        if terminal_event.session_id() != &finish.session_id {
            return Err(MachineError::InvalidRunEvent {
                reason: "event session_id does not match target run".to_string(),
            });
        }
        if terminal_event.seq() != seq {
            return Err(MachineError::InvalidRunEvent {
                reason: "event seq does not match allocated seq".to_string(),
            });
        }
        if !terminal_event.is_terminal() {
            return Err(MachineError::InvalidRunEvent {
                reason: "finish_run requires a terminal event".to_string(),
            });
        }

        let run_id = finish.run_id.clone();
        let finish = RunFinishRecord {
            run_id: finish.run_id,
            session_id: finish.session_id,
            status: finish.status,
            finish_reason: finish.finish_reason,
            error_code: finish.error_code,
            terminal_event,
            snapshot_json: finish.snapshot_json,
        };
        let result = self.store.finish_run(&finish).await?;
        self.registry
            .publish_terminal(&finish.run_id, result.terminal_event().clone());
        self.registry.remove(&run_id);
        drop(guard);
        self.remove_event_lock(&run_id).await;
        Ok(result)
    }

    pub async fn request_cancel(&self, run_id: &RunId) -> Result<Option<RunHandle>, MachineError> {
        let handle = self.registry.request_cancel(run_id);
        self.store.mark_cancelled(run_id).await?;
        Ok(handle)
    }

    pub async fn subscribe(
        &self,
        run_id: &RunId,
        scope: &Value,
        after_seq: i64,
    ) -> Result<RunSubscription<E>, MachineError> {
        let Some(lookup) = self.store.lookup_run(run_id, scope).await? else {
            return Ok(RunSubscription::Missing);
        };
        if lookup.status == RunStatus::Running
            && let Some(receiver) = self.registry.subscribe(run_id)
        {
            let replay = self.store.list_events(run_id, scope, after_seq).await?;
            let last_replay_seq = replay
                .last()
                .map(RunEvent::seq)
                .unwrap_or(after_seq)
                .max(after_seq);
            return Ok(RunSubscription::Active {
                replay,
                tail: RunTail::new(receiver, last_replay_seq),
            });
        }
        let lookup = if lookup.status == RunStatus::Running {
            self.store
                .lookup_run(run_id, scope)
                .await?
                .unwrap_or(lookup)
        } else {
            lookup
        };
        let replay = self.store.list_events(run_id, scope, after_seq).await?;
        Ok(RunSubscription::Inactive {
            status: lookup.status,
            replay,
        })
    }

    pub async fn find_idempotent_run(
        &self,
        scope: &Value,
        session_id: &crate::run::SessionId,
        key: &str,
    ) -> Result<Option<RunLookup>, MachineError> {
        self.store.find_idempotent_run(scope, session_id, key).await
    }
}

impl<P, S> RunLifecycle<RunEventEnvelope<P>, S>
where
    P: RunEventPayload,
    S: RunStore<RunEventEnvelope<P>>,
{
    pub async fn append_event(
        &self,
        run_id: &RunId,
        session_id: &SessionId,
        payload: P,
    ) -> Result<AppendEventResult<RunEventEnvelope<P>>, MachineError> {
        let event_run_id = run_id.clone();
        let event_session_id = session_id.clone();
        self.append_with(run_id, move |seq| {
            RunEventEnvelope::new(event_run_id, event_session_id, seq, payload)
        })
        .await
    }

    pub async fn finish_run(
        &self,
        finish: RunFinish,
        payload: P,
    ) -> Result<FinishRunResult<RunEventEnvelope<P>>, MachineError> {
        let event_run_id = finish.run_id.clone();
        let event_session_id = finish.session_id.clone();
        self.finish_with(finish, move |seq| {
            RunEventEnvelope::new(event_run_id, event_session_id, seq, payload)
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MemoryRunStore, RunStart, RunStore};
    use async_rt::sync::Notify;
    use async_trait::async_trait;
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
            agent_kind: "test".to_string(),
            model: None,
            client_run_key: key.map(str::to_string),
            parent_run_id: None,
            retry_of_run_id: None,
            scope: scope(),
            metadata: serde_json::json!({}),
        }
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
            status,
            finish_reason: "stop".to_string(),
            error_code: None,
            snapshot_json: None,
        }
    }

    #[test]
    fn lifecycle_appends_persists_and_publishes_event() {
        block_on(async {
            let lifecycle = lifecycle();
            let (sender, mut receiver) = mpsc::unbounded_channel();
            lifecycle
                .start_run(
                    run_start("run-a", None),
                    RunHandle::new("token".to_string()),
                    Some(sender),
                )
                .await
                .expect("start");

            let result = lifecycle
                .append_event(
                    &RunId::from("run-a"),
                    &SessionId::from("session-a"),
                    payload(false),
                )
                .await
                .expect("append");
            assert!(matches!(result, AppendEventResult::Recorded(_)));
            assert_eq!(receiver.try_recv().expect("published").seq, 1);

            let events = lifecycle
                .store()
                .list_events(&RunId::from("run-a"), &scope(), 0)
                .await
                .expect("events");
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].seq, 1);
        });
    }

    #[test]
    fn lifecycle_finishes_once_and_blocks_later_appends() {
        block_on(async {
            let lifecycle = lifecycle();
            lifecycle
                .start_run(
                    run_start("run-a", None),
                    RunHandle::new("token".to_string()),
                    None,
                )
                .await
                .expect("start");
            let terminal = lifecycle
                .finish_run(finish_request("run-a", RunStatus::Completed), payload(true))
                .await
                .expect("finish");
            assert!(terminal.is_finished());
            assert_eq!(terminal.terminal_event().seq, 1);
            assert_eq!(lifecycle.registry().len(), 0);
            let append = lifecycle
                .append_event(
                    &RunId::from("run-a"),
                    &SessionId::from("session-a"),
                    payload(false),
                )
                .await;
            assert!(matches!(append, Err(MachineError::RunNotFound)));
        });
    }

    #[test]
    fn lifecycle_terminal_seq_follows_last_appended_event() {
        block_on(async {
            let lifecycle = lifecycle();
            lifecycle
                .start_run(
                    run_start("run-a", None),
                    RunHandle::new("token".to_string()),
                    None,
                )
                .await
                .expect("start");
            lifecycle
                .append_event(
                    &RunId::from("run-a"),
                    &SessionId::from("session-a"),
                    payload(false),
                )
                .await
                .expect("append");

            let terminal = lifecycle
                .finish_run(finish_request("run-a", RunStatus::Completed), payload(true))
                .await
                .expect("finish");
            assert_eq!(terminal.terminal_event().seq, 2);
        });
    }

    #[test]
    fn lifecycle_terminal_releases_capacity() {
        block_on(async {
            let lifecycle = RunLifecycle::new(
                RunRegistry::new(),
                Arc::new(MemoryRunStore::<TestEvent>::new()),
                1,
            );
            lifecycle
                .start_run(
                    run_start("run-a", None),
                    RunHandle::new("token-a".to_string()),
                    None,
                )
                .await
                .expect("start first");
            lifecycle
                .finish_run(finish_request("run-a", RunStatus::Completed), payload(true))
                .await
                .expect("finish");

            assert!(matches!(
                lifecycle
                    .start_run(
                        run_start("run-b", None),
                        RunHandle::new("token-b".to_string()),
                        None,
                    )
                    .await
                    .expect("start second"),
                StartRunResult::Created
            ));
        });
    }

    #[test]
    fn lifecycle_repeated_finish_after_registry_cleanup_returns_existing_terminal() {
        block_on(async {
            let lifecycle = lifecycle();
            lifecycle
                .start_run(
                    run_start("run-a", None),
                    RunHandle::new("token".to_string()),
                    None,
                )
                .await
                .expect("start");
            let first = lifecycle
                .finish_run(finish_request("run-a", RunStatus::Completed), payload(true))
                .await
                .expect("first finish");
            let called = Arc::new(AtomicBool::new(false));

            let second = lifecycle
                .finish_with(finish_request("run-a", RunStatus::Error), {
                    let called = Arc::clone(&called);
                    move |seq| {
                        called.store(true, Ordering::SeqCst);
                        event("run-a", seq, true)
                    }
                })
                .await
                .expect("second finish");

            assert!(matches!(second, FinishRunResult::AlreadyFinished(_)));
            assert_eq!(second.terminal_event(), first.terminal_event());
            assert!(!called.load(Ordering::SeqCst));
        });
    }

    #[test]
    fn lifecycle_serializes_appends_for_one_run() {
        block_on(async {
            let store = Arc::new(BlockingRecordStore::new(1));
            let lifecycle = RunLifecycle::new(RunRegistry::new(), Arc::clone(&store), 8);
            lifecycle
                .start_run(
                    run_start("run-a", None),
                    RunHandle::new("token".to_string()),
                    None,
                )
                .await
                .expect("start");

            let first_lifecycle = lifecycle.clone();
            let first = async_rt::spawn(async move {
                first_lifecycle
                    .append_event(
                        &RunId::from("run-a"),
                        &SessionId::from("session-a"),
                        payload(false),
                    )
                    .await
            });
            async_rt::time::timeout(Duration::from_secs(1), store.blocked.notified())
                .await
                .expect("first append should block in store");

            let second_lifecycle = lifecycle.clone();
            let second = async_rt::spawn(async move {
                second_lifecycle
                    .append_event(
                        &RunId::from("run-a"),
                        &SessionId::from("session-a"),
                        payload(false),
                    )
                    .await
            });
            async_rt::time::sleep(Duration::from_millis(20)).await;
            assert!(
                store
                    .inner
                    .list_events(&RunId::from("run-a"), &scope(), 0)
                    .await
                    .expect("events")
                    .is_empty()
            );

            store.release.notify_one();
            first
                .await
                .expect("first task")
                .expect("first append should finish");
            second
                .await
                .expect("second task")
                .expect("second append should finish");

            let events = store
                .inner
                .list_events(&RunId::from("run-a"), &scope(), 0)
                .await
                .expect("events");
            assert_eq!(
                events.iter().map(|event| event.seq).collect::<Vec<_>>(),
                vec![1, 2]
            );
        });
    }

    #[test]
    fn lifecycle_subscribe_replays_then_tails_active_run() {
        block_on(async {
            let lifecycle = lifecycle();
            lifecycle
                .start_run(
                    run_start("run-a", None),
                    RunHandle::new("token".to_string()),
                    None,
                )
                .await
                .expect("start");
            lifecycle
                .append_event(
                    &RunId::from("run-a"),
                    &SessionId::from("session-a"),
                    payload(false),
                )
                .await
                .expect("append");

            let RunSubscription::Active { replay, mut tail } = lifecycle
                .subscribe(&RunId::from("run-a"), &scope(), 0)
                .await
                .expect("subscribe")
            else {
                panic!("expected active subscription");
            };
            assert_eq!(replay.len(), 1);
            assert_eq!(tail.cursor(), 1);

            lifecycle
                .append_event(
                    &RunId::from("run-a"),
                    &SessionId::from("session-a"),
                    payload(false),
                )
                .await
                .expect("append live");
            assert_eq!(tail.next_event().await.expect("live").seq, 2);
        });
    }

    #[test]
    fn run_tail_filters_events_at_or_before_cursor() {
        block_on(async {
            let (sender, receiver) = mpsc::unbounded_channel();
            let mut tail = RunTail::new(receiver, 1);
            sender.send(event("run-a", 1, false)).expect("send dup");
            sender.send(event("run-a", 2, false)).expect("send fresh");

            assert_eq!(tail.next_event().await.expect("fresh").seq, 2);
            assert_eq!(tail.cursor(), 2);
        });
    }

    #[test]
    fn lifecycle_subscribe_returns_terminal_replay_when_inactive() {
        block_on(async {
            let lifecycle = lifecycle();
            lifecycle
                .start_run(
                    run_start("run-a", None),
                    RunHandle::new("token".to_string()),
                    None,
                )
                .await
                .expect("start");
            lifecycle
                .finish_run(finish_request("run-a", RunStatus::Completed), payload(true))
                .await
                .expect("finish");

            let RunSubscription::Inactive { status, replay } = lifecycle
                .subscribe(&RunId::from("run-a"), &scope(), 0)
                .await
                .expect("subscribe")
            else {
                panic!("expected inactive subscription");
            };
            assert_eq!(status, RunStatus::Completed);
            assert_eq!(replay.len(), 1);
            assert!(replay[0].is_terminal());
        });
    }

    #[test]
    fn lifecycle_subscribe_reports_missing_run() {
        block_on(async {
            let lifecycle = lifecycle();
            assert!(matches!(
                lifecycle
                    .subscribe(&RunId::from("missing"), &scope(), 0)
                    .await
                    .expect("subscribe"),
                RunSubscription::Missing
            ));
        });
    }

    #[test]
    fn lifecycle_request_cancel_marks_registry_and_store() {
        block_on(async {
            let lifecycle = lifecycle();
            lifecycle
                .start_run(
                    run_start("run-a", None),
                    RunHandle::new("token".to_string()),
                    None,
                )
                .await
                .expect("start");
            let handle = lifecycle
                .request_cancel(&RunId::from("run-a"))
                .await
                .expect("cancel")
                .expect("active handle");
            assert!(handle.is_cancelled());
            assert!(
                lifecycle
                    .store()
                    .lookup_run(&RunId::from("run-a"), &scope())
                    .await
                    .expect("lookup")
                    .expect("run")
                    .cancel_requested
            );
        });
    }

    #[test]
    fn lifecycle_start_run_returns_idempotent_existing_without_registry_insert() {
        block_on(async {
            let lifecycle = lifecycle();
            assert!(matches!(
                lifecycle
                    .start_run(
                        run_start("run-a", Some("client-key")),
                        RunHandle::new("token-a".to_string()),
                        None,
                    )
                    .await
                    .expect("first"),
                StartRunResult::Created
            ));
            match lifecycle
                .start_run(
                    run_start("run-b", Some("client-key")),
                    RunHandle::new("token-b".to_string()),
                    None,
                )
                .await
                .expect("second")
            {
                StartRunResult::Existing(existing) => {
                    assert_eq!(existing.run_id, RunId::from("run-a"));
                }
                StartRunResult::Created => panic!("expected idempotent existing run"),
            }
            assert_eq!(lifecycle.registry().len(), 1);
        });
    }

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
        async fn ensure_session(
            &self,
            session_id: Option<SessionId>,
            scope: &Value,
        ) -> Result<SessionId, MachineError> {
            self.inner.ensure_session(session_id, scope).await
        }

        async fn start_run(&self, run: &RunStart) -> Result<StartRunResult, MachineError> {
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
            finish: &RunFinishRecord<TestEvent>,
        ) -> Result<FinishRunResult<TestEvent>, MachineError> {
            self.inner.finish_run(finish).await
        }

        async fn terminal_event(&self, run_id: &RunId) -> Result<Option<TestEvent>, MachineError> {
            self.inner.terminal_event(run_id).await
        }

        async fn find_idempotent_run(
            &self,
            scope: &Value,
            session_id: &SessionId,
            key: &str,
        ) -> Result<Option<RunLookup>, MachineError> {
            self.inner.find_idempotent_run(scope, session_id, key).await
        }

        async fn mark_cancelled(&self, run_id: &RunId) -> Result<(), MachineError> {
            self.inner.mark_cancelled(run_id).await
        }

        async fn record_event(
            &self,
            run_id: &RunId,
            event: &TestEvent,
        ) -> Result<bool, MachineError> {
            if event.seq() == self.block_seq {
                self.blocked.notify_one();
                self.release.notified().await;
            }
            self.inner.record_event(run_id, event).await
        }

        async fn list_events(
            &self,
            run_id: &RunId,
            scope: &Value,
            after_seq: i64,
        ) -> Result<Vec<TestEvent>, MachineError> {
            self.inner.list_events(run_id, scope, after_seq).await
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
}
