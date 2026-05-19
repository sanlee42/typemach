use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use async_rt::sync::{Mutex, mpsc};
use serde::{Deserialize, Serialize};

use crate::error::MachineError;
use crate::registry::{RunHandle, RunRegistry};
use crate::run::{RunId, SessionId};
use crate::store::{
    CommitPlan, FinishRunResult, RunCommit, RunCommitResult, RunEvent, RunEventEnvelope,
    RunEventPayload, RunFinish, RunLookup, RunStart, RunStatus, RunStore, RunTx, StoreStartResult,
};

pub(crate) const REPLAY_LIMIT: usize = 1024;

#[derive(Debug)]
pub enum AppendEventResult<E: RunEvent> {
    Recorded(E),
    Skipped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartRunRejection {
    CapacityExceeded,
    RunAlreadyActive,
}

#[derive(Debug, Clone)]
pub enum StartRunResult {
    Started,
    Existing(RunLookup),
    NotRegistered(StartRunRejection),
}

#[derive(Debug)]
pub enum RunSubscription<E: RunEvent> {
    Active { replay: Vec<E>, tail: RunTail<E> },
    Replay { page: ReplayPage<E> },
    Inactive { status: RunStatus, replay: Vec<E> },
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunCursor(i64);

impl RunCursor {
    pub const START: Self = Self(0);

    pub fn new(seq: i64) -> Self {
        Self(seq)
    }

    pub fn as_i64(self) -> i64 {
        self.0
    }
}

impl Default for RunCursor {
    fn default() -> Self {
        Self::START
    }
}

impl From<i64> for RunCursor {
    fn from(seq: i64) -> Self {
        Self::new(seq)
    }
}

impl From<RunCursor> for i64 {
    fn from(cursor: RunCursor) -> Self {
        cursor.0
    }
}

#[derive(Debug, Clone)]
pub struct ReplayPage<E: RunEvent> {
    events: Vec<E>,
    cursor: RunCursor,
}

impl<E: RunEvent> ReplayPage<E> {
    fn new(events: Vec<E>, cursor: RunCursor) -> Self {
        Self { events, cursor }
    }

    pub fn events(&self) -> &[E] {
        &self.events
    }

    pub fn into_events(self) -> Vec<E> {
        self.events
    }

    pub fn cursor(&self) -> RunCursor {
        self.cursor
    }
}

impl<E: RunEvent> Deref for ReplayPage<E> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        &self.events
    }
}

#[derive(Debug)]
pub struct RunTail<E: RunEvent> {
    receiver: mpsc::UnboundedReceiver<E>,
    cursor: RunCursor,
}

impl<E: RunEvent> RunTail<E> {
    fn new(receiver: mpsc::UnboundedReceiver<E>, cursor: RunCursor) -> Self {
        Self { receiver, cursor }
    }

    pub fn cursor(&self) -> RunCursor {
        self.cursor
    }

    pub async fn next_event(&mut self) -> Option<E> {
        while let Some(event) = self.receiver.recv().await {
            if event.seq() <= self.cursor.as_i64() {
                continue;
            }
            self.cursor = RunCursor::new(event.seq());
            return Some(event);
        }
        None
    }
}

#[derive(Debug)]
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

impl<E, S> Clone for RunLifecycle<E, S>
where
    E: RunEvent,
    S: RunStore<E>,
{
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            store: Arc::clone(&self.store),
            event_locks: Arc::clone(&self.event_locks),
            max_in_flight: self.max_in_flight,
        }
    }
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
        scope: &S::Scope,
    ) -> Result<SessionId, MachineError> {
        self.store.ensure_session(session_id, scope).await
    }

    pub async fn start_run(
        &self,
        run: RunStart<S::Scope>,
        handle: RunHandle,
        stream_sender: Option<mpsc::UnboundedSender<E>>,
    ) -> Result<StartRunResult, MachineError> {
        let run_id = run.run_id.clone();
        if let Some(existing) = self.store.lookup_run(&run_id, &run.scope).await? {
            self.store.check_run_start(&existing.run_id, &run).await?;
            return Ok(StartRunResult::Existing(existing));
        }
        if let Some(client_run_key) = &run.client_run_key
            && let Some(existing) = self
                .store
                .find_idempotent_run(&run.scope, &run.session_id, client_run_key)
                .await?
        {
            self.store.check_run_start(&existing.run_id, &run).await?;
            return Ok(StartRunResult::Existing(existing));
        }

        match self
            .registry
            .try_insert(run_id.clone(), handle, stream_sender, self.max_in_flight)
        {
            Ok(()) => {}
            Err(MachineError::CapacityExceeded) => {
                return Ok(StartRunResult::NotRegistered(
                    StartRunRejection::CapacityExceeded,
                ));
            }
            Err(MachineError::RunAlreadyActive) => {
                return Ok(StartRunResult::NotRegistered(
                    StartRunRejection::RunAlreadyActive,
                ));
            }
            Err(err) => return Err(err),
        }

        match self.store.start_run(&run).await {
            Ok(StoreStartResult::Created) => Ok(StartRunResult::Started),
            Ok(StoreStartResult::Existing(existing)) => {
                self.registry.remove(&run_id);
                Ok(StartRunResult::Existing(existing))
            }
            Err(err) => {
                self.registry.remove(&run_id);
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

    async fn remove_event_lock(&self, run_id: &RunId, lock: &Arc<Mutex<()>>) {
        let mut locks = self.event_locks.lock().await;
        if locks
            .get(run_id)
            .is_some_and(|current| Arc::ptr_eq(current, lock) && Arc::strong_count(current) <= 2)
        {
            locks.remove(run_id);
        }
    }

    pub async fn append_with<F>(
        &self,
        run_id: &RunId,
        scope: &S::Scope,
        build_event: F,
    ) -> Result<AppendEventResult<E>, MachineError>
    where
        F: FnOnce(i64) -> E,
    {
        let lock = self.event_lock(run_id).await;
        let guard = lock.lock().await;
        if self.store.lookup_run(run_id, scope).await?.is_none() {
            let active = self.registry.handle(run_id).is_some();
            drop(guard);
            if !active {
                self.remove_event_lock(run_id, &lock).await;
            }
            return Err(MachineError::RunNotFound);
        }
        let Some(seq) = self.registry.next_seq(run_id) else {
            drop(guard);
            self.remove_event_lock(run_id, &lock).await;
            return Err(MachineError::RunNotFound);
        };
        let event = build_event(seq);
        if event.run_id() != run_id {
            self.registry.rewind_seq(run_id, seq);
            return Err(MachineError::InvalidRunEvent {
                reason: "event run_id does not match target run".to_string(),
            });
        }
        if event.seq() != seq {
            self.registry.rewind_seq(run_id, seq);
            return Err(MachineError::InvalidRunEvent {
                reason: "event seq does not match allocated seq".to_string(),
            });
        }
        if event.is_terminal() {
            self.registry.rewind_seq(run_id, seq);
            return Err(MachineError::InvalidRunEvent {
                reason: "append_event does not accept terminal events".to_string(),
            });
        }
        let recorded = self.store.record_event(run_id, scope, &event).await?;
        if recorded {
            self.registry.publish(run_id, event.clone());
            Ok(AppendEventResult::Recorded(event))
        } else {
            self.registry.remove(run_id);
            drop(guard);
            self.remove_event_lock(run_id, &lock).await;
            Ok(AppendEventResult::Skipped)
        }
    }

    pub async fn finish_with<F>(
        &self,
        finish: RunFinish<S::FinishData, S::Scope>,
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
        if self
            .store
            .lookup_run(&finish.run_id, &finish.scope)
            .await?
            .is_none()
        {
            let active = self.registry.handle(&finish.run_id).is_some();
            drop(guard);
            if !active {
                self.remove_event_lock(&finish.run_id, &lock).await;
            }
            return Err(MachineError::RunNotFound);
        }
        let Some(seq) = self.registry.next_seq(&finish.run_id) else {
            if let Some(terminal_event) = self
                .store
                .terminal_event(&finish.run_id, &finish.scope)
                .await?
            {
                drop(guard);
                self.remove_event_lock(&finish.run_id, &lock).await;
                return Ok(FinishRunResult::AlreadyFinished(terminal_event));
            }
            drop(guard);
            self.remove_event_lock(&finish.run_id, &lock).await;
            return Err(MachineError::RunNotFound);
        };

        let terminal_event = build_event(seq);
        if let Err(err) = validate_finish_event(&finish, &terminal_event, seq) {
            self.registry.rewind_seq(&finish.run_id, seq);
            drop(guard);
            return Err(err);
        }

        let run_id = finish.run_id.clone();
        let finish = finish.into_record(terminal_event);
        let result = self.store.finish_run(&finish).await?;
        self.registry
            .publish_terminal(&finish.run_id, result.terminal_event().clone());
        self.registry.remove(&run_id);
        drop(guard);
        self.remove_event_lock(&run_id, &lock).await;
        Ok(result)
    }

    pub async fn finish_detached_with<F>(
        &self,
        finish: RunFinish<S::FinishData, S::Scope>,
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
        if self
            .store
            .lookup_run(&finish.run_id, &finish.scope)
            .await?
            .is_none()
        {
            let active = self.registry.handle(&finish.run_id).is_some();
            drop(guard);
            if !active {
                self.remove_event_lock(&finish.run_id, &lock).await;
            }
            return Err(MachineError::RunNotFound);
        }
        if self.registry.handle(&finish.run_id).is_some() {
            drop(guard);
            return Err(MachineError::RunAlreadyActive);
        }
        if let Some(terminal_event) = self
            .store
            .terminal_event(&finish.run_id, &finish.scope)
            .await?
        {
            drop(guard);
            self.remove_event_lock(&finish.run_id, &lock).await;
            return Ok(FinishRunResult::AlreadyFinished(terminal_event));
        }

        let next_seq = self
            .store
            .list_events(&finish.run_id, &finish.scope, 0, usize::MAX)
            .await?
            .items
            .last()
            .map(RunEvent::seq)
            .unwrap_or(0)
            + 1;
        let terminal_event = build_event(next_seq);
        if let Err(err) = validate_finish_event(&finish, &terminal_event, next_seq) {
            drop(guard);
            return Err(err);
        }

        let run_id = finish.run_id.clone();
        let finish = finish.into_record(terminal_event);
        let result = self.store.finish_run(&finish).await?;
        drop(guard);
        self.remove_event_lock(&run_id, &lock).await;
        Ok(result)
    }

    pub async fn commit_with<F>(
        &self,
        run_id: &RunId,
        session_id: &SessionId,
        scope: &S::Scope,
        plan: CommitPlan<S::FinishData, S::Scope>,
        mut build_event: F,
    ) -> Result<RunCommitResult<E>, MachineError>
    where
        S: RunTx<E>,
        F: FnMut(i64) -> E,
    {
        if plan.event_count == 0
            && plan.effects.is_empty()
            && plan.items.is_empty()
            && plan.entries.is_empty()
        {
            return Err(MachineError::InvalidRunEvent {
                reason: "commit requires an event, effect, item, or entry".to_string(),
            });
        }
        if let Some(finish) = &plan.finish {
            if !finish.status.is_terminal() {
                return Err(MachineError::InvalidRunEvent {
                    reason: "finish_run requires a terminal status".to_string(),
                });
            }
            if finish.run_id != *run_id || finish.session_id != *session_id {
                return Err(MachineError::InvalidRunEvent {
                    reason: "finish target does not match committed run".to_string(),
                });
            }
        }

        let lock = self.event_lock(run_id).await;
        let guard = lock.lock().await;
        if self.store.lookup_run(run_id, scope).await?.is_none() {
            let active = self.registry.handle(run_id).is_some();
            drop(guard);
            if !active {
                self.remove_event_lock(run_id, &lock).await;
            }
            return Err(MachineError::RunNotFound);
        }

        let mut seqs = Vec::with_capacity(plan.event_count);
        for _ in 0..plan.event_count {
            let Some(seq) = self.registry.next_seq(run_id) else {
                if plan.finish.is_some()
                    && let Some(terminal_event) = self.store.terminal_event(run_id, scope).await?
                {
                    drop(guard);
                    self.remove_event_lock(run_id, &lock).await;
                    return Ok(RunCommitResult::Finished {
                        events: vec![terminal_event.clone()],
                        result: FinishRunResult::AlreadyFinished(terminal_event),
                    });
                }
                drop(guard);
                self.remove_event_lock(run_id, &lock).await;
                return Err(MachineError::RunNotFound);
            };
            seqs.push(seq);
        }

        let events = seqs.iter().map(|seq| build_event(*seq)).collect::<Vec<_>>();
        if let Err(err) =
            validate_commit_events(run_id, session_id, plan.finish.as_ref(), &events, &seqs)
        {
            rewind_seqs(&self.registry, run_id, &seqs);
            drop(guard);
            return Err(err);
        }

        let commit = RunCommit {
            run_id: run_id.clone(),
            session_id: session_id.clone(),
            scope: scope.clone(),
            lease: plan.lease,
            checkpoint: plan.checkpoint,
            events,
            effects: plan.effects,
            items: plan.items,
            entries: plan.entries,
            finish: plan.finish,
        };
        let result = match self.store.commit_run(&commit).await {
            Ok(result) => result,
            Err(err) => {
                rewind_seqs(&self.registry, run_id, &seqs);
                drop(guard);
                return Err(err);
            }
        };

        match &result {
            RunCommitResult::Recorded(events) => {
                for event in events {
                    self.registry.publish(run_id, event.clone());
                }
            }
            RunCommitResult::Finished { events, .. } => {
                for event in events {
                    if event.is_terminal() {
                        self.registry.publish_terminal(run_id, event.clone());
                    } else {
                        self.registry.publish(run_id, event.clone());
                    }
                }
                self.registry.remove(run_id);
            }
            RunCommitResult::Skipped => {
                self.registry.remove(run_id);
            }
        }
        drop(guard);
        if !matches!(result, RunCommitResult::Recorded(_)) {
            self.remove_event_lock(run_id, &lock).await;
        }
        Ok(result)
    }

    pub async fn request_cancel(
        &self,
        run_id: &RunId,
        scope: &S::Scope,
    ) -> Result<Option<RunHandle>, MachineError> {
        let Some(lookup) = self.store.lookup_run(run_id, scope).await? else {
            return Err(MachineError::RunNotFound);
        };
        if lookup.status.is_terminal() {
            return Ok(None);
        }
        let Some(handle) = self.registry.request_cancel(run_id) else {
            return Err(MachineError::NotOwner {
                owner: lookup.owner,
            });
        };
        self.store.mark_cancelled(run_id, scope).await?;
        Ok(Some(handle))
    }

    pub async fn subscribe(
        &self,
        run_id: &RunId,
        scope: &S::Scope,
        cursor: impl Into<RunCursor>,
    ) -> Result<RunSubscription<E>, MachineError> {
        let cursor = cursor.into();
        let Some(lookup) = self.store.lookup_run(run_id, scope).await? else {
            return Ok(RunSubscription::Missing);
        };
        if lookup.status == RunStatus::Running
            && let Some(receiver) = self.registry.subscribe(run_id)
        {
            let page = self
                .store
                .list_events(run_id, scope, cursor.as_i64(), REPLAY_LIMIT)
                .await?;
            if let Some(next) = page.next {
                return Ok(RunSubscription::Replay {
                    page: ReplayPage::new(page.items, RunCursor::new(next)),
                });
            }
            let replay = page.items;
            let last_replay_seq = replay
                .last()
                .map(RunEvent::seq)
                .unwrap_or(cursor.as_i64())
                .max(cursor.as_i64());
            return Ok(RunSubscription::Active {
                replay,
                tail: RunTail::new(receiver, RunCursor::new(last_replay_seq)),
            });
        }
        if lookup.status == RunStatus::Running {
            return Err(MachineError::NotOwner {
                owner: lookup.owner,
            });
        }
        let replay = self
            .store
            .list_events(run_id, scope, cursor.as_i64(), REPLAY_LIMIT)
            .await?;
        if let Some(next) = replay.next {
            return Ok(RunSubscription::Replay {
                page: ReplayPage::new(replay.items, RunCursor::new(next)),
            });
        }
        Ok(RunSubscription::Inactive {
            status: lookup.status,
            replay: replay.items,
        })
    }

    pub async fn find_idempotent_run(
        &self,
        scope: &S::Scope,
        session_id: &SessionId,
        key: &str,
    ) -> Result<Option<RunLookup>, MachineError> {
        self.store.find_idempotent_run(scope, session_id, key).await
    }
}

fn validate_finish_event<E, Data, Scope>(
    finish: &RunFinish<Data, Scope>,
    terminal_event: &E,
    seq: i64,
) -> Result<(), MachineError>
where
    E: RunEvent,
{
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
    Ok(())
}

fn validate_commit_events<E, Data, Scope>(
    run_id: &RunId,
    session_id: &SessionId,
    finish: Option<&RunFinish<Data, Scope>>,
    events: &[E],
    seqs: &[i64],
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    for (event, seq) in events.iter().zip(seqs) {
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
        if event.seq() != *seq {
            return Err(MachineError::InvalidRunEvent {
                reason: "event seq does not match allocated seq".to_string(),
            });
        }
    }

    match finish {
        Some(finish) => {
            let Some((last, seq)) = events.last().zip(seqs.last()) else {
                return Err(MachineError::InvalidRunEvent {
                    reason: "finish commit requires a terminal event".to_string(),
                });
            };
            for event in &events[..events.len().saturating_sub(1)] {
                if event.is_terminal() {
                    return Err(MachineError::InvalidRunEvent {
                        reason: "only the last commit event may be terminal".to_string(),
                    });
                }
            }
            validate_finish_event(finish, last, *seq)?;
        }
        None => {
            if events.iter().any(RunEvent::is_terminal) {
                return Err(MachineError::InvalidRunEvent {
                    reason: "non-finish commit does not accept terminal events".to_string(),
                });
            }
        }
    }
    Ok(())
}

fn rewind_seqs<E>(registry: &RunRegistry<E>, run_id: &RunId, seqs: &[i64]) {
    for seq in seqs.iter().rev() {
        registry.rewind_seq(run_id, *seq);
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
        scope: &S::Scope,
        payload: P,
    ) -> Result<AppendEventResult<RunEventEnvelope<P>>, MachineError> {
        let event_run_id = run_id.clone();
        let event_session_id = session_id.clone();
        self.append_with(run_id, scope, move |seq| {
            RunEventEnvelope::new(event_run_id, event_session_id, seq, payload)
        })
        .await
    }

    pub async fn finish_run(
        &self,
        finish: RunFinish<S::FinishData, S::Scope>,
        payload: P,
    ) -> Result<FinishRunResult<RunEventEnvelope<P>>, MachineError> {
        let event_run_id = finish.run_id.clone();
        let event_session_id = finish.session_id.clone();
        self.finish_with(finish, move |seq| {
            RunEventEnvelope::new(event_run_id, event_session_id, seq, payload)
        })
        .await
    }

    pub async fn finish_detached(
        &self,
        finish: RunFinish<S::FinishData, S::Scope>,
        payload: P,
    ) -> Result<FinishRunResult<RunEventEnvelope<P>>, MachineError> {
        let event_run_id = finish.run_id.clone();
        let event_session_id = finish.session_id.clone();
        self.finish_detached_with(finish, move |seq| {
            RunEventEnvelope::new(event_run_id, event_session_id, seq, payload)
        })
        .await
    }

    pub async fn commit_events(
        &self,
        run_id: &RunId,
        session_id: &SessionId,
        scope: &S::Scope,
        mut plan: CommitPlan<S::FinishData, S::Scope>,
        payloads: Vec<P>,
    ) -> Result<RunCommitResult<RunEventEnvelope<P>>, MachineError>
    where
        S: RunTx<RunEventEnvelope<P>>,
    {
        let event_count = payloads.len();
        plan.event_count = event_count;
        let event_run_id = run_id.clone();
        let event_session_id = session_id.clone();
        let mut payloads = payloads.into_iter();
        self.commit_with(run_id, session_id, scope, plan, move |seq| {
            RunEventEnvelope::new(
                event_run_id.clone(),
                event_session_id.clone(),
                seq,
                payloads.next().expect("payload count matches event count"),
            )
        })
        .await
    }
}

#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod lifecycle_tests;
