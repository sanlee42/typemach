use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_rt::sync::Mutex;
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;

use super::*;
use crate::checkpoint::{CheckpointRecord, CheckpointStore};

#[derive(Debug)]
pub struct MemoryRunStore<E: RunEvent, Scope = Value, FinishData = ()> {
    inner: Arc<Mutex<MemoryRunStoreInner<E, Scope, FinishData>>>,
}

impl<E: RunEvent, Scope, FinishData> Clone for MemoryRunStore<E, Scope, FinishData> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<E, Scope, FinishData> Default for MemoryRunStore<E, Scope, FinishData>
where
    E: RunEvent,
    Scope: Clone + PartialEq + Serialize + Send + Sync + 'static,
    FinishData: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<E, Scope, FinishData> MemoryRunStore<E, Scope, FinishData>
where
    E: RunEvent,
    Scope: Clone + PartialEq + Serialize + Send + Sync + 'static,
    FinishData: Clone + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MemoryRunStoreInner::default())),
        }
    }

    pub async fn finish_data(&self, run_id: &RunId) -> Option<FinishData> {
        self.inner
            .lock()
            .await
            .runs
            .get(run_id)
            .and_then(|run| run.finish_data.clone())
    }
}

#[derive(Debug)]
struct MemoryRunStoreInner<E: RunEvent, Scope, FinishData> {
    next_session: u64,
    sessions: HashMap<SessionId, Scope>,
    runs: HashMap<RunId, MemoryRun<E, Scope, FinishData>>,
    thread_leases: HashMap<ThreadId, MemoryThreadLease>,
    checkpoints: HashMap<ThreadId, CheckpointRecord>,
    idempotency: HashMap<(String, SessionId, String), RunId>,
}

impl<E: RunEvent, Scope, FinishData> Default for MemoryRunStoreInner<E, Scope, FinishData> {
    fn default() -> Self {
        Self {
            next_session: 0,
            sessions: HashMap::new(),
            runs: HashMap::new(),
            thread_leases: HashMap::new(),
            checkpoints: HashMap::new(),
            idempotency: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct MemoryRun<E: RunEvent, Scope, FinishData> {
    start: RunStart<Scope>,
    status: RunStatus,
    finish_reason: Option<String>,
    cancel_requested: bool,
    finish_data: Option<FinishData>,
    terminal_event: Option<E>,
    events: Vec<E>,
    lease: Option<MemoryLease>,
}

#[derive(Debug, Clone)]
struct MemoryLease {
    owner: WorkerId,
    id: LeaseId,
    until: Instant,
}

#[derive(Debug, Clone)]
struct MemoryThreadLease {
    run: RunId,
    owner: WorkerId,
    id: LeaseId,
    until: Instant,
}

#[async_trait]
impl<E, Scope, FinishData> RunStore<E> for MemoryRunStore<E, Scope, FinishData>
where
    E: RunEvent,
    Scope: Clone + PartialEq + Serialize + Send + Sync + 'static,
    FinishData: Clone + Send + Sync + 'static,
{
    type Scope = Scope;
    type FinishData = FinishData;

    async fn ensure_session(
        &self,
        session_id: Option<SessionId>,
        scope: &Scope,
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

    async fn start_run(&self, run: &RunStart<Scope>) -> Result<StoreStartResult, MachineError> {
        let mut inner = self.inner.lock().await;
        inner
            .sessions
            .entry(run.session_id.clone())
            .or_insert_with(|| run.scope.clone());
        if let Some(existing) = inner.runs.get(&run.run_id) {
            return Ok(StoreStartResult::Existing(run_lookup(existing)));
        }
        let idempotency_key = if let Some(client_key) = &run.client_run_key {
            let key = (
                scope_key(&run.scope)?,
                run.session_id.clone(),
                client_key.clone(),
            );
            if let Some(existing_run_id) = inner.idempotency.get(&key)
                && let Some(existing) = inner.runs.get(existing_run_id)
            {
                return Ok(StoreStartResult::Existing(run_lookup(existing)));
            }
            Some(key)
        } else {
            None
        };
        if let Some((owner, run_id)) = running_memory_thread(&inner, &run.thread_id, &run.run_id) {
            return Err(MachineError::ThreadBusy { owner, run: run_id });
        }
        if let Some(claim) = &run.lease {
            claim_memory_thread(&mut inner, run, claim)?;
        }
        if let Some(key) = idempotency_key {
            inner.idempotency.insert(key, run.run_id.clone());
        }
        inner.runs.insert(
            run.run_id.clone(),
            MemoryRun {
                start: run.clone(),
                status: RunStatus::Running,
                finish_reason: None,
                cancel_requested: false,
                finish_data: None,
                terminal_event: None,
                events: Vec::new(),
                lease: run.lease.as_ref().map(memory_lease),
            },
        );
        Ok(StoreStartResult::Created)
    }

    async fn lookup_run(
        &self,
        run_id: &RunId,
        scope: &Scope,
    ) -> Result<Option<RunLookup>, MachineError> {
        let inner = self.inner.lock().await;
        Ok(inner
            .runs
            .get(run_id)
            .filter(|run| run.start.scope == *scope)
            .map(run_lookup))
    }

    async fn finish_run(
        &self,
        finish: &RunFinishRecord<E, FinishData, Scope>,
    ) -> Result<FinishRunResult<E>, MachineError> {
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
        if finish.scope != run.start.scope {
            return Err(MachineError::RunNotFound);
        }
        if run.status.is_terminal() {
            let terminal_event = run
                .terminal_event
                .clone()
                .ok_or(MachineError::RunNotFound)?;
            return Ok(FinishRunResult::AlreadyFinished(terminal_event));
        }
        if run.lease.is_some() {
            return Err(MachineError::LeaseLost);
        }
        validate_next_seq(run, &finish.terminal_event)?;
        run.status = finish.status;
        run.finish_reason = Some(finish.finish_reason.clone());
        run.finish_data = Some(finish.data.clone());
        run.terminal_event = Some(finish.terminal_event.clone());
        run.events.push(finish.terminal_event.clone());
        run.lease = None;
        Ok(FinishRunResult::Finished(finish.terminal_event.clone()))
    }

    async fn terminal_event(
        &self,
        run_id: &RunId,
        scope: &Scope,
    ) -> Result<Option<E>, MachineError> {
        let inner = self.inner.lock().await;
        Ok(inner
            .runs
            .get(run_id)
            .filter(|run| run.start.scope == *scope)
            .and_then(|run| run.terminal_event.clone()))
    }

    async fn find_idempotent_run(
        &self,
        scope: &Scope,
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

    async fn mark_cancelled(&self, run_id: &RunId, scope: &Scope) -> Result<(), MachineError> {
        let mut inner = self.inner.lock().await;
        let run = inner
            .runs
            .get_mut(run_id)
            .ok_or(MachineError::RunNotFound)?;
        if run.start.scope != *scope {
            return Err(MachineError::RunNotFound);
        }
        run.cancel_requested = true;
        Ok(())
    }

    async fn record_event(
        &self,
        run_id: &RunId,
        scope: &Scope,
        event: &E,
    ) -> Result<bool, MachineError> {
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
        if run.start.scope != *scope {
            return Err(MachineError::RunNotFound);
        }
        if run.status.is_terminal() {
            return Ok(false);
        }
        if run.lease.is_some() {
            return Err(MachineError::LeaseLost);
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
        scope: &Scope,
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

#[async_trait]
impl<E, Scope, FinishData> CheckpointStore for MemoryRunStore<E, Scope, FinishData>
where
    E: RunEvent,
    Scope: Clone + PartialEq + Serialize + Send + Sync + 'static,
    FinishData: Clone + Send + Sync + 'static,
{
    async fn load_checkpoint(
        &self,
        thread_id: &str,
    ) -> Result<Option<CheckpointRecord>, MachineError> {
        Ok(self
            .inner
            .lock()
            .await
            .checkpoints
            .get(&ThreadId::from(thread_id))
            .cloned())
    }
}

#[async_trait]
impl<E, Scope, FinishData> RunTx<E> for MemoryRunStore<E, Scope, FinishData>
where
    E: RunEvent,
    Scope: Clone + PartialEq + Serialize + Send + Sync + 'static,
    FinishData: Clone + Send + Sync + 'static,
{
    async fn commit_run(
        &self,
        commit: &RunCommit<E, FinishData, Scope>,
    ) -> Result<RunCommitResult<E>, MachineError> {
        if commit.events.is_empty() {
            return Err(MachineError::InvalidRunEvent {
                reason: "commit requires at least one event".to_string(),
            });
        }
        validate_commit_events(commit)?;

        let mut inner = self.inner.lock().await;
        let Some(run) = inner.runs.get(&commit.run_id) else {
            return Ok(RunCommitResult::Skipped);
        };
        if run.start.scope != commit.scope {
            return Err(MachineError::RunNotFound);
        }
        if run.start.session_id != commit.session_id {
            return Err(MachineError::InvalidRunEvent {
                reason: "event session_id does not match target run".to_string(),
            });
        }
        if run.status.is_terminal() {
            if commit.finish.is_some()
                && let Some(event) = run.terminal_event.clone()
            {
                return Ok(RunCommitResult::Finished {
                    events: vec![event.clone()],
                    result: FinishRunResult::AlreadyFinished(event),
                });
            }
            return Ok(RunCommitResult::Skipped);
        }
        check_memory_lease(run, commit.lease.as_ref())?;
        if let Some(checkpoint) = &commit.checkpoint {
            check_memory_thread(&inner, run, checkpoint, commit.lease.as_ref())?;
        }
        validate_event_sequence(run, &commit.events)?;
        if let Some(checkpoint) = &commit.checkpoint {
            inner
                .checkpoints
                .insert(checkpoint.thread_id.clone(), checkpoint.record.clone());
        }
        let run = inner
            .runs
            .get_mut(&commit.run_id)
            .ok_or(MachineError::RunNotFound)?;

        for event in &commit.events {
            run.events.push(event.clone());
        }

        if let Some(finish) = &commit.finish {
            let terminal_event =
                commit
                    .events
                    .last()
                    .cloned()
                    .ok_or_else(|| MachineError::InvalidRunEvent {
                        reason: "finish commit requires a terminal event".to_string(),
                    })?;
            run.status = finish.status;
            run.finish_reason = Some(finish.finish_reason.clone());
            run.finish_data = Some(finish.data.clone());
            run.terminal_event = Some(terminal_event.clone());
            let thread_id = run.start.thread_id.clone();
            run.lease = None;
            inner.thread_leases.remove(&thread_id);
            return Ok(RunCommitResult::Finished {
                events: commit.events.clone(),
                result: FinishRunResult::Finished(terminal_event),
            });
        }

        Ok(RunCommitResult::Recorded(commit.events.clone()))
    }
}

#[async_trait]
impl<E, Scope, FinishData> RunLease<E> for MemoryRunStore<E, Scope, FinishData>
where
    E: RunEvent,
    Scope: Clone + PartialEq + Serialize + Send + Sync + 'static,
    FinishData: Clone + Default + Send + Sync + 'static,
{
    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<bool, MachineError> {
        let mut inner = self.inner.lock().await;
        let Some(run) = inner.runs.get(&lease.run) else {
            return Ok(false);
        };
        let Some(active) = &run.lease else {
            return Ok(false);
        };
        if run.status != RunStatus::Running
            || active.owner != lease.owner
            || active.id != lease.id
            || active.until <= Instant::now()
        {
            return Ok(false);
        }
        let thread_id = run.start.thread_id.clone();
        let Some(thread) = inner.thread_leases.get_mut(&thread_id) else {
            return Ok(false);
        };
        if thread.run != lease.run
            || thread.owner != lease.owner
            || thread.id != lease.id
            || thread.until <= Instant::now()
        {
            return Ok(false);
        }
        let until = Instant::now() + ttl;
        thread.until = until;
        let run = inner
            .runs
            .get_mut(&lease.run)
            .ok_or(MachineError::RunNotFound)?;
        if let Some(active) = &mut run.lease {
            active.until = until;
        }
        Ok(true)
    }

    async fn release(&self, lease: &Lease) -> Result<(), MachineError> {
        let mut inner = self.inner.lock().await;
        let mut thread = None;
        if let Some(run) = inner.runs.get_mut(&lease.run)
            && run
                .lease
                .as_ref()
                .is_some_and(|active| active.owner == lease.owner && active.id == lease.id)
        {
            thread = Some(run.start.thread_id.clone());
            run.lease = None;
        }
        if let Some(thread) = thread {
            inner.thread_leases.remove(&thread);
        }
        Ok(())
    }

    async fn reap_stale<F>(
        &self,
        _owner: &WorkerId,
        limit: usize,
        mut build_event: F,
    ) -> Result<Vec<RunLookup>, MachineError>
    where
        F: FnMut(&RunLookup, i64) -> E + Send,
    {
        let mut inner = self.inner.lock().await;
        let now = Instant::now();
        let mut reaped = Vec::new();
        let run_ids = inner.runs.keys().cloned().collect::<Vec<_>>();
        for run_id in run_ids {
            if reaped.len() >= limit {
                break;
            }
            let Some(run) = inner.runs.get_mut(&run_id) else {
                continue;
            };
            if run.status != RunStatus::Running
                || run.lease.as_ref().is_none_or(|lease| lease.until > now)
            {
                continue;
            }
            let mut lookup = run_lookup(run);
            let seq = run.events.last().map(RunEvent::seq).unwrap_or(0) + 1;
            let event = build_event(&lookup, seq);
            if event.run_id() != &run.start.run_id || event.session_id() != &run.start.session_id {
                return Err(MachineError::InvalidRunEvent {
                    reason: "reap event target does not match run".to_string(),
                });
            }
            if event.seq() != seq || !event.is_terminal() {
                return Err(MachineError::InvalidRunEvent {
                    reason: "reap requires the next terminal event".to_string(),
                });
            }
            run.status = RunStatus::Error;
            run.finish_reason = Some("lease_expired".to_string());
            run.finish_data = Some(FinishData::default());
            run.terminal_event = Some(event.clone());
            run.events.push(event);
            let thread_id = run.start.thread_id.clone();
            run.lease = None;
            inner.thread_leases.remove(&thread_id);
            lookup.status = RunStatus::Error;
            lookup.finish_reason = Some("lease_expired".to_string());
            lookup.owner = None;
            reaped.push(lookup);
        }
        Ok(reaped)
    }
}

fn run_lookup<E, Scope, FinishData>(run: &MemoryRun<E, Scope, FinishData>) -> RunLookup
where
    E: RunEvent,
{
    RunLookup {
        run_id: run.start.run_id.clone(),
        session_id: run.start.session_id.clone(),
        thread_id: run.start.thread_id.clone(),
        status: run.status,
        finish_reason: run.finish_reason.clone(),
        cancel_requested: run.cancel_requested,
        owner: run.lease.as_ref().map(|lease| lease.owner.clone()),
    }
}

fn running_memory_thread<E, Scope, FinishData>(
    inner: &MemoryRunStoreInner<E, Scope, FinishData>,
    thread_id: &ThreadId,
    except: &RunId,
) -> Option<(Option<WorkerId>, Option<RunId>)>
where
    E: RunEvent,
{
    inner
        .runs
        .values()
        .find(|run| {
            run.status == RunStatus::Running
                && run.start.thread_id == *thread_id
                && run.start.run_id != *except
        })
        .map(|run| {
            (
                run.lease.as_ref().map(|lease| lease.owner.clone()),
                Some(run.start.run_id.clone()),
            )
        })
}

fn memory_lease(claim: &LeaseClaim) -> MemoryLease {
    MemoryLease {
        owner: claim.owner.clone(),
        id: claim.id.clone(),
        until: Instant::now() + claim.ttl,
    }
}

fn claim_memory_thread<E, Scope, FinishData>(
    inner: &mut MemoryRunStoreInner<E, Scope, FinishData>,
    run: &RunStart<Scope>,
    claim: &LeaseClaim,
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    let now = Instant::now();
    if let Some(active) = inner.thread_leases.get(&run.thread_id)
        && active.until > now
    {
        return Err(MachineError::ThreadBusy {
            owner: Some(active.owner.clone()),
            run: Some(active.run.clone()),
        });
    }
    inner.thread_leases.insert(
        run.thread_id.clone(),
        MemoryThreadLease {
            run: run.run_id.clone(),
            owner: claim.owner.clone(),
            id: claim.id.clone(),
            until: now + claim.ttl,
        },
    );
    Ok(())
}

fn check_memory_lease<E, Scope, FinishData>(
    run: &MemoryRun<E, Scope, FinishData>,
    lease: Option<&LeaseId>,
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    let Some(active) = &run.lease else {
        return Ok(());
    };
    if lease != Some(&active.id) || active.until <= Instant::now() {
        return Err(MachineError::LeaseLost);
    }
    Ok(())
}

fn check_memory_thread<E, Scope, FinishData>(
    inner: &MemoryRunStoreInner<E, Scope, FinishData>,
    run: &MemoryRun<E, Scope, FinishData>,
    checkpoint: &CheckpointWrite,
    lease: Option<&LeaseId>,
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    if checkpoint.thread_id != run.start.thread_id {
        return Err(MachineError::InvalidRunEvent {
            reason: "checkpoint thread_id does not match target run".to_string(),
        });
    }
    if let Some((owner, run_id)) =
        running_memory_thread(inner, &run.start.thread_id, &run.start.run_id)
    {
        return Err(MachineError::ThreadBusy { owner, run: run_id });
    }
    let Some(active) = &run.lease else {
        if let Some(thread) = inner.thread_leases.get(&run.start.thread_id)
            && thread.until > Instant::now()
        {
            return Err(MachineError::ThreadBusy {
                owner: Some(thread.owner.clone()),
                run: Some(thread.run.clone()),
            });
        }
        return Ok(());
    };
    let Some(thread) = inner.thread_leases.get(&run.start.thread_id) else {
        return Err(MachineError::LeaseLost);
    };
    if thread.run != run.start.run_id
        || thread.owner != active.owner
        || thread.id != active.id
        || lease != Some(&thread.id)
        || thread.until <= Instant::now()
    {
        return Err(MachineError::LeaseLost);
    }
    Ok(())
}

mod validate;
use validate::*;
