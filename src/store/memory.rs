use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_rt::sync::Mutex;
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;

use super::*;
use crate::checkpoint::{CheckpointRecord, CheckpointStore};
mod tx;

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
    entry_seq: HashMap<(String, SessionId), i64>,
    entries: HashMap<(String, SessionId, String), Entry>,
    effects: HashMap<(RunId, String), Effect>,
    items: HashMap<(RunId, String), Item>,
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
            entry_seq: HashMap::new(),
            entries: HashMap::new(),
            effects: HashMap::new(),
            items: HashMap::new(),
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
        let scope_key = scope_key(&run.scope)?;
        let sig = start_sig(run)?;
        if let Some(existing) = inner.runs.get(&run.run_id) {
            if existing.start.scope != run.scope {
                return Err(MachineError::RunNotFound);
            }
            if start_sig(&existing.start)? != sig {
                return Err(MachineError::StartConflict);
            }
            return Ok(StoreStartResult::Existing(run_lookup(existing)));
        }
        let idempotency_key = if let Some(client_key) = &run.client_run_key {
            let key = (
                scope_key.clone(),
                run.session_id.clone(),
                client_key.clone(),
            );
            if let Some(existing_run_id) = inner.idempotency.get(&key)
                && let Some(existing) = inner.runs.get(existing_run_id)
            {
                if start_sig(&existing.start)? != sig {
                    return Err(MachineError::StartConflict);
                }
                return Ok(StoreStartResult::Existing(run_lookup(existing)));
            }
            Some(key)
        } else {
            None
        };
        validate_entries(&inner, &scope_key, &run.session_id, &run.entries)?;
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
        apply_entries(
            &mut inner,
            &scope_key,
            &run.run_id,
            &run.session_id,
            &run.thread_id,
            &run.entries,
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
            .get(&finish.run_id)
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
        let scope_key = scope_key(&finish.scope)?;
        validate_entries(&inner, &scope_key, &finish.session_id, &finish.entries)?;
        validate_next_seq(run, &finish.terminal_event)?;
        let thread_id = run.start.thread_id.clone();
        apply_entries(
            &mut inner,
            &scope_key,
            &finish.run_id,
            &finish.session_id,
            &thread_id,
            &finish.entries,
        );
        let run = inner
            .runs
            .get_mut(&finish.run_id)
            .ok_or(MachineError::RunNotFound)?;
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

    async fn check_run_start(
        &self,
        run_id: &RunId,
        start: &RunStart<Scope>,
    ) -> Result<(), MachineError> {
        let inner = self.inner.lock().await;
        let run = inner.runs.get(run_id).ok_or(MachineError::RunNotFound)?;
        if run.start.scope != start.scope {
            return Err(MachineError::RunNotFound);
        }
        if start_sig(&run.start)? != start_sig(start)? {
            return Err(MachineError::StartConflict);
        }
        Ok(())
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
        limit: usize,
    ) -> Result<Page<E>, MachineError> {
        if limit == 0 {
            return Err(MachineError::InvalidPageLimit);
        }
        let inner = self.inner.lock().await;
        let mut events = inner
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
            .unwrap_or_default();
        let next = if events.len() > limit {
            events.truncate(limit);
            events.last().map(RunEvent::seq)
        } else {
            None
        };
        Ok(Page::new(events, next))
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

mod lease;
mod op;
mod validate;
use lease::*;
use op::*;
use validate::*;
