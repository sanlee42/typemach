use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Serialize;

use super::*;

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

pub(super) fn memory_lease(claim: &LeaseClaim) -> MemoryLease {
    MemoryLease {
        owner: claim.owner.clone(),
        id: claim.id.clone(),
        until: Instant::now() + claim.ttl,
    }
}

pub(super) fn claim_memory_thread<E, Scope, FinishData>(
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

pub(super) fn check_memory_lease<E, Scope, FinishData>(
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
