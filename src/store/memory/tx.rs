use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;

use super::op::*;
use super::*;
use crate::op::{EffectStatus, EntryQuery};
use crate::store::{FinishRunResult, RunTx};

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
        if commit.events.is_empty()
            && commit.effects.is_empty()
            && commit.items.is_empty()
            && commit.entries.is_empty()
        {
            return Err(MachineError::InvalidRunEvent {
                reason: "commit requires an event, effect, item, or entry".to_string(),
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
        validate_effect_updates(&inner, &commit.run_id, &commit.effects)?;
        validate_items(&inner, &commit.run_id, &commit.items)?;
        let scope_key = scope_key(&commit.scope)?;
        validate_entries(&inner, &scope_key, &commit.session_id, &commit.entries)?;
        let thread_id = run.start.thread_id.clone();
        if let Some(checkpoint) = &commit.checkpoint {
            inner
                .checkpoints
                .insert(checkpoint.thread_id.clone(), checkpoint.record.clone());
        }
        apply_effect_updates(&mut inner, &commit.run_id, &commit.effects)?;
        apply_items(&mut inner, &commit.run_id, &commit.items)?;
        apply_entries(
            &mut inner,
            &scope_key,
            &commit.run_id,
            &commit.session_id,
            &thread_id,
            &commit.entries,
        );
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

    async fn reserve_effect(
        &self,
        run_id: &RunId,
        scope: &Scope,
        lease: Option<&LeaseId>,
        key: &str,
        kind: &str,
        request: Value,
    ) -> Result<Effect, MachineError> {
        let mut inner = self.inner.lock().await;
        let run = inner.runs.get(run_id).ok_or(MachineError::RunNotFound)?;
        if run.start.scope != *scope {
            return Err(MachineError::RunNotFound);
        }
        if run.status.is_terminal() {
            return Err(MachineError::RunNotFound);
        }
        check_memory_lease(run, lease)?;
        let effect_key = (run_id.clone(), key.to_string());
        if let Some(existing) = inner.effects.get(&effect_key) {
            if existing.kind != kind || existing.request != request {
                return Err(MachineError::EffectConflict);
            }
            return Ok(existing.clone());
        }
        let now = now_ms();
        let effect = Effect {
            run_id: run_id.clone(),
            key: key.to_string(),
            kind: kind.to_string(),
            status: EffectStatus::Reserved,
            request,
            result: None,
            error_code: None,
            error_message: None,
            created_at: now,
            updated_at: now,
        };
        inner.effects.insert(effect_key, effect.clone());
        Ok(effect)
    }

    async fn start_effect(
        &self,
        run_id: &RunId,
        scope: &Scope,
        lease: Option<&LeaseId>,
        key: &str,
    ) -> Result<Effect, MachineError> {
        let mut inner = self.inner.lock().await;
        let run = inner.runs.get(run_id).ok_or(MachineError::RunNotFound)?;
        if run.start.scope != *scope {
            return Err(MachineError::RunNotFound);
        }
        if run.status.is_terminal() {
            return Err(MachineError::RunNotFound);
        }
        check_memory_lease(run, lease)?;
        let effect = inner
            .effects
            .get_mut(&(run_id.clone(), key.to_string()))
            .ok_or(MachineError::EffectNotFound)?;
        if effect.status.is_blocking() {
            return Err(MachineError::EffectPending);
        }
        if effect.status == EffectStatus::Done {
            return Ok(effect.clone());
        }
        effect.status = EffectStatus::Started;
        effect.result = None;
        effect.error_code = None;
        effect.error_message = None;
        effect.updated_at = now_ms();
        Ok(effect.clone())
    }

    async fn list_items(
        &self,
        run_id: &RunId,
        scope: &Scope,
        limit: usize,
    ) -> Result<Vec<Item>, MachineError> {
        if limit == 0 {
            return Err(MachineError::InvalidPageLimit);
        }
        let inner = self.inner.lock().await;
        let Some(run) = inner.runs.get(run_id) else {
            return Ok(Vec::new());
        };
        if run.start.scope != *scope {
            return Ok(Vec::new());
        }
        let mut items = inner
            .items
            .values()
            .filter(|item| item.run_id == *run_id)
            .cloned()
            .collect::<Vec<_>>();
        items.sort_by(|left, right| left.key.cmp(&right.key));
        items.truncate(limit);
        Ok(items)
    }

    async fn list_effects(
        &self,
        run_id: &RunId,
        scope: &Scope,
        limit: usize,
    ) -> Result<Vec<Effect>, MachineError> {
        if limit == 0 {
            return Err(MachineError::InvalidPageLimit);
        }
        let inner = self.inner.lock().await;
        let Some(run) = inner.runs.get(run_id) else {
            return Ok(Vec::new());
        };
        if run.start.scope != *scope {
            return Ok(Vec::new());
        }
        let mut effects = inner
            .effects
            .values()
            .filter(|effect| effect.run_id == *run_id)
            .cloned()
            .collect::<Vec<_>>();
        effects.sort_by(|left, right| left.key.cmp(&right.key));
        effects.truncate(limit);
        Ok(effects)
    }

    async fn list_entries(
        &self,
        query: EntryQuery<'_, Scope>,
    ) -> Result<Page<Entry>, MachineError> {
        if query.limit == 0 {
            return Err(MachineError::InvalidPageLimit);
        }
        let scope_key = scope_key(query.scope)?;
        let inner = self.inner.lock().await;
        let mut entries = inner
            .entries
            .iter()
            .filter(|((entry_scope, entry_session, _), entry)| {
                entry_scope == &scope_key
                    && entry_session == query.session_id
                    && entry.seq > query.after_seq
                    && query
                        .thread_id
                        .is_none_or(|thread_id| entry.thread_id == *thread_id)
                    && query.kind.is_none_or(|kind| entry.kind == kind)
                    && query.vis.is_none_or(|vis| entry.vis == vis)
            })
            .map(|(_, entry)| entry)
            .cloned()
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.seq);
        let next = if entries.len() > query.limit {
            entries.truncate(query.limit);
            entries.last().map(|entry| entry.seq)
        } else {
            None
        };
        Ok(Page::new(entries, next))
    }

    async fn record_entry(
        &self,
        run_id: &RunId,
        scope: &Scope,
        lease: Option<&LeaseId>,
        entry: EntryWrite,
    ) -> Result<Entry, MachineError> {
        let scope_key = scope_key(scope)?;
        let mut inner = self.inner.lock().await;
        let (session_id, thread_id) = {
            let run = inner.runs.get(run_id).ok_or(MachineError::RunNotFound)?;
            if run.start.scope != *scope {
                return Err(MachineError::RunNotFound);
            }
            if run.status.is_terminal() {
                return Err(MachineError::RunNotFound);
            }
            check_memory_lease(run, lease)?;
            (run.start.session_id.clone(), run.start.thread_id.clone())
        };
        validate_entries(
            &inner,
            &scope_key,
            &session_id,
            std::slice::from_ref(&entry),
        )?;
        let key = (scope_key.clone(), session_id.clone(), entry.key.clone());
        apply_entries(
            &mut inner,
            &scope_key,
            run_id,
            &session_id,
            &thread_id,
            std::slice::from_ref(&entry),
        );
        inner
            .entries
            .get(&key)
            .cloned()
            .ok_or(MachineError::RunNotFound)
    }

    async fn latest_entry(
        &self,
        scope: &Scope,
        session_id: &SessionId,
        thread_id: Option<&ThreadId>,
        kind: &str,
        vis: Option<Vis>,
    ) -> Result<Option<Entry>, MachineError> {
        let page = self
            .list_entries(EntryQuery {
                scope,
                session_id,
                thread_id,
                kind: Some(kind),
                vis,
                after_seq: 0,
                limit: usize::MAX,
            })
            .await?;
        Ok(page.items.into_iter().max_by_key(|entry| entry.seq))
    }
}
