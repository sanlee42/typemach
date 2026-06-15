use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::*;
use crate::op::{Effect, EffectUpdate, EntryWrite, ItemWrite, RunOps};
use crate::store::{RunCommit, RunCommitResult};

#[derive(Debug, Default)]
pub(super) struct PendingOps {
    pub(super) effects: Vec<EffectUpdate>,
    pub(super) items: Vec<ItemWrite>,
    pub(super) entries: Vec<EntryWrite>,
}

pub(super) struct TxRunOps<S>
where
    S: RunTx<Event>,
{
    store: Arc<S>,
    run_id: RunId,
    session_id: SessionId,
    scope: S::Scope,
    lease: LeaseId,
    pending: async_rt::sync::Mutex<PendingOps>,
}

impl<S> TxRunOps<S>
where
    S: RunTx<Event>,
{
    pub(super) fn new(
        store: Arc<S>,
        run_id: RunId,
        session_id: SessionId,
        scope: S::Scope,
        lease: LeaseId,
    ) -> Self {
        Self {
            store,
            run_id,
            session_id,
            scope,
            lease,
            pending: async_rt::sync::Mutex::new(PendingOps::default()),
        }
    }

    pub(super) async fn take(&self) -> PendingOps {
        std::mem::take(&mut *self.pending.lock().await)
    }

    fn check_run(&self, run_id: &RunId) -> Result<(), MachineError> {
        if run_id == &self.run_id {
            return Ok(());
        }
        Err(MachineError::InvalidRunEvent {
            reason: "runtime operation target does not match run".to_string(),
        })
    }
}

#[async_trait]
impl<S> RunOps for TxRunOps<S>
where
    S: RunTx<Event> + 'static,
{
    async fn reserve(
        &self,
        run_id: &RunId,
        key: &str,
        kind: &str,
        request: Value,
    ) -> Result<Effect, MachineError> {
        self.check_run(run_id)?;
        self.store
            .reserve_effect(run_id, &self.scope, Some(&self.lease), key, kind, request)
            .await
    }

    async fn start(&self, run_id: &RunId, key: &str) -> Result<Effect, MachineError> {
        self.check_run(run_id)?;
        self.store
            .start_effect(run_id, &self.scope, Some(&self.lease), key)
            .await
    }

    async fn push_effect(&self, run_id: &RunId, update: EffectUpdate) -> Result<(), MachineError> {
        self.check_run(run_id)?;
        let mut pending = self.pending.lock().await;
        if let Some(existing) = pending
            .effects
            .iter()
            .find(|existing| existing.key == update.key)
        {
            if existing == &update {
                return Ok(());
            }
            return Err(MachineError::EffectConflict);
        }
        pending.effects.push(update);
        Ok(())
    }

    async fn push_item(&self, run_id: &RunId, item: ItemWrite) -> Result<(), MachineError> {
        self.check_run(run_id)?;
        let mut pending = self.pending.lock().await;
        if let Some(existing) = pending
            .items
            .iter()
            .find(|existing| existing.key == item.key)
        {
            if existing == &item {
                return Ok(());
            }
            return Err(MachineError::ItemConflict);
        }
        pending.items.push(item);
        Ok(())
    }

    async fn push_entry(&self, run_id: &RunId, entry: EntryWrite) -> Result<(), MachineError> {
        self.check_run(run_id)?;
        let mut pending = self.pending.lock().await;
        if let Some(existing) = pending
            .entries
            .iter()
            .find(|existing| existing.key == entry.key)
        {
            if existing == &entry {
                return Ok(());
            }
            return Err(MachineError::EntryConflict);
        }
        pending.entries.push(entry);
        Ok(())
    }

    async fn push_live_entry(&self, run_id: &RunId, entry: EntryWrite) -> Result<(), MachineError> {
        self.check_run(run_id)?;
        let commit = RunCommit {
            run_id: self.run_id.clone(),
            session_id: self.session_id.clone(),
            scope: self.scope.clone(),
            lease: Some(self.lease.clone()),
            checkpoint: None,
            events: Vec::new(),
            effects: Vec::new(),
            items: Vec::new(),
            entries: vec![entry],
            finish: None,
        };
        match self.store.commit_run(&commit).await? {
            RunCommitResult::Recorded(_) => Ok(()),
            RunCommitResult::Skipped => Err(MachineError::RunNotFound),
            RunCommitResult::Finished { .. } => Err(MachineError::InvalidRunEvent {
                reason: "live entry commit produced a terminal result".to_string(),
            }),
        }
    }
}
