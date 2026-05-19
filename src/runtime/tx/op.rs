use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::*;
use crate::op::{Effect, EffectUpdate, ItemWrite, RunOps};

#[derive(Debug, Default)]
pub(super) struct PendingOps {
    pub(super) effects: Vec<EffectUpdate>,
    pub(super) items: Vec<ItemWrite>,
}

pub(super) struct TxRunOps<S>
where
    S: RunTx<Event>,
{
    store: Arc<S>,
    run_id: RunId,
    scope: S::Scope,
    lease: LeaseId,
    pending: async_rt::sync::Mutex<PendingOps>,
}

impl<S> TxRunOps<S>
where
    S: RunTx<Event>,
{
    pub(super) fn new(store: Arc<S>, run_id: RunId, scope: S::Scope, lease: LeaseId) -> Self {
        Self {
            store,
            run_id,
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
}
