use std::time::{SystemTime, UNIX_EPOCH};

use crate::op::{EffectStatus, EffectUpdate, Item, ItemWrite};

use super::*;

pub(super) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or_default()
}

pub(super) fn validate_effect_updates<E, Scope, FinishData>(
    inner: &MemoryRunStoreInner<E, Scope, FinishData>,
    run_id: &RunId,
    updates: &[EffectUpdate],
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    for (index, update) in updates.iter().enumerate() {
        if updates[..index].iter().any(|prev| prev.key == update.key) {
            return Err(MachineError::EffectConflict);
        }
        let Some(effect) = inner.effects.get(&(run_id.clone(), update.key.clone())) else {
            return Err(MachineError::EffectNotFound);
        };
        if matches!(
            update.status,
            EffectStatus::Reserved | EffectStatus::Started
        ) {
            return Err(MachineError::InvalidRunEvent {
                reason: "effect commit requires a terminal or unknown status".to_string(),
            });
        }
        if effect.status == EffectStatus::Done
            && (update.status != EffectStatus::Done || effect.result != update.result)
        {
            return Err(MachineError::EffectConflict);
        }
    }
    Ok(())
}

pub(super) fn validate_items<E, Scope, FinishData>(
    inner: &MemoryRunStoreInner<E, Scope, FinishData>,
    run_id: &RunId,
    items: &[ItemWrite],
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    for (index, item) in items.iter().enumerate() {
        if items[..index].iter().any(|prev| prev.key == item.key) {
            return Err(MachineError::ItemConflict);
        }
        if let Some(existing) = inner.items.get(&(run_id.clone(), item.key.clone()))
            && (existing.kind != item.kind || existing.body != item.body)
        {
            return Err(MachineError::ItemConflict);
        }
    }
    Ok(())
}

pub(super) fn apply_effect_updates<E, Scope, FinishData>(
    inner: &mut MemoryRunStoreInner<E, Scope, FinishData>,
    run_id: &RunId,
    updates: &[EffectUpdate],
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    let now = now_ms();
    for update in updates {
        let effect = inner
            .effects
            .get_mut(&(run_id.clone(), update.key.clone()))
            .ok_or(MachineError::EffectNotFound)?;
        if effect.status == EffectStatus::Done && update.status == EffectStatus::Done {
            continue;
        }
        effect.status = update.status.clone();
        effect.result = update.result.clone();
        effect.error_code = update.error_code.clone();
        effect.error_message = update.error_message.clone();
        effect.updated_at = now;
    }
    Ok(())
}

pub(super) fn apply_items<E, Scope, FinishData>(
    inner: &mut MemoryRunStoreInner<E, Scope, FinishData>,
    run_id: &RunId,
    items: &[ItemWrite],
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    let now = now_ms();
    for item in items {
        let key = (run_id.clone(), item.key.clone());
        if inner.items.contains_key(&key) {
            continue;
        }
        inner.items.insert(
            key,
            Item {
                run_id: run_id.clone(),
                key: item.key.clone(),
                kind: item.kind.clone(),
                body: item.body.clone(),
                created_at: now,
                updated_at: now,
            },
        );
    }
    Ok(())
}
