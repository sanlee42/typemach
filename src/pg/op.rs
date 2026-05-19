use deadpool_postgres::GenericClient;
use deadpool_postgres::tokio_postgres::Row;
use serde_json::Value;

use super::*;
use crate::op::{Effect, EffectStatus, EffectUpdate, Item, ItemWrite};
use crate::run::LeaseId;

mod entry;
pub(super) use entry::*;

pub(super) async fn check_op_run_tx<C>(
    tx: &C,
    run_id: &RunId,
    scope_key: &str,
    lease: Option<&LeaseId>,
) -> Result<(), MachineError>
where
    C: GenericClient + Sync,
{
    let row = tx
        .query_opt(
            "SELECT status,
                    lease_id,
                    lease_expires_at IS NOT NULL AND lease_expires_at <= now()
             FROM typemach_runs
             WHERE run_id = $1 AND scope_key = $2
             FOR UPDATE",
            &[&run_id.as_str(), &scope_key],
        )
        .await
        .map_err(store_db)?
        .ok_or(MachineError::RunNotFound)?;
    let status: String = row.get(0);
    let status = RunStatus::parse(&status).ok_or_else(|| MachineError::InvalidRunEvent {
        reason: format!("invalid stored run status: {status}"),
    })?;
    if status.is_terminal() {
        return Err(MachineError::RunNotFound);
    }
    let stored: Option<String> = row.get(1);
    let expired: bool = row.get(2);
    let Some(stored) = stored else {
        return Ok(());
    };
    if lease.map(LeaseId::as_str) != Some(stored.as_str()) || expired {
        return Err(MachineError::LeaseLost);
    }
    Ok(())
}

pub(super) async fn reserve_effect_tx<C>(
    tx: &C,
    run_id: &RunId,
    key: &str,
    kind: &str,
    request: Value,
) -> Result<Effect, MachineError>
where
    C: GenericClient + Sync,
{
    let request_json = json_text(&request)?;
    tx.execute(
        "INSERT INTO typemach_effects (run_id, key, kind, status, request, updated_at)
         VALUES ($1, $2, $3, $4, $5::text::jsonb, now())
         ON CONFLICT (run_id, key) DO NOTHING",
        &[
            &run_id.as_str(),
            &key,
            &kind,
            &EffectStatus::Reserved.as_str(),
            &request_json,
        ],
    )
    .await
    .map_err(store_db)?;
    let effect = load_effect_tx(tx, run_id, key)
        .await?
        .ok_or(MachineError::EffectNotFound)?;
    if effect.kind != kind || effect.request != request {
        return Err(MachineError::EffectConflict);
    }
    Ok(effect)
}

pub(super) async fn start_effect_tx<C>(
    tx: &C,
    run_id: &RunId,
    key: &str,
) -> Result<Effect, MachineError>
where
    C: GenericClient + Sync,
{
    let effect = load_effect_tx(tx, run_id, key)
        .await?
        .ok_or(MachineError::EffectNotFound)?;
    if effect.status.is_blocking() {
        return Err(MachineError::EffectPending);
    }
    if effect.status == EffectStatus::Done {
        return Ok(effect);
    }
    tx.execute(
        "UPDATE typemach_effects
         SET status = $3,
             result = NULL,
             error_code = NULL,
             error_message = NULL,
             updated_at = now()
         WHERE run_id = $1 AND key = $2",
        &[&run_id.as_str(), &key, &EffectStatus::Started.as_str()],
    )
    .await
    .map_err(store_db)?;
    load_effect_tx(tx, run_id, key)
        .await?
        .ok_or(MachineError::EffectNotFound)
}

pub(super) async fn validate_effect_updates_tx<C>(
    tx: &C,
    run_id: &RunId,
    updates: &[EffectUpdate],
) -> Result<(), MachineError>
where
    C: GenericClient + Sync,
{
    for (index, update) in updates.iter().enumerate() {
        if updates[..index].iter().any(|prev| prev.key == update.key) {
            return Err(MachineError::EffectConflict);
        }
        if matches!(
            update.status,
            EffectStatus::Reserved | EffectStatus::Started
        ) {
            return Err(MachineError::InvalidRunEvent {
                reason: "effect commit requires a terminal or unknown status".to_string(),
            });
        }
        let effect = load_effect_tx(tx, run_id, &update.key)
            .await?
            .ok_or(MachineError::EffectNotFound)?;
        if effect.status == EffectStatus::Done
            && (update.status != EffectStatus::Done || effect.result != update.result)
        {
            return Err(MachineError::EffectConflict);
        }
    }
    Ok(())
}

pub(super) async fn apply_effect_updates_tx<C>(
    tx: &C,
    run_id: &RunId,
    updates: &[EffectUpdate],
) -> Result<(), MachineError>
where
    C: GenericClient + Sync,
{
    for update in updates {
        tx.execute(
            "UPDATE typemach_effects
             SET status = $3,
                 result = $4::text::jsonb,
                 error_code = $5,
                 error_message = $6,
                 updated_at = now()
             WHERE run_id = $1 AND key = $2",
            &[
                &run_id.as_str(),
                &update.key,
                &update.status.as_str(),
                &update.result.as_ref().map(json_text).transpose()?,
                &update.error_code,
                &update.error_message,
            ],
        )
        .await
        .map_err(store_db)?;
    }
    Ok(())
}

pub(super) async fn validate_items_tx<C>(
    tx: &C,
    run_id: &RunId,
    items: &[ItemWrite],
) -> Result<(), MachineError>
where
    C: GenericClient + Sync,
{
    for (index, item) in items.iter().enumerate() {
        if items[..index].iter().any(|prev| prev.key == item.key) {
            return Err(MachineError::ItemConflict);
        }
        if let Some(existing) = load_item_tx(tx, run_id, &item.key).await?
            && (existing.kind != item.kind || existing.body != item.body)
        {
            return Err(MachineError::ItemConflict);
        }
    }
    Ok(())
}

pub(super) async fn apply_items_tx<C>(
    tx: &C,
    run_id: &RunId,
    items: &[ItemWrite],
) -> Result<(), MachineError>
where
    C: GenericClient + Sync,
{
    for item in items {
        let body = json_text(&item.body)?;
        tx.execute(
            "INSERT INTO typemach_items (run_id, key, kind, body, updated_at)
             VALUES ($1, $2, $3, $4::text::jsonb, now())
             ON CONFLICT (run_id, key) DO NOTHING",
            &[&run_id.as_str(), &item.key, &item.kind, &body],
        )
        .await
        .map_err(store_db)?;
    }
    Ok(())
}

pub(super) async fn list_items_tx<C>(
    tx: &C,
    run_id: &RunId,
    limit: usize,
) -> Result<Vec<Item>, MachineError>
where
    C: GenericClient + Sync,
{
    let limit = limit.min(i64::MAX as usize) as i64;
    let rows = tx
        .query(
            "SELECT key,
                    kind,
                    body::text,
                    (EXTRACT(EPOCH FROM created_at) * 1000)::bigint,
                    (EXTRACT(EPOCH FROM updated_at) * 1000)::bigint
             FROM typemach_items
             WHERE run_id = $1
             ORDER BY key ASC
             LIMIT $2",
            &[&run_id.as_str(), &limit],
        )
        .await
        .map_err(store_db)?;
    rows.into_iter().map(|row| row_item(run_id, &row)).collect()
}

pub(super) async fn list_effects_tx<C>(
    tx: &C,
    run_id: &RunId,
    limit: usize,
) -> Result<Vec<Effect>, MachineError>
where
    C: GenericClient + Sync,
{
    let limit = limit.min(i64::MAX as usize) as i64;
    let rows = tx
        .query(
            "SELECT key,
                    kind,
                    status,
                    request::text,
                    result::text,
                    error_code,
                    error_message,
                    (EXTRACT(EPOCH FROM created_at) * 1000)::bigint,
                    (EXTRACT(EPOCH FROM updated_at) * 1000)::bigint
             FROM typemach_effects
             WHERE run_id = $1
             ORDER BY key ASC
             LIMIT $2",
            &[&run_id.as_str(), &limit],
        )
        .await
        .map_err(store_db)?;
    rows.into_iter()
        .map(|row| row_effect(run_id, &row))
        .collect()
}

async fn load_effect_tx<C>(
    tx: &C,
    run_id: &RunId,
    key: &str,
) -> Result<Option<Effect>, MachineError>
where
    C: GenericClient + Sync,
{
    let row = tx
        .query_opt(
            "SELECT key,
                    kind,
                    status,
                    request::text,
                    result::text,
                    error_code,
                    error_message,
                    (EXTRACT(EPOCH FROM created_at) * 1000)::bigint,
                    (EXTRACT(EPOCH FROM updated_at) * 1000)::bigint
             FROM typemach_effects
             WHERE run_id = $1 AND key = $2
             FOR UPDATE",
            &[&run_id.as_str(), &key],
        )
        .await
        .map_err(store_db)?;
    row.map(|row| row_effect(run_id, &row)).transpose()
}

async fn load_item_tx<C>(tx: &C, run_id: &RunId, key: &str) -> Result<Option<Item>, MachineError>
where
    C: GenericClient + Sync,
{
    let row = tx
        .query_opt(
            "SELECT key,
                    kind,
                    body::text,
                    (EXTRACT(EPOCH FROM created_at) * 1000)::bigint,
                    (EXTRACT(EPOCH FROM updated_at) * 1000)::bigint
             FROM typemach_items
             WHERE run_id = $1 AND key = $2
             FOR UPDATE",
            &[&run_id.as_str(), &key],
        )
        .await
        .map_err(store_db)?;
    row.map(|row| row_item(run_id, &row)).transpose()
}

fn row_effect(run_id: &RunId, row: &Row) -> Result<Effect, MachineError> {
    let status: String = row.get(2);
    Ok(Effect {
        run_id: run_id.clone(),
        key: row.get(0),
        kind: row.get(1),
        status: EffectStatus::parse(&status).ok_or_else(|| MachineError::InvalidRunEvent {
            reason: format!("invalid effect status: {status}"),
        })?,
        request: serde_json::from_str(&row.get::<_, String>(3))
            .map_err(MachineError::Deserialization)?,
        result: row
            .get::<_, Option<String>>(4)
            .map(|raw| serde_json::from_str(raw.as_str()))
            .transpose()
            .map_err(MachineError::Deserialization)?,
        error_code: row.get(5),
        error_message: row.get(6),
        created_at: row.get(7),
        updated_at: row.get(8),
    })
}

fn row_item(run_id: &RunId, row: &Row) -> Result<Item, MachineError> {
    Ok(Item {
        run_id: run_id.clone(),
        key: row.get(0),
        kind: row.get(1),
        body: serde_json::from_str(&row.get::<_, String>(2))
            .map_err(MachineError::Deserialization)?,
        created_at: row.get(3),
        updated_at: row.get(4),
    })
}
