use tokio_rusqlite::rusqlite::{OptionalExtension, Row, Transaction, params};

use super::*;
use crate::op::{Effect, EffectStatus, EffectUpdate, Item, ItemWrite};
use crate::run::LeaseId;

mod entry;
pub(super) use entry::*;

pub(super) fn check_op_run_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    scope_key: &str,
    lease: Option<&LeaseId>,
) -> Result<(), MachineError> {
    let row = tx
        .query_row(
            "SELECT status, lease_id, lease_expires_at
             FROM typemach_runs
             WHERE run_id = ?1 AND scope_key = ?2",
            params![run_id.as_str(), scope_key],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                ))
            },
        )
        .optional()
        .map_err(store_db)?
        .ok_or(MachineError::RunNotFound)?;
    let status = RunStatus::parse(&row.0).ok_or_else(|| MachineError::InvalidRunEvent {
        reason: format!("invalid stored run status: {}", row.0),
    })?;
    if status.is_terminal() {
        return Err(MachineError::RunNotFound);
    }
    let Some(stored) = row.1 else {
        return Ok(());
    };
    if lease.map(LeaseId::as_str) != Some(stored.as_str())
        || row.2.is_some_and(|expires| expires <= now_ms())
    {
        return Err(MachineError::LeaseLost);
    }
    Ok(())
}

pub(super) fn reserve_effect_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    key: &str,
    kind: &str,
    request: serde_json::Value,
) -> Result<Effect, MachineError> {
    let request_json = json_text(&request)?;
    let now = now_ms();
    tx.execute(
        "INSERT OR IGNORE INTO typemach_effects
            (run_id, key, kind, status, request, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
        params![
            run_id.as_str(),
            key,
            kind,
            EffectStatus::Reserved.as_str(),
            request_json,
            now,
        ],
    )
    .map_err(store_db)?;
    let effect = load_effect_tx(tx, run_id, key)?.ok_or(MachineError::EffectNotFound)?;
    if effect.kind != kind || effect.request != request {
        return Err(MachineError::EffectConflict);
    }
    Ok(effect)
}

pub(super) fn start_effect_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    key: &str,
) -> Result<Effect, MachineError> {
    let effect = load_effect_tx(tx, run_id, key)?.ok_or(MachineError::EffectNotFound)?;
    if effect.status.is_blocking() {
        return Err(MachineError::EffectPending);
    }
    if effect.status == EffectStatus::Done {
        return Ok(effect);
    }
    tx.execute(
        "UPDATE typemach_effects
         SET status = ?3,
             result = NULL,
             error_code = NULL,
             error_message = NULL,
             updated_at = ?4
         WHERE run_id = ?1 AND key = ?2",
        params![
            run_id.as_str(),
            key,
            EffectStatus::Started.as_str(),
            now_ms()
        ],
    )
    .map_err(store_db)?;
    load_effect_tx(tx, run_id, key)?.ok_or(MachineError::EffectNotFound)
}

pub(super) fn validate_effect_updates_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    updates: &[EffectUpdate],
) -> Result<(), MachineError> {
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
        let effect =
            load_effect_tx(tx, run_id, &update.key)?.ok_or(MachineError::EffectNotFound)?;
        if effect.status == EffectStatus::Done
            && (update.status != EffectStatus::Done || effect.result != update.result)
        {
            return Err(MachineError::EffectConflict);
        }
    }
    Ok(())
}

pub(super) fn apply_effect_updates_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    updates: &[EffectUpdate],
) -> Result<(), MachineError> {
    for update in updates {
        tx.execute(
            "UPDATE typemach_effects
             SET status = ?3,
                 result = ?4,
                 error_code = ?5,
                 error_message = ?6,
                 updated_at = ?7
             WHERE run_id = ?1 AND key = ?2",
            params![
                run_id.as_str(),
                update.key,
                update.status.as_str(),
                update.result.as_ref().map(json_text).transpose()?,
                update.error_code,
                update.error_message,
                now_ms(),
            ],
        )
        .map_err(store_db)?;
    }
    Ok(())
}

pub(super) fn validate_items_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    items: &[ItemWrite],
) -> Result<(), MachineError> {
    for (index, item) in items.iter().enumerate() {
        if items[..index].iter().any(|prev| prev.key == item.key) {
            return Err(MachineError::ItemConflict);
        }
        if let Some(existing) = load_item_tx(tx, run_id, &item.key)?
            && (existing.kind != item.kind || existing.body != item.body)
        {
            return Err(MachineError::ItemConflict);
        }
    }
    Ok(())
}

pub(super) fn apply_items_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    items: &[ItemWrite],
) -> Result<(), MachineError> {
    for item in items {
        let now = now_ms();
        tx.execute(
            "INSERT OR IGNORE INTO typemach_items
                (run_id, key, kind, body, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
            params![
                run_id.as_str(),
                item.key,
                item.kind,
                json_text(&item.body)?,
                now
            ],
        )
        .map_err(store_db)?;
    }
    Ok(())
}

pub(super) fn list_items_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    limit: usize,
) -> Result<Vec<Item>, MachineError> {
    let mut stmt = tx
        .prepare(
            "SELECT key, kind, body, created_at, updated_at
             FROM typemach_items
             WHERE run_id = ?1
             ORDER BY key ASC
             LIMIT ?2",
        )
        .map_err(store_db)?;
    let rows = stmt
        .query_map(
            params![run_id.as_str(), limit.min(i64::MAX as usize) as i64],
            |row| row_item(run_id, row),
        )
        .map_err(store_db)?;
    let mut items = Vec::new();
    for row in rows {
        items.push(row.map_err(store_db)??);
    }
    Ok(items)
}

pub(super) fn list_effects_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    limit: usize,
) -> Result<Vec<Effect>, MachineError> {
    let mut stmt = tx
        .prepare(
            "SELECT key,
                    kind,
                    status,
                    request,
                    result,
                    error_code,
                    error_message,
                    created_at,
                    updated_at
             FROM typemach_effects
             WHERE run_id = ?1
             ORDER BY key ASC
             LIMIT ?2",
        )
        .map_err(store_db)?;
    let rows = stmt
        .query_map(
            params![run_id.as_str(), limit.min(i64::MAX as usize) as i64],
            |row| row_effect(run_id, row),
        )
        .map_err(store_db)?;
    let mut effects = Vec::new();
    for row in rows {
        effects.push(row.map_err(store_db)??);
    }
    Ok(effects)
}

fn load_effect_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    key: &str,
) -> Result<Option<Effect>, MachineError> {
    tx.query_row(
        "SELECT key,
                kind,
                status,
                request,
                result,
                error_code,
                error_message,
                created_at,
                updated_at
         FROM typemach_effects
         WHERE run_id = ?1 AND key = ?2",
        params![run_id.as_str(), key],
        |row| row_effect(run_id, row),
    )
    .optional()
    .map_err(store_db)?
    .transpose()
}

fn load_item_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    key: &str,
) -> Result<Option<Item>, MachineError> {
    tx.query_row(
        "SELECT key, kind, body, created_at, updated_at
         FROM typemach_items
         WHERE run_id = ?1 AND key = ?2",
        params![run_id.as_str(), key],
        |row| row_item(run_id, row),
    )
    .optional()
    .map_err(store_db)?
    .transpose()
}

fn row_effect(
    run_id: &RunId,
    row: &Row<'_>,
) -> tokio_rusqlite::rusqlite::Result<Result<Effect, MachineError>> {
    let status: String = row.get(2)?;
    let Some(status) = EffectStatus::parse(&status) else {
        return Ok(Err(MachineError::InvalidRunEvent {
            reason: format!("invalid effect status: {status}"),
        }));
    };
    let request_raw: String = row.get(3)?;
    let request = match serde_json::from_str(&request_raw) {
        Ok(request) => request,
        Err(err) => return Ok(Err(MachineError::Deserialization(err))),
    };
    let result_raw: Option<String> = row.get(4)?;
    let result = match result_raw
        .map(|raw| serde_json::from_str(raw.as_str()))
        .transpose()
    {
        Ok(result) => result,
        Err(err) => return Ok(Err(MachineError::Deserialization(err))),
    };
    Ok(Ok(Effect {
        run_id: run_id.clone(),
        key: row.get(0)?,
        kind: row.get(1)?,
        status,
        request,
        result,
        error_code: row.get(5)?,
        error_message: row.get(6)?,
        created_at: row.get(7)?,
        updated_at: row.get(8)?,
    }))
}

fn row_item(
    run_id: &RunId,
    row: &Row<'_>,
) -> tokio_rusqlite::rusqlite::Result<Result<Item, MachineError>> {
    let body_raw: String = row.get(2)?;
    let body = match serde_json::from_str(&body_raw) {
        Ok(body) => body,
        Err(err) => return Ok(Err(MachineError::Deserialization(err))),
    };
    Ok(Ok(Item {
        run_id: run_id.clone(),
        key: row.get(0)?,
        kind: row.get(1)?,
        body,
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
    }))
}
