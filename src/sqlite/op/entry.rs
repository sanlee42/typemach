use tokio_rusqlite::rusqlite::{OptionalExtension, Row, Transaction, params};

use super::{json_text, now_ms, store_db};
use crate::error::MachineError;
use crate::op::{Entry, EntryQuery, EntryWrite, Vis};
use crate::run::{RunId, SessionId, ThreadId};

pub(in crate::sqlite) fn apply_entries_tx(
    tx: &Transaction<'_>,
    scope_key: &str,
    run_id: &RunId,
    session_id: &SessionId,
    thread_id: &ThreadId,
    entries: &[EntryWrite],
) -> Result<(), MachineError> {
    if entries.is_empty() {
        return Ok(());
    }
    let _ = tx
        .query_row(
            "SELECT 1 FROM typemach_sessions WHERE scope_key = ?1 AND session_id = ?2",
            params![scope_key, session_id.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(store_db)?;
    check_entries_tx(tx, scope_key, session_id, entries)?;
    for entry in entries {
        if let Some(existing) = load_entry_tx(tx, scope_key, session_id, &entry.key)? {
            if existing.kind != entry.kind
                || existing.vis != entry.vis
                || existing.body != entry.body
            {
                return Err(MachineError::EntryConflict);
            }
            continue;
        }
        let seq = tx
            .query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1
                 FROM typemach_entries
                 WHERE scope_key = ?1 AND session_id = ?2",
                params![scope_key, session_id.as_str()],
                |row| row.get::<_, i64>(0),
            )
            .map_err(store_db)?;
        let now = now_ms();
        tx.execute(
            "INSERT INTO typemach_entries (
                scope_key, session_id, seq, run_id, thread_id, key, kind, vis, body, created_at, updated_at
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?10)",
            params![
                scope_key,
                session_id.as_str(),
                seq,
                run_id.as_str(),
                thread_id.as_str(),
                entry.key,
                entry.kind,
                entry.vis.as_str(),
                json_text(&entry.body)?,
                now,
            ],
        )
        .map_err(store_db)?;
    }
    Ok(())
}

pub(in crate::sqlite) fn check_entries_tx(
    tx: &Transaction<'_>,
    scope_key: &str,
    session_id: &SessionId,
    entries: &[EntryWrite],
) -> Result<(), MachineError> {
    for (index, entry) in entries.iter().enumerate() {
        if entries[..index].iter().any(|prev| prev.key == entry.key) {
            return Err(MachineError::EntryConflict);
        }
        if let Some(existing) = load_entry_tx(tx, scope_key, session_id, &entry.key)?
            && (existing.kind != entry.kind
                || existing.vis != entry.vis
                || existing.body != entry.body)
        {
            return Err(MachineError::EntryConflict);
        }
    }
    Ok(())
}

pub(in crate::sqlite) fn list_entries_tx(
    tx: &Transaction<'_>,
    query: EntryQuery<'_, str>,
) -> Result<Vec<Entry>, MachineError> {
    let thread = query.thread_id.map(ThreadId::as_str);
    let vis = query.vis.map(Vis::as_str);
    let mut stmt = tx
        .prepare(
            "SELECT run_id, thread_id, seq, key, kind, vis, body, created_at, updated_at
             FROM typemach_entries
             WHERE scope_key = ?1
               AND session_id = ?2
               AND (?3 IS NULL OR thread_id = ?3)
               AND (?4 IS NULL OR kind = ?4)
               AND (?5 IS NULL OR vis = ?5)
               AND seq > ?6
             ORDER BY seq ASC
             LIMIT ?7",
        )
        .map_err(store_db)?;
    let rows = stmt
        .query_map(
            params![
                query.scope,
                query.session_id.as_str(),
                thread,
                query.kind,
                vis,
                query.after_seq,
                query.limit.min(i64::MAX as usize) as i64,
            ],
            |row| row_entry(query.session_id, row),
        )
        .map_err(store_db)?;
    let mut entries = Vec::new();
    for row in rows {
        entries.push(row.map_err(store_db)??);
    }
    Ok(entries)
}

pub(in crate::sqlite) fn latest_entry_tx(
    tx: &Transaction<'_>,
    scope_key: &str,
    session_id: &SessionId,
    thread_id: Option<&ThreadId>,
    kind: &str,
    vis: Option<Vis>,
) -> Result<Option<Entry>, MachineError> {
    let thread = thread_id.map(ThreadId::as_str);
    let vis = vis.map(Vis::as_str);
    tx.query_row(
        "SELECT run_id, thread_id, seq, key, kind, vis, body, created_at, updated_at
         FROM typemach_entries
         WHERE scope_key = ?1
           AND session_id = ?2
           AND (?3 IS NULL OR thread_id = ?3)
           AND kind = ?4
           AND (?5 IS NULL OR vis = ?5)
         ORDER BY seq DESC
         LIMIT 1",
        params![scope_key, session_id.as_str(), thread, kind, vis],
        |row| row_entry(session_id, row),
    )
    .optional()
    .map_err(store_db)?
    .transpose()
}

fn load_entry_tx(
    tx: &Transaction<'_>,
    scope_key: &str,
    session_id: &SessionId,
    key: &str,
) -> Result<Option<Entry>, MachineError> {
    tx.query_row(
        "SELECT run_id, thread_id, seq, key, kind, vis, body, created_at, updated_at
         FROM typemach_entries
         WHERE scope_key = ?1 AND session_id = ?2 AND key = ?3",
        params![scope_key, session_id.as_str(), key],
        |row| row_entry(session_id, row),
    )
    .optional()
    .map_err(store_db)?
    .transpose()
}

fn row_entry(
    session_id: &SessionId,
    row: &Row<'_>,
) -> tokio_rusqlite::rusqlite::Result<Result<Entry, MachineError>> {
    let vis: String = row.get(5)?;
    let Some(vis) = Vis::parse(&vis) else {
        return Ok(Err(MachineError::InvalidRunEvent {
            reason: format!("invalid entry visibility: {vis}"),
        }));
    };
    let body_raw: String = row.get(6)?;
    let body = match serde_json::from_str(&body_raw) {
        Ok(body) => body,
        Err(err) => return Ok(Err(MachineError::Deserialization(err))),
    };
    Ok(Ok(Entry {
        run_id: RunId::from(row.get::<_, String>(0)?),
        session_id: session_id.clone(),
        thread_id: ThreadId::from(row.get::<_, String>(1)?),
        seq: row.get(2)?,
        key: row.get(3)?,
        kind: row.get(4)?,
        vis,
        body,
        created_at: row.get(7)?,
        updated_at: row.get(8)?,
    }))
}
