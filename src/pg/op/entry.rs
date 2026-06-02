use deadpool_postgres::GenericClient;
use deadpool_postgres::tokio_postgres::Row;

use super::{json_text, store_db};
use crate::error::MachineError;
use crate::op::{Entry, EntryQuery, EntryWrite, Vis};
use crate::run::{RunId, SessionId, ThreadId};

pub(in crate::pg) async fn apply_entries_tx<C>(
    tx: &C,
    scope_key: &str,
    run_id: &RunId,
    session_id: &SessionId,
    thread_id: &ThreadId,
    entries: &[EntryWrite],
) -> Result<(), MachineError>
where
    C: GenericClient + Sync,
{
    if entries.is_empty() {
        return Ok(());
    }
    tx.query_one(
        "SELECT 1 FROM typemach_sessions
         WHERE scope_key = $1 AND session_id = $2
         FOR UPDATE",
        &[&scope_key, &session_id.as_str()],
    )
    .await
    .map_err(store_db)?;
    check_entries_tx(tx, scope_key, session_id, entries).await?;

    for entry in entries {
        if let Some(existing) = load_entry_tx(tx, scope_key, session_id, &entry.key).await? {
            if existing.kind != entry.kind
                || existing.vis != entry.vis
                || existing.body != entry.body
            {
                return Err(MachineError::EntryConflict);
            }
            continue;
        }
        let row = tx
            .query_one(
                "SELECT COALESCE(MAX(seq), 0) + 1
                 FROM typemach_entries
                 WHERE scope_key = $1 AND session_id = $2",
                &[&scope_key, &session_id.as_str()],
            )
            .await
            .map_err(store_db)?;
        let seq: i64 = row.get(0);
        let body = json_text(&entry.body)?;
        tx.execute(
            "INSERT INTO typemach_entries (
                scope_key, session_id, seq, run_id, thread_id, key, kind, vis, body, updated_at
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::text::jsonb, now())",
            &[
                &scope_key,
                &session_id.as_str(),
                &seq,
                &run_id.as_str(),
                &thread_id.as_str(),
                &entry.key,
                &entry.kind,
                &entry.vis.as_str(),
                &body,
            ],
        )
        .await
        .map_err(store_db)?;
    }
    Ok(())
}

pub(in crate::pg) async fn record_entry_tx<C>(
    tx: &C,
    scope_key: &str,
    run_id: &RunId,
    session_id: &SessionId,
    thread_id: &ThreadId,
    entry: &EntryWrite,
) -> Result<Entry, MachineError>
where
    C: GenericClient + Sync,
{
    apply_entries_tx(
        tx,
        scope_key,
        run_id,
        session_id,
        thread_id,
        std::slice::from_ref(entry),
    )
    .await?;
    load_entry_tx(tx, scope_key, session_id, &entry.key)
        .await?
        .ok_or(MachineError::RunNotFound)
}

pub(in crate::pg) async fn check_entries_tx<C>(
    tx: &C,
    scope_key: &str,
    session_id: &SessionId,
    entries: &[EntryWrite],
) -> Result<(), MachineError>
where
    C: GenericClient + Sync,
{
    for (index, entry) in entries.iter().enumerate() {
        if entries[..index].iter().any(|prev| prev.key == entry.key) {
            return Err(MachineError::EntryConflict);
        }
        if let Some(existing) = load_entry_tx(tx, scope_key, session_id, &entry.key).await?
            && (existing.kind != entry.kind
                || existing.vis != entry.vis
                || existing.body != entry.body)
        {
            return Err(MachineError::EntryConflict);
        }
    }
    Ok(())
}

pub(in crate::pg) async fn list_entries_tx<C>(
    tx: &C,
    query: EntryQuery<'_, str>,
) -> Result<Vec<Entry>, MachineError>
where
    C: GenericClient + Sync,
{
    let thread = query.thread_id.map(ThreadId::as_str);
    let vis = query.vis.map(Vis::as_str);
    let limit = query.limit.min(i64::MAX as usize) as i64;
    let rows = tx
        .query(
            "SELECT run_id,
                    thread_id,
                    seq,
                    key,
                    kind,
                    vis,
                    body::text,
                    (EXTRACT(EPOCH FROM created_at) * 1000)::bigint,
                    (EXTRACT(EPOCH FROM updated_at) * 1000)::bigint
             FROM typemach_entries
             WHERE scope_key = $1
               AND session_id = $2
               AND ($3::text IS NULL OR thread_id = $3)
               AND ($4::text IS NULL OR kind = $4)
               AND ($5::text IS NULL OR vis = $5)
               AND seq > $6
             ORDER BY seq ASC
             LIMIT $7",
            &[
                &query.scope,
                &query.session_id.as_str(),
                &thread,
                &query.kind,
                &vis,
                &query.after_seq,
                &limit,
            ],
        )
        .await
        .map_err(store_db)?;
    rows.into_iter()
        .map(|row| row_entry(query.session_id, &row))
        .collect()
}

pub(in crate::pg) async fn latest_entry_tx<C>(
    tx: &C,
    scope_key: &str,
    session_id: &SessionId,
    thread_id: Option<&ThreadId>,
    kind: &str,
    vis: Option<Vis>,
) -> Result<Option<Entry>, MachineError>
where
    C: GenericClient + Sync,
{
    let thread = thread_id.map(ThreadId::as_str);
    let vis = vis.map(Vis::as_str);
    let row = tx
        .query_opt(
            "SELECT run_id,
                    thread_id,
                    seq,
                    key,
                    kind,
                    vis,
                    body::text,
                    (EXTRACT(EPOCH FROM created_at) * 1000)::bigint,
                    (EXTRACT(EPOCH FROM updated_at) * 1000)::bigint
             FROM typemach_entries
             WHERE scope_key = $1
               AND session_id = $2
               AND ($3::text IS NULL OR thread_id = $3)
               AND kind = $4
               AND ($5::text IS NULL OR vis = $5)
             ORDER BY seq DESC
             LIMIT 1",
            &[&scope_key, &session_id.as_str(), &thread, &kind, &vis],
        )
        .await
        .map_err(store_db)?;
    row.map(|row| row_entry(session_id, &row)).transpose()
}

async fn load_entry_tx<C>(
    tx: &C,
    scope_key: &str,
    session_id: &SessionId,
    key: &str,
) -> Result<Option<Entry>, MachineError>
where
    C: GenericClient + Sync,
{
    let row = tx
        .query_opt(
            "SELECT run_id,
                    thread_id,
                    seq,
                    key,
                    kind,
                    vis,
                    body::text,
                    (EXTRACT(EPOCH FROM created_at) * 1000)::bigint,
                    (EXTRACT(EPOCH FROM updated_at) * 1000)::bigint
             FROM typemach_entries
             WHERE scope_key = $1 AND session_id = $2 AND key = $3
             FOR UPDATE",
            &[&scope_key, &session_id.as_str(), &key],
        )
        .await
        .map_err(store_db)?;
    row.map(|row| row_entry(session_id, &row)).transpose()
}

fn row_entry(session_id: &SessionId, row: &Row) -> Result<Entry, MachineError> {
    let vis: String = row.get(5);
    Ok(Entry {
        run_id: RunId::from(row.get::<_, String>(0)),
        session_id: session_id.clone(),
        thread_id: ThreadId::from(row.get::<_, String>(1)),
        seq: row.get(2),
        key: row.get(3),
        kind: row.get(4),
        vis: Vis::parse(&vis).ok_or_else(|| MachineError::InvalidRunEvent {
            reason: format!("invalid entry visibility: {vis}"),
        })?,
        body: serde_json::from_str(&row.get::<_, String>(6))
            .map_err(MachineError::Deserialization)?,
        created_at: row.get(7),
        updated_at: row.get(8),
    })
}
