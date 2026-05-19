use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio_rusqlite::rusqlite::{OptionalExtension, Row, Transaction, params};

use super::*;
use crate::run::{LeaseId, ThreadId};
use crate::store::CheckpointWrite;

#[derive(Debug)]
pub(super) struct LookupRow {
    pub(super) run_id: String,
    pub(super) session_id: String,
    pub(super) thread_id: String,
    pub(super) status: String,
    pub(super) finish_reason: Option<String>,
    pub(super) cancel_requested: bool,
    pub(super) owner: Option<String>,
}

impl LookupRow {
    pub(super) fn into_lookup(self) -> Result<RunLookup, MachineError> {
        Ok(RunLookup {
            run_id: RunId::from(self.run_id),
            session_id: SessionId::from(self.session_id),
            thread_id: ThreadId::from(self.thread_id),
            status: RunStatus::parse(&self.status).ok_or_else(|| {
                MachineError::InvalidRunEvent {
                    reason: format!("invalid stored run status: {}", self.status),
                }
            })?,
            finish_reason: self.finish_reason,
            cancel_requested: self.cancel_requested,
            owner: self.owner.map(WorkerId::from),
        })
    }
}

#[derive(Debug)]
pub(super) struct CommitRow {
    pub(super) session_id: SessionId,
    pub(super) thread_id: ThreadId,
    pub(super) status: RunStatus,
    pub(super) lease_id: Option<LeaseId>,
    pub(super) lease_expired: bool,
}

#[derive(Debug)]
pub(super) struct CheckpointRow {
    pub(super) version: i64,
    pub(super) state: String,
    pub(super) next_step: Option<String>,
    pub(super) interrupted_step: Option<String>,
    pub(super) interrupt: Option<String>,
    pub(super) run_id: Option<String>,
}

pub(super) fn insert_session(
    conn: &tokio_rusqlite::rusqlite::Connection,
    session_id: &SessionId,
    scope_key: &str,
    scope_json: &str,
) -> Result<(), MachineError> {
    conn.execute(
        "INSERT INTO typemach_sessions (session_id, scope_key, scope)
         VALUES (?1, ?2, ?3)
         ON CONFLICT(scope_key, session_id) DO UPDATE SET scope = excluded.scope",
        params![session_id.as_str(), scope_key, scope_json],
    )
    .map_err(store_db)?;
    Ok(())
}

pub(super) fn insert_session_tx(
    tx: &Transaction<'_>,
    session_id: &SessionId,
    scope_key: &str,
    scope_json: &str,
    _now: i64,
) -> Result<(), MachineError> {
    tx.execute(
        "INSERT INTO typemach_sessions (session_id, scope_key, scope)
         VALUES (?1, ?2, ?3)
         ON CONFLICT(scope_key, session_id) DO UPDATE SET scope = excluded.scope",
        params![session_id.as_str(), scope_key, scope_json],
    )
    .map_err(store_db)?;
    Ok(())
}

pub(super) fn lookup_run_conn(
    conn: &tokio_rusqlite::rusqlite::Connection,
    run_id: &RunId,
    scope_key: &str,
) -> Result<Option<RunLookup>, MachineError> {
    let row = conn
        .query_row(
            "SELECT run_id, session_id, thread_id, status, finish_reason, cancel_requested, owner_id
             FROM typemach_runs
             WHERE run_id = ?1 AND scope_key = ?2",
            params![run_id.as_str(), scope_key],
            lookup_row,
        )
        .optional()
        .map_err(store_db)?;
    row.map(LookupRow::into_lookup).transpose()
}

pub(super) fn lookup_run_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    scope_key: &str,
) -> Result<Option<RunLookup>, MachineError> {
    let row = tx
        .query_row(
            "SELECT run_id, session_id, thread_id, status, finish_reason, cancel_requested, owner_id
             FROM typemach_runs
             WHERE run_id = ?1 AND scope_key = ?2",
            params![run_id.as_str(), scope_key],
            lookup_row,
        )
        .optional()
        .map_err(store_db)?;
    row.map(LookupRow::into_lookup).transpose()
}

pub(super) fn find_idempotent_tx(
    tx: &Transaction<'_>,
    scope_key: &str,
    session_id: &SessionId,
    key: &str,
) -> Result<Option<RunLookup>, MachineError> {
    let row = tx
        .query_row(
            "SELECT run_id, session_id, thread_id, status, finish_reason, cancel_requested, owner_id
             FROM typemach_runs
             WHERE scope_key = ?1 AND session_id = ?2 AND client_run_key = ?3
             ORDER BY started_at DESC
             LIMIT 1",
            params![scope_key, session_id.as_str(), key],
            lookup_row,
        )
        .optional()
        .map_err(store_db)?;
    row.map(LookupRow::into_lookup).transpose()
}

pub(super) fn ensure_run_input_column(
    conn: &tokio_rusqlite::rusqlite::Connection,
) -> Result<(), MachineError> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(typemach_runs)")
        .map_err(store_db)?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(store_db)?;
    for row in rows {
        if row.map_err(store_db)? == "input" {
            return Ok(());
        }
    }
    conn.execute("ALTER TABLE typemach_runs ADD COLUMN input TEXT NULL", [])
        .map_err(store_db)?;
    Ok(())
}

pub(super) fn ensure_run_start_sig_column(
    conn: &tokio_rusqlite::rusqlite::Connection,
) -> Result<(), MachineError> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(typemach_runs)")
        .map_err(store_db)?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(store_db)?;
    for row in rows {
        if row.map_err(store_db)? == "start_sig" {
            return Ok(());
        }
    }
    conn.execute(
        "ALTER TABLE typemach_runs ADD COLUMN start_sig TEXT NOT NULL DEFAULT ''",
        [],
    )
    .map_err(store_db)?;
    Ok(())
}

pub(super) fn run_start_sig_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
) -> Result<String, MachineError> {
    tx.query_row(
        "SELECT start_sig FROM typemach_runs WHERE run_id = ?1",
        params![run_id.as_str()],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(store_db)?
    .ok_or(MachineError::RunNotFound)
}

pub(super) fn lookup_row(row: &Row<'_>) -> tokio_rusqlite::rusqlite::Result<LookupRow> {
    Ok(LookupRow {
        run_id: row.get(0)?,
        session_id: row.get(1)?,
        thread_id: row.get(2)?,
        status: row.get(3)?,
        finish_reason: row.get(4)?,
        cancel_requested: row.get::<_, i64>(5)? != 0,
        owner: row.get(6)?,
    })
}

pub(super) fn commit_row_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
    scope_key: &str,
) -> Result<Option<CommitRow>, MachineError> {
    tx.query_row(
        "SELECT session_id, status, lease_id, lease_expires_at, thread_id
         FROM typemach_runs
         WHERE run_id = ?1 AND scope_key = ?2",
        params![run_id.as_str(), scope_key],
        |row| {
            let session_id = SessionId::from(row.get::<_, String>(0)?);
            let status_raw: String = row.get(1)?;
            let lease_id = row.get::<_, Option<String>>(2)?.map(LeaseId::from);
            let expires = row.get::<_, Option<i64>>(3)?;
            let thread_id = ThreadId::from(row.get::<_, String>(4)?);
            Ok((session_id, status_raw, lease_id, expires, thread_id))
        },
    )
    .optional()
    .map_err(store_db)?
    .map(|(session_id, status_raw, lease_id, expires, thread_id)| {
        Ok(CommitRow {
            session_id,
            thread_id,
            status: RunStatus::parse(&status_raw).ok_or_else(|| MachineError::InvalidRunEvent {
                reason: format!("invalid stored run status: {status_raw}"),
            })?,
            lease_id,
            lease_expired: expires.is_some_and(|expires| expires <= now_ms()),
        })
    })
    .transpose()
}

pub(super) fn claim_thread_tx<Scope>(
    tx: &Transaction<'_>,
    run: &RunStart<Scope>,
    claim: &crate::store::LeaseClaim,
    now: i64,
) -> Result<(), MachineError> {
    let claimed = tx
        .execute(
            "INSERT INTO typemach_thread_leases (
            thread_id, run_id, owner_id, lease_id, lease_expires_at, updated_at
         )
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(thread_id) DO UPDATE SET
            run_id = excluded.run_id,
            owner_id = excluded.owner_id,
            lease_id = excluded.lease_id,
            lease_expires_at = excluded.lease_expires_at,
            updated_at = excluded.updated_at
         WHERE typemach_thread_leases.lease_expires_at <= ?6",
            params![
                run.thread_id.as_str(),
                run.run_id.as_str(),
                claim.owner.as_str(),
                claim.id.as_str(),
                now + duration_ms(claim.ttl),
                now,
            ],
        )
        .map_err(store_db)?;
    if claimed == 1 {
        return Ok(());
    }

    let busy = tx
        .query_row(
            "SELECT owner_id, run_id
             FROM typemach_thread_leases
             WHERE thread_id = ?1 AND lease_expires_at > ?2",
            params![run.thread_id.as_str(), now],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(store_db)?;
    Err(MachineError::ThreadBusy {
        owner: busy
            .as_ref()
            .map(|(owner, _)| WorkerId::from(owner.clone())),
        run: busy.as_ref().map(|(_, run)| RunId::from(run.clone())),
    })
}

pub(super) fn running_thread_tx(
    tx: &Transaction<'_>,
    thread_id: &ThreadId,
    except: Option<&RunId>,
) -> Result<Option<BusyThread>, MachineError> {
    let except = except.map(RunId::as_str);
    let row = tx
        .query_row(
            "SELECT owner_id, run_id
             FROM typemach_runs
             WHERE thread_id = ?1
               AND status = 'running'
               AND (?2 IS NULL OR run_id <> ?2)
             ORDER BY started_at ASC
             LIMIT 1",
            params![thread_id.as_str(), except],
            |row| Ok((row.get::<_, Option<String>>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(store_db)?;
    Ok(row.map(|(owner, run)| (owner.map(WorkerId::from), Some(RunId::from(run)))))
}

pub(super) fn check_sqlite_lease(
    row: &CommitRow,
    lease: Option<&LeaseId>,
) -> Result<(), MachineError> {
    let Some(stored) = &row.lease_id else {
        return Ok(());
    };
    if lease != Some(stored) || row.lease_expired {
        return Err(MachineError::LeaseLost);
    }
    Ok(())
}

pub(super) fn check_thread_tx<E, Data, Scope>(
    tx: &Transaction<'_>,
    row: &CommitRow,
    commit: &RunCommit<E, Data, Scope>,
    checkpoint: &CheckpointWrite,
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    if checkpoint.thread_id != row.thread_id {
        return Err(MachineError::InvalidRunEvent {
            reason: "checkpoint thread_id does not match target run".to_string(),
        });
    }
    if let Some((owner, run_id)) =
        running_thread_tx(tx, &checkpoint.thread_id, Some(&commit.run_id))?
    {
        return Err(MachineError::ThreadBusy { owner, run: run_id });
    }
    let Some(stored) = &row.lease_id else {
        let active = tx
            .query_row(
                "SELECT owner_id, run_id
                 FROM typemach_thread_leases
                 WHERE thread_id = ?1 AND lease_expires_at > ?2",
                params![checkpoint.thread_id.as_str(), now_ms()],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()
            .map_err(store_db)?;
        if let Some((owner, run)) = active {
            return Err(MachineError::ThreadBusy {
                owner: Some(WorkerId::from(owner)),
                run: Some(RunId::from(run)),
            });
        }
        return Ok(());
    };
    let Some(lease) = commit.lease.as_ref() else {
        return Err(MachineError::LeaseLost);
    };
    if lease != stored {
        return Err(MachineError::LeaseLost);
    }
    let active = tx
        .query_row(
            "SELECT 1
             FROM typemach_thread_leases
             WHERE thread_id = ?1
               AND run_id = ?2
               AND lease_id = ?3
               AND lease_expires_at > ?4",
            params![
                checkpoint.thread_id.as_str(),
                commit.run_id.as_str(),
                lease.as_str(),
                now_ms(),
            ],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(store_db)?;
    if active.is_none() {
        return Err(MachineError::LeaseLost);
    }
    Ok(())
}

pub(super) fn save_checkpoint_tx(
    tx: &Transaction<'_>,
    thread_id: &str,
    checkpoint: &CheckpointRecord,
) -> Result<(), MachineError> {
    let state = json_text(&checkpoint.state)?;
    let next_step = checkpoint.next_step.as_ref().map(json_text).transpose()?;
    let interrupted_step = checkpoint
        .interrupted_step
        .as_ref()
        .map(json_text)
        .transpose()?;
    let interrupt = checkpoint.interrupt.as_ref().map(json_text).transpose()?;
    let now = now_ms();
    tx.execute(
        "INSERT INTO typemach_checkpoints (
            thread_id, version, state, next_step, interrupted_step, interrupt, run_id, updated_at, created_at
         )
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8)
         ON CONFLICT(thread_id) DO UPDATE SET
            version = excluded.version,
            state = excluded.state,
            next_step = excluded.next_step,
            interrupted_step = excluded.interrupted_step,
            interrupt = excluded.interrupt,
            run_id = excluded.run_id,
            updated_at = excluded.updated_at",
        params![
            thread_id,
            checkpoint.version as i64,
            state,
            next_step,
            interrupted_step,
            interrupt,
            checkpoint.run_id,
            now,
        ],
    )
    .map_err(store_db)?;
    Ok(())
}

pub(super) fn checkpoint_row(row: &Row<'_>) -> tokio_rusqlite::rusqlite::Result<CheckpointRow> {
    Ok(CheckpointRow {
        version: row.get(0)?,
        state: row.get(1)?,
        next_step: row.get(2)?,
        interrupted_step: row.get(3)?,
        interrupt: row.get(4)?,
        run_id: row.get(5)?,
    })
}

pub(super) fn decode_checkpoint(row: CheckpointRow) -> Result<CheckpointRecord, MachineError> {
    Ok(CheckpointRecord {
        version: row.version as u32,
        state: serde_json::from_str(&row.state).map_err(MachineError::Deserialization)?,
        next_step: row
            .next_step
            .map(|raw| serde_json::from_str(raw.as_str()))
            .transpose()
            .map_err(MachineError::Deserialization)?,
        interrupted_step: row
            .interrupted_step
            .map(|raw| serde_json::from_str(raw.as_str()))
            .transpose()
            .map_err(MachineError::Deserialization)?,
        interrupt: row
            .interrupt
            .map(|raw| serde_json::from_str(raw.as_str()))
            .transpose()
            .map_err(MachineError::Deserialization)?,
        run_id: row.run_id,
    })
}

pub(super) fn last_seq_tx(tx: &Transaction<'_>, run_id: &RunId) -> Result<i64, MachineError> {
    tx.query_row(
        "SELECT COALESCE(MAX(seq), 0) FROM typemach_run_events WHERE run_id = ?1",
        params![run_id.as_str()],
        |row| row.get(0),
    )
    .map_err(store_db)
}

pub(super) fn insert_event_tx<E>(tx: &Transaction<'_>, event: &E) -> Result<(), MachineError>
where
    E: RunEvent + Serialize,
{
    let event_json = json_text(event)?;
    tx.execute(
        "INSERT INTO typemach_run_events (run_id, session_id, seq, terminal, event)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            event.run_id().as_str(),
            event.session_id().as_str(),
            event.seq(),
            i64::from(event.is_terminal()),
            event_json,
        ],
    )
    .map_err(store_db)?;
    Ok(())
}

pub(super) fn terminal_event_tx<E>(
    tx: &Transaction<'_>,
    run_id: &RunId,
) -> Result<Option<E>, MachineError>
where
    E: DeserializeOwned,
{
    let raw = tx
        .query_row(
            "SELECT event FROM typemach_run_events
             WHERE run_id = ?1 AND terminal = 1
             ORDER BY seq DESC
             LIMIT 1",
            params![run_id.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(store_db)?;
    raw.map(|raw| decode_event(raw.as_str())).transpose()
}

pub(super) fn delete_thread_tx(tx: &Transaction<'_>, run_id: &RunId) -> Result<(), MachineError> {
    tx.execute(
        "DELETE FROM typemach_thread_leases WHERE run_id = ?1",
        params![run_id.as_str()],
    )
    .map_err(store_db)?;
    Ok(())
}

pub(super) fn validate_commit<E, Data, Scope>(
    commit: &RunCommit<E, Data, Scope>,
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    for event in &commit.events {
        if event.run_id() != &commit.run_id {
            return Err(MachineError::InvalidRunEvent {
                reason: "event run_id does not match target run".to_string(),
            });
        }
        if event.session_id() != &commit.session_id {
            return Err(MachineError::InvalidRunEvent {
                reason: "event session_id does not match target run".to_string(),
            });
        }
        if event.seq() <= 0 {
            return Err(MachineError::InvalidRunEvent {
                reason: "event seq must be positive".to_string(),
            });
        }
    }
    if let Some(finish) = &commit.finish {
        if finish.run_id != commit.run_id || finish.session_id != commit.session_id {
            return Err(MachineError::InvalidRunEvent {
                reason: "finish target does not match committed run".to_string(),
            });
        }
        if !finish.status.is_terminal() {
            return Err(MachineError::InvalidRunEvent {
                reason: "finish_run requires a terminal status".to_string(),
            });
        }
        let Some(last) = commit.events.last() else {
            return Err(MachineError::InvalidRunEvent {
                reason: "finish commit requires a terminal event".to_string(),
            });
        };
        if !last.is_terminal() {
            return Err(MachineError::InvalidRunEvent {
                reason: "finish commit requires a terminal event".to_string(),
            });
        }
        if commit.events[..commit.events.len() - 1]
            .iter()
            .any(RunEvent::is_terminal)
        {
            return Err(MachineError::InvalidRunEvent {
                reason: "only the last commit event may be terminal".to_string(),
            });
        }
    } else if commit.events.iter().any(RunEvent::is_terminal) {
        return Err(MachineError::InvalidRunEvent {
            reason: "record_event does not accept terminal events".to_string(),
        });
    }
    Ok(())
}

pub(super) fn decode_event<E>(raw: &str) -> Result<E, MachineError>
where
    E: DeserializeOwned,
{
    serde_json::from_str(raw).map_err(MachineError::Deserialization)
}

pub(super) fn json_text<T>(value: &T) -> Result<String, MachineError>
where
    T: Serialize,
{
    serde_json::to_string(value).map_err(MachineError::Serialization)
}

pub(super) fn scope_key<Scope>(scope: &Scope) -> Result<String, MachineError>
where
    Scope: Serialize,
{
    let value = serde_json::to_value(scope).map_err(MachineError::Serialization)?;
    serde_json::to_string(&value).map_err(MachineError::Serialization)
}

pub(super) fn new_session_id() -> SessionId {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    SessionId::from(format!("session-{nanos}"))
}

pub(super) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or_default()
}

pub(super) fn duration_ms(duration: Duration) -> i64 {
    duration.as_millis().min(i64::MAX as u128) as i64
}

pub(super) fn call_db(err: tokio_rusqlite::Error<MachineError>) -> MachineError {
    match err {
        tokio_rusqlite::Error::Error(err) => err,
        err => store_msg(format!("sqlite call failed: {err}")),
    }
}

pub(super) fn checkpoint_db(err: tokio_rusqlite::rusqlite::Error) -> MachineError {
    MachineError::CheckpointDb(Box::new(err))
}

pub(super) fn store_db(err: tokio_rusqlite::rusqlite::Error) -> MachineError {
    MachineError::StoreDb(Box::new(err))
}

pub(super) fn store_msg(message: String) -> MachineError {
    MachineError::StoreDb(Box::new(std::io::Error::other(message)))
}
