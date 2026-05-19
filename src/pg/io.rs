use std::time::{SystemTime, UNIX_EPOCH};

use deadpool_postgres::tokio_postgres::Row;
use deadpool_postgres::{GenericClient, Transaction};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::*;
use crate::run::LeaseId;

pub(super) async fn claim_thread_tx<Scope>(
    tx: &Transaction<'_>,
    run: &RunStart<Scope>,
    claim: &crate::store::LeaseClaim,
) -> Result<(), MachineError> {
    let ttl = claim.ttl.as_secs_f64();
    let claimed = tx
        .query_opt(
            "INSERT INTO typemach_thread_leases (
                thread_id, run_id, owner_id, lease_id, lease_expires_at, updated_at
             )
             VALUES ($1, $2, $3, $4, now() + ($5::double precision * interval '1 second'), now())
             ON CONFLICT (thread_id) DO UPDATE SET
                run_id = EXCLUDED.run_id,
                owner_id = EXCLUDED.owner_id,
                lease_id = EXCLUDED.lease_id,
                lease_expires_at = EXCLUDED.lease_expires_at,
                updated_at = now()
             WHERE typemach_thread_leases.lease_expires_at <= now()
             RETURNING thread_id",
            &[
                &run.thread_id.as_str(),
                &run.run_id.as_str(),
                &claim.owner.as_str(),
                &claim.id.as_str(),
                &ttl,
            ],
        )
        .await
        .map_err(store_db)?;
    if claimed.is_some() {
        return Ok(());
    }

    let busy = tx
        .query_opt(
            "SELECT owner_id, run_id
             FROM typemach_thread_leases
             WHERE thread_id = $1 AND lease_expires_at > now()",
            &[&run.thread_id.as_str()],
        )
        .await
        .map_err(store_db)?;
    Err(MachineError::ThreadBusy {
        owner: busy
            .as_ref()
            .map(|row| WorkerId::from(row.get::<_, String>(0))),
        run: busy
            .as_ref()
            .map(|row| RunId::from(row.get::<_, String>(1))),
    })
}

pub(super) async fn running_thread_tx<C>(
    client: &C,
    thread_id: &ThreadId,
    except: Option<&RunId>,
) -> Result<Option<BusyThread>, MachineError>
where
    C: GenericClient + Sync,
{
    let except = except.map(RunId::as_str);
    let row = client
        .query_opt(
            "SELECT owner_id, run_id
             FROM typemach_runs
             WHERE thread_id = $1
               AND status = 'running'
               AND ($2::text IS NULL OR run_id <> $2)
             ORDER BY started_at ASC
             LIMIT 1",
            &[&thread_id.as_str(), &except],
        )
        .await
        .map_err(store_db)?;
    Ok(row.map(|row| {
        (
            row.get::<_, Option<String>>(0).map(WorkerId::from),
            Some(RunId::from(row.get::<_, String>(1))),
        )
    }))
}

pub(super) async fn check_thread_tx<E, Data, Scope>(
    tx: &Transaction<'_>,
    row: &Row,
    commit: &RunCommit<E, Data, Scope>,
    checkpoint: &crate::store::CheckpointWrite,
) -> Result<(), MachineError>
where
    E: RunEvent,
{
    let stored_thread: String = row.get(4);
    if checkpoint.thread_id.as_str() != stored_thread {
        return Err(MachineError::InvalidRunEvent {
            reason: "checkpoint thread_id does not match target run".to_string(),
        });
    }
    if let Some((owner, run_id)) =
        running_thread_tx(tx, &checkpoint.thread_id, Some(&commit.run_id)).await?
    {
        return Err(MachineError::ThreadBusy { owner, run: run_id });
    }
    let stored_lease: Option<String> = row.get(2);
    let Some(stored_lease) = stored_lease else {
        let active = tx
            .query_opt(
                "SELECT owner_id, run_id
                 FROM typemach_thread_leases
                 WHERE thread_id = $1 AND lease_expires_at > now()",
                &[&checkpoint.thread_id.as_str()],
            )
            .await
            .map_err(store_db)?;
        if let Some(active) = active {
            return Err(MachineError::ThreadBusy {
                owner: Some(WorkerId::from(active.get::<_, String>(0))),
                run: Some(RunId::from(active.get::<_, String>(1))),
            });
        }
        return Ok(());
    };
    let Some(lease) = commit.lease.as_ref() else {
        return Err(MachineError::LeaseLost);
    };
    if lease.as_str() != stored_lease {
        return Err(MachineError::LeaseLost);
    }
    let active = tx
        .query_opt(
            "SELECT 1
             FROM typemach_thread_leases
             WHERE thread_id = $1
               AND run_id = $2
               AND lease_id = $3
               AND lease_expires_at > now()",
            &[
                &checkpoint.thread_id.as_str(),
                &commit.run_id.as_str(),
                &lease.as_str(),
            ],
        )
        .await
        .map_err(store_db)?;
    if active.is_none() {
        return Err(MachineError::LeaseLost);
    }
    Ok(())
}

pub(super) async fn delete_thread_tx(
    tx: &Transaction<'_>,
    run_id: &RunId,
) -> Result<(), MachineError> {
    tx.execute(
        "DELETE FROM typemach_thread_leases WHERE run_id = $1",
        &[&run_id.as_str()],
    )
    .await
    .map_err(store_db)?;
    Ok(())
}

pub(super) async fn save_checkpoint<C>(
    client: &C,
    thread_id: &str,
    checkpoint: &CheckpointRecord,
    db_error: fn(deadpool_postgres::tokio_postgres::Error) -> MachineError,
) -> Result<(), MachineError>
where
    C: GenericClient + Sync,
{
    let raw = serde_json::to_string(&checkpoint.state).map_err(MachineError::Serialization)?;
    let next_step = checkpoint
        .next_step
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(MachineError::Serialization)?;
    let interrupted_step = checkpoint
        .interrupted_step
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(MachineError::Serialization)?;
    let interrupt = checkpoint
        .interrupt
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(MachineError::Serialization)?;
    client
        .execute(
            "INSERT INTO typemach_checkpoints (
                thread_id, version, state, next_step, interrupted_step, interrupt, run_id, updated_at
             )
             VALUES ($1, $2, $3::text::jsonb, $4::text::jsonb, $5::text::jsonb, $6::text::jsonb, $7, now())
             ON CONFLICT (thread_id) DO UPDATE SET
                version = EXCLUDED.version,
                state = EXCLUDED.state,
                next_step = EXCLUDED.next_step,
                interrupted_step = EXCLUDED.interrupted_step,
                interrupt = EXCLUDED.interrupt,
                run_id = EXCLUDED.run_id,
                updated_at = now()",
            &[
                &thread_id,
                &(checkpoint.version as i32),
                &raw,
                &next_step,
                &interrupted_step,
                &interrupt,
                &checkpoint.run_id,
            ],
        )
        .await
        .map_err(db_error)?;
    Ok(())
}

pub(super) async fn load_checkpoint<C>(
    client: &C,
    thread_id: &str,
) -> Result<Option<Row>, deadpool_postgres::tokio_postgres::Error>
where
    C: GenericClient + Sync,
{
    client
        .query_opt(
            "SELECT version,
                    state::text,
                    next_step::text,
                    interrupted_step::text,
                    interrupt::text,
                    run_id
             FROM typemach_checkpoints
             WHERE thread_id = $1",
            &[&thread_id],
        )
        .await
}

pub(super) fn decode_checkpoint(row: Row) -> Result<CheckpointRecord, MachineError> {
    let version: i32 = row.get(0);
    let raw: String = row.get(1);
    let raw_next_step: Option<String> = row.get(2);
    let raw_interrupted_step: Option<String> = row.get(3);
    let raw_interrupt: Option<String> = row.get(4);
    let run_id: Option<String> = row.get(5);
    Ok(CheckpointRecord {
        version: version as u32,
        state: serde_json::from_str(&raw).map_err(MachineError::Deserialization)?,
        next_step: raw_next_step
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(MachineError::Deserialization)?,
        interrupted_step: raw_interrupted_step
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(MachineError::Deserialization)?,
        interrupt: raw_interrupt
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(MachineError::Deserialization)?,
        run_id,
    })
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
    match &commit.finish {
        Some(finish) => {
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
                    reason: "finish_run requires a terminal event".to_string(),
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
        }
        None => {
            if commit.events.iter().any(RunEvent::is_terminal) {
                return Err(MachineError::InvalidRunEvent {
                    reason: "record_event does not accept terminal events".to_string(),
                });
            }
        }
    }
    Ok(())
}

pub(super) async fn terminal_event_tx<E>(
    tx: &Transaction<'_>,
    run_id: &RunId,
) -> Result<Option<E>, MachineError>
where
    E: RunEvent + DeserializeOwned,
{
    let row = tx
        .query_opt(
            "SELECT event::text
             FROM typemach_run_events
             WHERE run_id = $1 AND terminal
             ORDER BY seq DESC
             LIMIT 1",
            &[&run_id.as_str()],
        )
        .await
        .map_err(store_db)?;
    row.map(|row| decode_event(row.get::<_, String>(0).as_str()))
        .transpose()
}

pub(super) async fn run_input_tx<C>(
    client: &C,
    run_id: &RunId,
) -> Result<Option<Value>, MachineError>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_opt(
            "SELECT input::text FROM typemach_runs WHERE run_id = $1",
            &[&run_id.as_str()],
        )
        .await
        .map_err(store_db)?;
    row.and_then(|row| row.get::<_, Option<String>>(0))
        .map(|raw| serde_json::from_str(raw.as_str()).map_err(MachineError::Deserialization))
        .transpose()
}

pub(super) async fn last_seq_tx(tx: &Transaction<'_>, run_id: &RunId) -> Result<i64, MachineError> {
    let row = tx
        .query_one(
            "SELECT COALESCE(MAX(seq), 0)
             FROM typemach_run_events
             WHERE run_id = $1",
            &[&run_id.as_str()],
        )
        .await
        .map_err(store_db)?;
    Ok(row.get(0))
}

pub(super) async fn insert_event_tx<E>(tx: &Transaction<'_>, event: &E) -> Result<(), MachineError>
where
    E: RunEvent + Serialize,
{
    let event_json = json_text(event)?;
    tx.execute(
        "INSERT INTO typemach_run_events (run_id, session_id, seq, terminal, event)
         VALUES ($1, $2, $3, $4, $5::text::jsonb)",
        &[
            &event.run_id().as_str(),
            &event.session_id().as_str(),
            &event.seq(),
            &event.is_terminal(),
            &event_json,
        ],
    )
    .await
    .map_err(store_db)?;
    Ok(())
}

pub(super) fn row_lookup(row: &Row) -> Result<RunLookup, MachineError> {
    Ok(RunLookup {
        run_id: RunId::from(row.get::<_, String>(0)),
        session_id: SessionId::from(row.get::<_, String>(1)),
        thread_id: ThreadId::from(row.get::<_, String>(2)),
        status: row_status(row, 3)?,
        finish_reason: row.get(4),
        cancel_requested: row.get(5),
        owner: row.get::<_, Option<String>>(6).map(WorkerId::from),
    })
}

pub(super) fn row_status(row: &Row, index: usize) -> Result<RunStatus, MachineError> {
    let status: String = row.get(index);
    RunStatus::parse(&status).ok_or_else(|| MachineError::InvalidRunEvent {
        reason: format!("invalid stored run status: {status}"),
    })
}

pub(super) fn check_pg_lease(row: &Row, lease: Option<&LeaseId>) -> Result<(), MachineError> {
    let stored: Option<String> = row.get(2);
    let expired: bool = row.get(3);
    let Some(stored) = stored else {
        return Ok(());
    };
    if lease.map(LeaseId::as_str) != Some(stored.as_str()) || expired {
        return Err(MachineError::LeaseLost);
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
    SessionId::from(format!("session-{}-{nanos}", std::process::id()))
}

pub(super) fn is_unique_violation(err: &deadpool_postgres::tokio_postgres::Error) -> bool {
    err.code().is_some_and(|code| {
        *code == deadpool_postgres::tokio_postgres::error::SqlState::UNIQUE_VIOLATION
    })
}

pub(super) fn checkpoint_db(err: deadpool_postgres::tokio_postgres::Error) -> MachineError {
    MachineError::CheckpointDb(Box::new(err))
}

pub(super) fn store_db(err: deadpool_postgres::tokio_postgres::Error) -> MachineError {
    MachineError::StoreDb(Box::new(err))
}

pub(super) fn store_msg(message: String) -> MachineError {
    MachineError::StoreDb(Box::new(std::io::Error::other(message)))
}
