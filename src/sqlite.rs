use std::marker::PhantomData;
use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio_rusqlite::Connection;
use tokio_rusqlite::rusqlite::{OptionalExtension, TransactionBehavior, params};

use crate::checkpoint::{CheckpointRecord, CheckpointStore};
use crate::error::MachineError;
use crate::op::{EntryWrite, Page};
use crate::run::{RunId, SessionId, WorkerId};
use crate::runtime::Event;
use crate::store::{
    FinishRunResult, RunCommit, RunCommitResult, RunEvent, RunFinish, RunFinishRecord, RunLookup,
    RunStart, RunStatus, RunStore, RunTx, StoreStartResult, start_sig,
};

type SqliteTypes<E, Scope, Data> = fn() -> (E, Scope, Data);
type BusyThread = (Option<WorkerId>, Option<RunId>);

pub struct SqliteStore<E = Event, Scope = Value, Data = ()> {
    conn: Connection,
    _types: PhantomData<SqliteTypes<E, Scope, Data>>,
}

impl<E, Scope, Data> Clone for SqliteStore<E, Scope, Data> {
    fn clone(&self) -> Self {
        Self {
            conn: self.conn.clone(),
            _types: PhantomData,
        }
    }
}

impl<E, Scope, Data> SqliteStore<E, Scope, Data> {
    pub fn new(conn: Connection) -> Self {
        Self {
            conn,
            _types: PhantomData,
        }
    }

    pub async fn open(path: impl AsRef<Path>) -> Result<Self, MachineError> {
        let conn = Connection::open(path).await.map_err(store_db)?;
        let store = Self::new(conn);
        store.configure().await?;
        Ok(store)
    }

    pub async fn memory() -> Result<Self, MachineError> {
        let conn = Connection::open_in_memory().await.map_err(store_db)?;
        let store = Self::new(conn);
        store.configure().await?;
        Ok(store)
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    pub async fn ensure_schema(&self) -> Result<(), MachineError> {
        self.call(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS typemach_checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL DEFAULT 1,
                    state TEXT NOT NULL,
                    next_step TEXT NULL,
                    interrupted_step TEXT NULL,
                    interrupt TEXT NULL,
                    run_id TEXT NULL,
                    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
                );
                CREATE TABLE IF NOT EXISTS typemach_sessions (
                    scope_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    PRIMARY KEY (scope_key, session_id)
                );
                CREATE TABLE IF NOT EXISTS typemach_runs (
                    run_id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    agent_kind TEXT NOT NULL,
                    model TEXT NULL,
                    client_run_key TEXT NULL,
                    parent_run_id TEXT NULL,
                    retry_of_run_id TEXT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    input TEXT NULL,
                    start_sig TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    started_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    finished_at INTEGER NULL,
                    finish_reason TEXT NULL,
                    error_code TEXT NULL,
                    finish_data TEXT NULL,
                    owner_id TEXT NULL,
                    lease_id TEXT NULL,
                    lease_expires_at INTEGER NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    FOREIGN KEY (scope_key, session_id)
                        REFERENCES typemach_sessions(scope_key, session_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS typemach_runs_session_idx
                    ON typemach_runs (scope_key, session_id, started_at DESC);
                CREATE INDEX IF NOT EXISTS typemach_runs_scope_status_idx
                    ON typemach_runs (scope_key, status, started_at DESC);
                CREATE INDEX IF NOT EXISTS typemach_runs_lease_idx
                    ON typemach_runs (status, lease_expires_at)
                    WHERE status = 'running' AND lease_expires_at IS NOT NULL;
                CREATE INDEX IF NOT EXISTS typemach_runs_thread_idx
                    ON typemach_runs (thread_id, status);
                CREATE UNIQUE INDEX IF NOT EXISTS typemach_runs_running_thread_idx
                    ON typemach_runs (thread_id)
                    WHERE status = 'running';
                CREATE UNIQUE INDEX IF NOT EXISTS typemach_runs_idempotency_idx
                    ON typemach_runs (scope_key, session_id, client_run_key)
                    WHERE client_run_key IS NOT NULL;
                CREATE TABLE IF NOT EXISTS typemach_thread_leases (
                    thread_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES typemach_runs(run_id) ON DELETE CASCADE,
                    owner_id TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    lease_expires_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
                );
                CREATE TABLE IF NOT EXISTS typemach_run_events (
                    run_id TEXT NOT NULL REFERENCES typemach_runs(run_id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    terminal INTEGER NOT NULL DEFAULT 0,
                    event TEXT NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    PRIMARY KEY (run_id, seq)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS typemach_run_events_terminal_idx
                    ON typemach_run_events (run_id)
                    WHERE terminal = 1;
                CREATE TABLE IF NOT EXISTS typemach_effects (
                    run_id TEXT NOT NULL REFERENCES typemach_runs(run_id) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request TEXT NOT NULL,
                    result TEXT NULL,
                    error_code TEXT NULL,
                    error_message TEXT NULL,
                    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    PRIMARY KEY (run_id, key)
                );
                CREATE INDEX IF NOT EXISTS typemach_effects_status_idx
                    ON typemach_effects (run_id, status);
                CREATE TABLE IF NOT EXISTS typemach_items (
                    run_id TEXT NOT NULL REFERENCES typemach_runs(run_id) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    PRIMARY KEY (run_id, key)
                );
                CREATE TABLE IF NOT EXISTS typemach_entries (
                    scope_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    vis TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                    PRIMARY KEY (scope_key, session_id, key),
                    UNIQUE (scope_key, session_id, seq),
                    FOREIGN KEY (scope_key, session_id)
                        REFERENCES typemach_sessions(scope_key, session_id)
                        ON DELETE CASCADE
                );",
            )
            .map_err(store_db)?;
            ensure_run_input_column(conn)?;
            ensure_run_start_sig_column(conn)?;
            Ok(())
        })
        .await
    }

    async fn configure(&self) -> Result<(), MachineError> {
        self.call(|conn| {
            conn.busy_timeout(Duration::from_secs(5))
                .map_err(store_db)?;
            conn.pragma_update(None, "foreign_keys", "ON")
                .map_err(store_db)?;
            conn.pragma_update(None, "journal_mode", "WAL")
                .map_err(store_db)?;
            Ok(())
        })
        .await
    }

    async fn call<T, F>(&self, f: F) -> Result<T, MachineError>
    where
        T: Send + 'static,
        F: FnOnce(&mut tokio_rusqlite::rusqlite::Connection) -> Result<T, MachineError>
            + Send
            + 'static,
    {
        self.conn.call(f).await.map_err(call_db)
    }
}

#[async_trait]
impl<E, Scope, Data> CheckpointStore for SqliteStore<E, Scope, Data>
where
    E: Send + Sync,
    Scope: Send + Sync,
    Data: Send + Sync,
{
    async fn load_checkpoint(
        &self,
        thread_id: &str,
    ) -> Result<Option<CheckpointRecord>, MachineError> {
        let thread_id = thread_id.to_string();
        self.call(move |conn| {
            let row = conn
                .query_row(
                    "SELECT version, state, next_step, interrupted_step, interrupt, run_id
                     FROM typemach_checkpoints
                     WHERE thread_id = ?1",
                    params![thread_id],
                    checkpoint_row,
                )
                .optional()
                .map_err(checkpoint_db)?;
            row.map(decode_checkpoint).transpose()
        })
        .await
    }
}

#[async_trait]
impl<E, Scope, Data> RunStore<E> for SqliteStore<E, Scope, Data>
where
    E: RunEvent + Serialize + DeserializeOwned,
    Scope: Clone + Serialize + Send + Sync + 'static,
    Data: Clone + Serialize + Send + Sync + 'static,
{
    type Scope = Scope;
    type FinishData = Data;

    async fn ensure_session(
        &self,
        session_id: Option<SessionId>,
        scope: &Scope,
    ) -> Result<SessionId, MachineError> {
        let scope_key = scope_key(scope)?;
        let scope_json = json_text(scope)?;
        self.call(move |conn| {
            if let Some(session_id) = session_id {
                insert_session(conn, &session_id, &scope_key, &scope_json)?;
                return Ok(session_id);
            }
            loop {
                let session_id = new_session_id();
                let inserted = conn
                    .execute(
                        "INSERT OR IGNORE INTO typemach_sessions (session_id, scope_key, scope)
                         VALUES (?1, ?2, ?3)",
                        params![session_id.as_str(), scope_key, scope_json],
                    )
                    .map_err(store_db)?;
                if inserted == 1 {
                    return Ok(session_id);
                }
            }
        })
        .await
    }

    async fn start_run(&self, run: &RunStart<Scope>) -> Result<StoreStartResult, MachineError> {
        let run = run.clone();
        self.call(move |conn| {
            let scope_key = scope_key(&run.scope)?;
            let scope_json = json_text(&run.scope)?;
            let metadata = json_text(&run.metadata)?;
            let sig = start_sig(run.input.as_ref(), &run.entries)?;
            let now = now_ms();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(store_db)?;
            insert_session_tx(&tx, &run.session_id, &scope_key, &scope_json, now)?;
            if let Some(existing) = lookup_run_tx(&tx, &run.run_id, &scope_key)? {
                if run_start_sig_tx(&tx, &existing.run_id)? != sig {
                    return Err(MachineError::StartConflict);
                }
                tx.commit().map_err(store_db)?;
                return Ok(StoreStartResult::Existing(existing));
            }
            if let Some(key) = &run.client_run_key
                && let Some(existing) =
                    find_idempotent_tx(&tx, &scope_key, &run.session_id, key)?
            {
                if run_start_sig_tx(&tx, &existing.run_id)? != sig {
                    return Err(MachineError::StartConflict);
                }
                tx.commit().map_err(store_db)?;
                return Ok(StoreStartResult::Existing(existing));
            }
            if let Some((owner, run_id)) =
                running_thread_tx(&tx, &run.thread_id, Some(&run.run_id))?
            {
                tx.commit().map_err(store_db)?;
                return Err(MachineError::ThreadBusy { owner, run: run_id });
            }

            let parent = run.parent_run_id.as_ref().map(RunId::as_str);
            let retry = run.retry_of_run_id.as_ref().map(RunId::as_str);
            let input = run.input.as_ref().map(json_text).transpose()?;
            let owner = run.lease.as_ref().map(|lease| lease.owner.as_str());
            let lease_id = run.lease.as_ref().map(|lease| lease.id.as_str());
            let until = run
                .lease
                .as_ref()
                .map(|lease| now + duration_ms(lease.ttl));
            let attempt = i64::from(run.lease.is_some());
            tx.execute(
                "INSERT INTO typemach_runs (
                    run_id, session_id, thread_id, scope_key, scope, agent_kind, model,
                    client_run_key, parent_run_id, retry_of_run_id, metadata, input, start_sig, status,
                    owner_id, lease_id, lease_expires_at, attempt, started_at, created_at, updated_at
                 )
                 VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                    ?15, ?16, ?17, ?18, ?19, ?19, ?19
                 )",
                params![
                    run.run_id.as_str(),
                    run.session_id.as_str(),
                    run.thread_id.as_str(),
                    scope_key,
                    scope_json,
                    run.agent_kind,
                    run.model,
                    run.client_run_key,
                    parent,
                    retry,
                    metadata,
                    input,
                    sig,
                    RunStatus::Running.as_str(),
                    owner,
                    lease_id,
                    until,
                    attempt,
                    now,
                ],
            )
            .map_err(store_db)?;
            if let Some(claim) = &run.lease
                && let Err(err) = claim_thread_tx(&tx, &run, claim, now)
            {
                tx.rollback().map_err(store_db)?;
                return Err(err);
            }
            apply_entries_tx(
                &tx,
                &scope_key,
                &run.run_id,
                &run.session_id,
                &run.thread_id,
                &run.entries,
            )?;
            tx.commit().map_err(store_db)?;
            Ok(StoreStartResult::Created)
        })
        .await
    }

    async fn lookup_run(
        &self,
        run_id: &RunId,
        scope: &Scope,
    ) -> Result<Option<RunLookup>, MachineError> {
        let run_id = run_id.clone();
        let scope_key = scope_key(scope)?;
        self.call(move |conn| lookup_run_conn(conn, &run_id, &scope_key))
            .await
    }

    async fn finish_run(
        &self,
        finish: &RunFinishRecord<E, Data, Scope>,
    ) -> Result<FinishRunResult<E>, MachineError> {
        let commit = RunCommit {
            run_id: finish.run_id.clone(),
            session_id: finish.session_id.clone(),
            scope: finish.scope.clone(),
            lease: None,
            checkpoint: None,
            events: vec![finish.terminal_event.clone()],
            effects: Vec::new(),
            items: Vec::new(),
            entries: finish.entries.clone(),
            finish: Some(RunFinish {
                run_id: finish.run_id.clone(),
                session_id: finish.session_id.clone(),
                scope: finish.scope.clone(),
                status: finish.status,
                finish_reason: finish.finish_reason.clone(),
                error_code: finish.error_code.clone(),
                entries: Vec::new(),
                data: finish.data.clone(),
            }),
        };
        match self.commit_run(&commit).await? {
            RunCommitResult::Finished { result, .. } => Ok(result),
            RunCommitResult::Skipped => Err(MachineError::RunNotFound),
            RunCommitResult::Recorded(_) => Err(MachineError::InvalidRunEvent {
                reason: "finish_run did not produce a terminal result".to_string(),
            }),
        }
    }

    async fn terminal_event(
        &self,
        run_id: &RunId,
        scope: &Scope,
    ) -> Result<Option<E>, MachineError> {
        let run_id = run_id.clone();
        let scope_key = scope_key(scope)?;
        self.call(move |conn| {
            let raw = conn
                .query_row(
                    "SELECT event.event
                     FROM typemach_run_events event
                     JOIN typemach_runs run ON run.run_id = event.run_id
                     WHERE event.run_id = ?1 AND run.scope_key = ?2 AND event.terminal = 1
                     ORDER BY event.seq DESC
                     LIMIT 1",
                    params![run_id.as_str(), scope_key],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(store_db)?;
            raw.map(|raw| decode_event(raw.as_str())).transpose()
        })
        .await
    }

    async fn find_idempotent_run(
        &self,
        scope: &Scope,
        session_id: &SessionId,
        key: &str,
    ) -> Result<Option<RunLookup>, MachineError> {
        let scope_key = scope_key(scope)?;
        let session_id = session_id.clone();
        let key = key.to_string();
        self.call(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Deferred)
                .map_err(store_db)?;
            let lookup = find_idempotent_tx(&tx, &scope_key, &session_id, &key)?;
            tx.commit().map_err(store_db)?;
            Ok(lookup)
        })
        .await
    }

    async fn check_run_start(
        &self,
        run_id: &RunId,
        scope: &Scope,
        input: Option<&Value>,
        entries: &[EntryWrite],
    ) -> Result<(), MachineError> {
        let run_id = run_id.clone();
        let scope_key = scope_key(scope)?;
        let sig = start_sig(input, entries)?;
        self.call(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Deferred)
                .map_err(store_db)?;
            if lookup_run_tx(&tx, &run_id, &scope_key)?.is_none() {
                tx.commit().map_err(store_db)?;
                return Err(MachineError::RunNotFound);
            }
            if run_start_sig_tx(&tx, &run_id)? != sig {
                return Err(MachineError::StartConflict);
            }
            tx.commit().map_err(store_db)?;
            Ok(())
        })
        .await
    }

    async fn mark_cancelled(&self, run_id: &RunId, scope: &Scope) -> Result<(), MachineError> {
        let run_id = run_id.clone();
        let scope_key = scope_key(scope)?;
        self.call(move |conn| {
            let updated = conn
                .execute(
                    "UPDATE typemach_runs
                     SET cancel_requested = 1, updated_at = ?3
                     WHERE run_id = ?1 AND scope_key = ?2",
                    params![run_id.as_str(), scope_key, now_ms()],
                )
                .map_err(store_db)?;
            if updated == 0 {
                return Err(MachineError::RunNotFound);
            }
            Ok(())
        })
        .await
    }

    async fn record_event(
        &self,
        run_id: &RunId,
        scope: &Scope,
        event: &E,
    ) -> Result<bool, MachineError> {
        let commit = RunCommit {
            run_id: run_id.clone(),
            session_id: event.session_id().clone(),
            scope: scope.clone(),
            lease: None,
            checkpoint: None,
            events: vec![event.clone()],
            effects: Vec::new(),
            items: Vec::new(),
            entries: Vec::new(),
            finish: None,
        };
        match self.commit_run(&commit).await? {
            RunCommitResult::Recorded(_) => Ok(true),
            RunCommitResult::Skipped => Ok(false),
            RunCommitResult::Finished { .. } => Err(MachineError::InvalidRunEvent {
                reason: "record_event produced a terminal result".to_string(),
            }),
        }
    }

    async fn list_events(
        &self,
        run_id: &RunId,
        scope: &Scope,
        after_seq: i64,
        limit: usize,
    ) -> Result<Page<E>, MachineError> {
        if limit == 0 {
            return Err(MachineError::InvalidPageLimit);
        }
        let run_id = run_id.clone();
        let scope_key = scope_key(scope)?;
        let fetch = limit.saturating_add(1).min(i64::MAX as usize) as i64;
        self.call(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT event.event
                     FROM typemach_run_events event
                     JOIN typemach_runs run ON run.run_id = event.run_id
                     WHERE event.run_id = ?1 AND run.scope_key = ?2 AND event.seq > ?3
                     ORDER BY event.seq ASC
                     LIMIT ?4",
                )
                .map_err(store_db)?;
            let rows = stmt
                .query_map(
                    params![run_id.as_str(), scope_key, after_seq, fetch],
                    |row| row.get::<_, String>(0),
                )
                .map_err(store_db)?;
            let mut events = Vec::new();
            for row in rows {
                let raw = row.map_err(store_db)?;
                events.push(decode_event(raw.as_str())?);
            }
            let next = if events.len() > limit {
                events.truncate(limit);
                events.last().map(RunEvent::seq)
            } else {
                None
            };
            Ok(Page::new(events, next))
        })
        .await
    }
}

mod io;
mod lease;
mod op;
#[cfg(test)]
mod tests;
mod tx;

use io::*;
use op::*;
