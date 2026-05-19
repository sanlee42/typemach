use std::marker::PhantomData;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use deadpool_postgres::tokio_postgres::Row;
use deadpool_postgres::{GenericClient, Pool, Transaction};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::checkpoint::{CheckpointRecord, CheckpointStore};
use crate::error::MachineError;
use crate::run::{LeaseId, RunId, SessionId, WorkerId};
use crate::runtime::Event;
use crate::store::{
    FinishRunResult, Lease, RunCommit, RunCommitResult, RunEvent, RunFinish, RunFinishRecord,
    RunLease, RunLookup, RunStart, RunStatus, RunStore, RunTx, StoreStartResult,
};

type PgTypes<E, Scope, Data> = fn() -> (E, Scope, Data);

/// PostgreSQL run store for `TxRuntime`.
///
/// `PgStore` intentionally does not implement `CheckpointSaver`; checkpoint
/// writes must go through `RunTx::commit_run` so checkpoint and run events stay
/// in one transaction.
///
/// ```compile_fail
/// fn needs_saver<T: typemach::CheckpointSaver>() {}
/// needs_saver::<typemach::PgStore>();
/// ```
pub struct PgStore<E = Event, Scope = Value, Data = ()> {
    pool: Pool,
    _types: PhantomData<PgTypes<E, Scope, Data>>,
}

impl<E, Scope, Data> Clone for PgStore<E, Scope, Data> {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            _types: PhantomData,
        }
    }
}

impl<E, Scope, Data> PgStore<E, Scope, Data> {
    pub fn new(pool: Pool) -> Self {
        Self {
            pool,
            _types: PhantomData,
        }
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    pub async fn ensure_schema(&self) -> Result<(), MachineError> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        client
            .batch_execute(
                "CREATE TABLE IF NOT EXISTS typemach_checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL DEFAULT 1,
                    state JSONB NOT NULL,
                    next_step JSONB NULL,
                    interrupted_step JSONB NULL,
                    interrupt JSONB NULL,
                    run_id TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE TABLE IF NOT EXISTS typemach_sessions (
                    scope_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    scope JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (scope_key, session_id)
                );
                CREATE TABLE IF NOT EXISTS typemach_runs (
                    run_id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    scope JSONB NOT NULL,
                    agent_kind TEXT NOT NULL,
                    model TEXT NULL,
                    client_run_key TEXT NULL,
                    parent_run_id TEXT NULL,
                    retry_of_run_id TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    status TEXT NOT NULL,
                    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    finished_at TIMESTAMPTZ NULL,
                    finish_reason TEXT NULL,
                    error_code TEXT NULL,
                    finish_data JSONB NULL,
                    owner_id TEXT NULL,
                    lease_id TEXT NULL,
                    lease_expires_at TIMESTAMPTZ NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
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
                CREATE UNIQUE INDEX IF NOT EXISTS typemach_runs_idempotency_idx
                    ON typemach_runs (scope_key, session_id, client_run_key)
                    WHERE client_run_key IS NOT NULL;
                CREATE TABLE IF NOT EXISTS typemach_run_events (
                    run_id TEXT NOT NULL REFERENCES typemach_runs(run_id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL,
                    seq BIGINT NOT NULL,
                    terminal BOOLEAN NOT NULL DEFAULT FALSE,
                    event JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (run_id, seq)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS typemach_run_events_terminal_idx
                    ON typemach_run_events (run_id)
                    WHERE terminal;",
            )
            .await
            .map_err(store_db)?;
        Ok(())
    }
}

#[async_trait]
impl<E, Scope, Data> CheckpointStore for PgStore<E, Scope, Data>
where
    E: Send + Sync,
    Scope: Send + Sync,
    Data: Send + Sync,
{
    async fn load_checkpoint(
        &self,
        thread_id: &str,
    ) -> Result<Option<CheckpointRecord>, MachineError> {
        let client =
            self.pool.get().await.map_err(|err| {
                MachineError::CheckpointPool(format!("acquire pool client: {err}"))
            })?;
        load_checkpoint(&client, thread_id)
            .await
            .map_err(checkpoint_db)?
            .map(decode_checkpoint)
            .transpose()
    }
}

#[async_trait]
impl<E, Scope, Data> RunStore<E> for PgStore<E, Scope, Data>
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
        let client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        if let Some(session_id) = session_id {
            client
                .execute(
                    "INSERT INTO typemach_sessions (session_id, scope_key, scope)
                     VALUES ($1, $2, $3::text::jsonb)
                     ON CONFLICT (scope_key, session_id) DO UPDATE
                     SET scope = EXCLUDED.scope",
                    &[&session_id.as_str(), &scope_key, &scope_json],
                )
                .await
                .map_err(store_db)?;
            return Ok(session_id);
        }

        loop {
            let session_id = new_session_id();
            let inserted = client
                .execute(
                    "INSERT INTO typemach_sessions (session_id, scope_key, scope)
                     VALUES ($1, $2, $3::text::jsonb)
                     ON CONFLICT (scope_key, session_id) DO NOTHING",
                    &[&session_id.as_str(), &scope_key, &scope_json],
                )
                .await
                .map_err(store_db)?;
            if inserted == 1 {
                return Ok(session_id);
            }
        }
    }

    async fn start_run(&self, run: &RunStart<Scope>) -> Result<StoreStartResult, MachineError> {
        self.ensure_session(Some(run.session_id.clone()), &run.scope)
            .await?;
        if let Some(existing) = self.lookup_run(&run.run_id, &run.scope).await? {
            return Ok(StoreStartResult::Existing(existing));
        }
        if let Some(client_run_key) = &run.client_run_key
            && let Some(existing) = self
                .find_idempotent_run(&run.scope, &run.session_id, client_run_key)
                .await?
        {
            return Ok(StoreStartResult::Existing(existing));
        }

        let scope_key = scope_key(&run.scope)?;
        let scope_json = json_text(&run.scope)?;
        let metadata = json_text(&run.metadata)?;
        let parent = run.parent_run_id.as_ref().map(RunId::as_str);
        let retry = run.retry_of_run_id.as_ref().map(RunId::as_str);
        let owner = run.lease.as_ref().map(|lease| lease.owner.as_str());
        let lease_id = run.lease.as_ref().map(|lease| lease.id.as_str());
        let ttl = run.lease.as_ref().map(|lease| lease.ttl.as_secs_f64());
        let attempt = i32::from(run.lease.is_some());
        let client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let insert = client
            .execute(
                "INSERT INTO typemach_runs (
                    run_id, session_id, scope_key, scope, agent_kind, model, client_run_key,
                    parent_run_id, retry_of_run_id, metadata, status, owner_id, lease_id,
                    lease_expires_at, attempt, updated_at
                 )
                 VALUES (
                    $1, $2, $3, $4::text::jsonb, $5, $6, $7,
                    $8, $9, $10::text::jsonb, $11, $12, $13,
                    CASE
                        WHEN $13::text IS NULL THEN NULL
                        ELSE now() + ($14::double precision * interval '1 second')
                    END,
                    $15, now()
                 )",
                &[
                    &run.run_id.as_str(),
                    &run.session_id.as_str(),
                    &scope_key,
                    &scope_json,
                    &run.agent_kind,
                    &run.model,
                    &run.client_run_key,
                    &parent,
                    &retry,
                    &metadata,
                    &RunStatus::Running.as_str(),
                    &owner,
                    &lease_id,
                    &ttl,
                    &attempt,
                ],
            )
            .await;
        match insert {
            Ok(_) => Ok(StoreStartResult::Created),
            Err(err) if is_unique_violation(&err) => {
                if let Some(existing) = self.lookup_run(&run.run_id, &run.scope).await? {
                    return Ok(StoreStartResult::Existing(existing));
                }
                if let Some(client_run_key) = &run.client_run_key
                    && let Some(existing) = self
                        .find_idempotent_run(&run.scope, &run.session_id, client_run_key)
                        .await?
                {
                    return Ok(StoreStartResult::Existing(existing));
                }
                Err(store_db(err))
            }
            Err(err) => Err(store_db(err)),
        }
    }

    async fn lookup_run(
        &self,
        run_id: &RunId,
        scope: &Scope,
    ) -> Result<Option<RunLookup>, MachineError> {
        let scope_key = scope_key(scope)?;
        let client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let row = client
            .query_opt(
                "SELECT run_id, session_id, status, finish_reason, cancel_requested, owner_id
                 FROM typemach_runs
                 WHERE run_id = $1 AND scope_key = $2",
                &[&run_id.as_str(), &scope_key],
            )
            .await
            .map_err(store_db)?;
        row.map(|row| row_lookup(&row)).transpose()
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
            finish: Some(RunFinish {
                run_id: finish.run_id.clone(),
                session_id: finish.session_id.clone(),
                scope: finish.scope.clone(),
                status: finish.status,
                finish_reason: finish.finish_reason.clone(),
                error_code: finish.error_code.clone(),
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
        let scope_key = scope_key(scope)?;
        let client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let row = client
            .query_opt(
                "SELECT event::text
                 FROM typemach_run_events event
                 JOIN typemach_runs run ON run.run_id = event.run_id
                 WHERE event.run_id = $1 AND run.scope_key = $2 AND event.terminal
                 ORDER BY event.seq DESC
                 LIMIT 1",
                &[&run_id.as_str(), &scope_key],
            )
            .await
            .map_err(store_db)?;
        row.map(|row| decode_event(row.get::<_, String>(0).as_str()))
            .transpose()
    }

    async fn find_idempotent_run(
        &self,
        scope: &Scope,
        session_id: &SessionId,
        key: &str,
    ) -> Result<Option<RunLookup>, MachineError> {
        let scope_key = scope_key(scope)?;
        let client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let row = client
            .query_opt(
                "SELECT run_id, session_id, status, finish_reason, cancel_requested, owner_id
                 FROM typemach_runs
                 WHERE scope_key = $1 AND session_id = $2 AND client_run_key = $3
                 ORDER BY started_at DESC
                 LIMIT 1",
                &[&scope_key, &session_id.as_str(), &key],
            )
            .await
            .map_err(store_db)?;
        row.map(|row| row_lookup(&row)).transpose()
    }

    async fn mark_cancelled(&self, run_id: &RunId, scope: &Scope) -> Result<(), MachineError> {
        let scope_key = scope_key(scope)?;
        let client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let updated = client
            .execute(
                "UPDATE typemach_runs
                 SET cancel_requested = TRUE, updated_at = now()
                 WHERE run_id = $1 AND scope_key = $2",
                &[&run_id.as_str(), &scope_key],
            )
            .await
            .map_err(store_db)?;
        if updated == 0 {
            return Err(MachineError::RunNotFound);
        }
        Ok(())
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
    ) -> Result<Vec<E>, MachineError> {
        let scope_key = scope_key(scope)?;
        let client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let rows = client
            .query(
                "SELECT event.event::text
                 FROM typemach_run_events event
                 JOIN typemach_runs run ON run.run_id = event.run_id
                 WHERE event.run_id = $1 AND run.scope_key = $2 AND event.seq > $3
                 ORDER BY event.seq ASC",
                &[&run_id.as_str(), &scope_key, &after_seq],
            )
            .await
            .map_err(store_db)?;
        rows.into_iter()
            .map(|row| decode_event(row.get::<_, String>(0).as_str()))
            .collect()
    }
}

#[async_trait]
impl<E, Scope, Data> RunTx<E> for PgStore<E, Scope, Data>
where
    E: RunEvent + Serialize + DeserializeOwned,
    Scope: Clone + Serialize + Send + Sync + 'static,
    Data: Clone + Serialize + Send + Sync + 'static,
{
    async fn commit_run(
        &self,
        commit: &RunCommit<E, Data, Scope>,
    ) -> Result<RunCommitResult<E>, MachineError> {
        if commit.events.is_empty() {
            return Err(MachineError::InvalidRunEvent {
                reason: "commit requires at least one event".to_string(),
            });
        }
        validate_commit(commit)?;

        let scope_key = scope_key(&commit.scope)?;
        let mut client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let tx = client.transaction().await.map_err(store_db)?;
        let Some(row) = tx
            .query_opt(
                "SELECT session_id, status, lease_id,
                        lease_expires_at IS NOT NULL AND lease_expires_at <= now()
                 FROM typemach_runs
                 WHERE run_id = $1 AND scope_key = $2
                 FOR UPDATE",
                &[&commit.run_id.as_str(), &scope_key],
            )
            .await
            .map_err(store_db)?
        else {
            tx.commit().await.map_err(store_db)?;
            return Ok(RunCommitResult::Skipped);
        };

        let stored_session: String = row.get(0);
        if stored_session != commit.session_id.as_str() {
            return Err(MachineError::InvalidRunEvent {
                reason: "event session_id does not match target run".to_string(),
            });
        }
        let status = row_status(&row, 1)?;
        if status.is_terminal() {
            if commit.finish.is_some()
                && let Some(event) = terminal_event_tx::<E>(&tx, &commit.run_id).await?
            {
                tx.commit().await.map_err(store_db)?;
                return Ok(RunCommitResult::Finished {
                    events: vec![event.clone()],
                    result: FinishRunResult::AlreadyFinished(event),
                });
            }
            tx.commit().await.map_err(store_db)?;
            return Ok(RunCommitResult::Skipped);
        }
        check_pg_lease(&row, commit.lease.as_ref())?;

        if let Some(checkpoint) = &commit.checkpoint {
            save_checkpoint(
                &tx,
                checkpoint.thread_id.as_str(),
                &checkpoint.record,
                store_db,
            )
            .await?;
        }

        let mut last_seq = last_seq_tx(&tx, &commit.run_id).await?;
        for event in &commit.events {
            if event.seq() <= last_seq {
                return Err(MachineError::InvalidRunEvent {
                    reason: "event seq must increase monotonically".to_string(),
                });
            }
            insert_event_tx(&tx, event).await?;
            last_seq = event.seq();
        }

        if let Some(finish) = &commit.finish {
            let terminal_event =
                commit
                    .events
                    .last()
                    .cloned()
                    .ok_or_else(|| MachineError::InvalidRunEvent {
                        reason: "finish commit requires a terminal event".to_string(),
                    })?;
            let finish_data = json_text(&finish.data)?;
            tx.execute(
                "UPDATE typemach_runs
                 SET status = $2,
                     finished_at = now(),
                     finish_reason = $3,
                     error_code = $4,
                     finish_data = $5::text::jsonb,
                     owner_id = NULL,
                     lease_id = NULL,
                     lease_expires_at = NULL,
                     updated_at = now()
                 WHERE run_id = $1",
                &[
                    &commit.run_id.as_str(),
                    &finish.status.as_str(),
                    &finish.finish_reason,
                    &finish.error_code,
                    &finish_data,
                ],
            )
            .await
            .map_err(store_db)?;
            tx.commit().await.map_err(store_db)?;
            return Ok(RunCommitResult::Finished {
                events: commit.events.clone(),
                result: FinishRunResult::Finished(terminal_event),
            });
        }

        tx.commit().await.map_err(store_db)?;
        Ok(RunCommitResult::Recorded(commit.events.clone()))
    }
}

#[async_trait]
impl<E, Scope, Data> RunLease<E> for PgStore<E, Scope, Data>
where
    E: RunEvent + Serialize + DeserializeOwned,
    Scope: Clone + Serialize + Send + Sync + 'static,
    Data: Clone + Serialize + Send + Sync + 'static,
{
    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<bool, MachineError> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let ttl = ttl.as_secs_f64();
        let updated = client
            .execute(
                "UPDATE typemach_runs
                 SET lease_expires_at = now() + ($4::double precision * interval '1 second'),
                     updated_at = now()
                 WHERE run_id = $1
                   AND owner_id = $2
                   AND lease_id = $3
                   AND status = 'running'
                   AND lease_expires_at > now()",
                &[
                    &lease.run.as_str(),
                    &lease.owner.as_str(),
                    &lease.id.as_str(),
                    &ttl,
                ],
            )
            .await
            .map_err(store_db)?;
        Ok(updated == 1)
    }

    async fn release(&self, lease: &Lease) -> Result<(), MachineError> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        client
            .execute(
                "UPDATE typemach_runs
                 SET owner_id = NULL,
                     lease_id = NULL,
                     lease_expires_at = NULL,
                     updated_at = now()
                 WHERE run_id = $1
                   AND owner_id = $2
                   AND lease_id = $3
                   AND status = 'running'",
                &[
                    &lease.run.as_str(),
                    &lease.owner.as_str(),
                    &lease.id.as_str(),
                ],
            )
            .await
            .map_err(store_db)?;
        Ok(())
    }

    async fn reap_stale<F>(
        &self,
        _owner: &WorkerId,
        limit: usize,
        mut build_event: F,
    ) -> Result<Vec<RunLookup>, MachineError>
    where
        F: FnMut(&RunLookup, i64) -> E + Send,
    {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let tx = client.transaction().await.map_err(store_db)?;
        let limit = limit as i64;
        let rows = tx
            .query(
                "SELECT run_id, session_id, status, finish_reason, cancel_requested, owner_id,
                        COALESCE(
                            (SELECT MAX(seq) FROM typemach_run_events event
                             WHERE event.run_id = typemach_runs.run_id),
                            0
                        ) + 1 AS next_seq
                 FROM typemach_runs
                 WHERE status = 'running'
                   AND lease_expires_at IS NOT NULL
                   AND lease_expires_at <= now()
                 ORDER BY lease_expires_at ASC
                 LIMIT $1
                 FOR UPDATE SKIP LOCKED",
                &[&limit],
            )
            .await
            .map_err(store_db)?;

        let mut reaped = Vec::with_capacity(rows.len());
        for row in rows {
            let mut lookup = row_lookup(&row)?;
            let seq: i64 = row.get(6);
            let event = build_event(&lookup, seq);
            if event.run_id() != &lookup.run_id || event.session_id() != &lookup.session_id {
                return Err(MachineError::InvalidRunEvent {
                    reason: "reap event target does not match run".to_string(),
                });
            }
            if event.seq() != seq || !event.is_terminal() {
                return Err(MachineError::InvalidRunEvent {
                    reason: "reap requires the next terminal event".to_string(),
                });
            }
            insert_event_tx(&tx, &event).await?;
            tx.execute(
                "UPDATE typemach_runs
                 SET status = $2,
                     finished_at = now(),
                     finish_reason = $3,
                     error_code = $4,
                     finish_data = NULL,
                     owner_id = NULL,
                     lease_id = NULL,
                     lease_expires_at = NULL,
                     updated_at = now()
                 WHERE run_id = $1",
                &[
                    &lookup.run_id.as_str(),
                    &RunStatus::Error.as_str(),
                    &"lease_expired",
                    &"lease_lost",
                ],
            )
            .await
            .map_err(store_db)?;
            lookup.status = RunStatus::Error;
            lookup.finish_reason = Some("lease_expired".to_string());
            lookup.owner = None;
            reaped.push(lookup);
        }
        tx.commit().await.map_err(store_db)?;
        Ok(reaped)
    }
}

async fn save_checkpoint<C>(
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

async fn load_checkpoint<C>(
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

fn decode_checkpoint(row: Row) -> Result<CheckpointRecord, MachineError> {
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

fn validate_commit<E, Data, Scope>(commit: &RunCommit<E, Data, Scope>) -> Result<(), MachineError>
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

async fn terminal_event_tx<E>(
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

async fn last_seq_tx(tx: &Transaction<'_>, run_id: &RunId) -> Result<i64, MachineError> {
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

async fn insert_event_tx<E>(tx: &Transaction<'_>, event: &E) -> Result<(), MachineError>
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

fn row_lookup(row: &Row) -> Result<RunLookup, MachineError> {
    Ok(RunLookup {
        run_id: RunId::from(row.get::<_, String>(0)),
        session_id: SessionId::from(row.get::<_, String>(1)),
        status: row_status(row, 2)?,
        finish_reason: row.get(3),
        cancel_requested: row.get(4),
        owner: row.get::<_, Option<String>>(5).map(WorkerId::from),
    })
}

fn row_status(row: &Row, index: usize) -> Result<RunStatus, MachineError> {
    let status: String = row.get(index);
    RunStatus::parse(&status).ok_or_else(|| MachineError::InvalidRunEvent {
        reason: format!("invalid stored run status: {status}"),
    })
}

fn check_pg_lease(row: &Row, lease: Option<&LeaseId>) -> Result<(), MachineError> {
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

fn decode_event<E>(raw: &str) -> Result<E, MachineError>
where
    E: DeserializeOwned,
{
    serde_json::from_str(raw).map_err(MachineError::Deserialization)
}

fn json_text<T>(value: &T) -> Result<String, MachineError>
where
    T: Serialize,
{
    serde_json::to_string(value).map_err(MachineError::Serialization)
}

fn scope_key<Scope>(scope: &Scope) -> Result<String, MachineError>
where
    Scope: Serialize,
{
    let value = serde_json::to_value(scope).map_err(MachineError::Serialization)?;
    serde_json::to_string(&value).map_err(MachineError::Serialization)
}

fn new_session_id() -> SessionId {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    SessionId::from(format!("session-{}-{nanos}", std::process::id()))
}

fn is_unique_violation(err: &deadpool_postgres::tokio_postgres::Error) -> bool {
    err.code().is_some_and(|code| {
        *code == deadpool_postgres::tokio_postgres::error::SqlState::UNIQUE_VIOLATION
    })
}

fn checkpoint_db(err: deadpool_postgres::tokio_postgres::Error) -> MachineError {
    MachineError::CheckpointDb(Box::new(err))
}

fn store_db(err: deadpool_postgres::tokio_postgres::Error) -> MachineError {
    MachineError::StoreDb(Box::new(err))
}

fn store_msg(message: String) -> MachineError {
    MachineError::StoreDb(Box::new(std::io::Error::other(message)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use deadpool_postgres::Runtime;
    use deadpool_postgres::tokio_postgres::NoTls;

    use crate::runtime::Payload;
    use crate::store::{LeaseClaim, RunEventEnvelope};

    #[test]
    fn pg_store_roundtrip_skips_without_test_database_url() {
        let Some(url) = std::env::var("TEST_DATABASE_URL").ok() else {
            return;
        };
        if !url.to_ascii_lowercase().contains("test") || url.ends_with("/postgres") {
            panic!("refusing to run typemach pg test against non-test database");
        }

        block_on(async {
            let store = PgStore::<Event>::new(pool(url));
            reset_schema(&store).await;
            store.ensure_schema().await.expect("schema");
            let scope = serde_json::json!({"tenant": "typemach-test"});
            let run_id = RunId::from(format!("run-{}", unique()));
            let session_id = SessionId::from(format!("session-{}", unique()));
            store
                .start_run(&RunStart {
                    run_id: run_id.clone(),
                    session_id: session_id.clone(),
                    agent_kind: "test".to_string(),
                    model: None,
                    client_run_key: Some("key-a".to_string()),
                    parent_run_id: None,
                    retry_of_run_id: None,
                    scope: scope.clone(),
                    metadata: serde_json::json!({}),
                    lease: None,
                })
                .await
                .expect("start");

            let checkpoint = CheckpointRecord::running(
                serde_json::json!({"value": 1}),
                Some(serde_json::json!("done")),
                run_id.as_str(),
            );
            let event = RunEventEnvelope::new(
                run_id.clone(),
                session_id.clone(),
                1,
                Payload::StepDone {
                    step: serde_json::json!("start"),
                    result: crate::run::StepResult::Next,
                },
            );
            let result = store
                .commit_run(&RunCommit {
                    run_id: run_id.clone(),
                    session_id: session_id.clone(),
                    scope: scope.clone(),
                    lease: None,
                    checkpoint: Some(crate::store::CheckpointWrite::new(
                        crate::run::ThreadId::from(format!("thread-{}", unique())),
                        checkpoint.clone(),
                    )),
                    events: vec![event],
                    finish: None,
                })
                .await
                .expect("commit");
            assert!(matches!(result, RunCommitResult::Recorded(_)));
            assert_eq!(
                store
                    .list_events(&run_id, &scope, 0)
                    .await
                    .expect("events")
                    .len(),
                1
            );

            let lease_run = RunId::from(format!("run-lease-{}", unique()));
            let lease_session = SessionId::from(format!("session-lease-{}", unique()));
            let owner = WorkerId::from("worker-a");
            let lease_id = LeaseId::from("lease-a");
            store
                .start_run(&RunStart {
                    run_id: lease_run.clone(),
                    session_id: lease_session.clone(),
                    agent_kind: "test".to_string(),
                    model: None,
                    client_run_key: None,
                    parent_run_id: None,
                    retry_of_run_id: None,
                    scope: scope.clone(),
                    metadata: serde_json::json!({}),
                    lease: Some(LeaseClaim::new(
                        owner.clone(),
                        lease_id.clone(),
                        Duration::from_secs(30),
                    )),
                })
                .await
                .expect("start leased");
            let leased_event = RunEventEnvelope::new(
                lease_run.clone(),
                lease_session.clone(),
                1,
                Payload::Beat {
                    thread_id: crate::run::ThreadId::from("thread-lease"),
                },
            );
            let missing_lease = RunCommit {
                run_id: lease_run.clone(),
                session_id: lease_session.clone(),
                scope: scope.clone(),
                lease: None,
                checkpoint: None,
                events: vec![leased_event.clone()],
                finish: None,
            };
            assert!(matches!(
                store.commit_run(&missing_lease).await,
                Err(MachineError::LeaseLost)
            ));
            assert!(matches!(
                store
                    .commit_run(&RunCommit {
                        lease: Some(LeaseId::from("wrong-lease")),
                        ..missing_lease.clone()
                    })
                    .await,
                Err(MachineError::LeaseLost)
            ));
            assert!(
                store
                    .renew(
                        &Lease::new(lease_run.clone(), owner.clone(), lease_id.clone()),
                        Duration::from_secs(30)
                    )
                    .await
                    .expect("renew")
            );
            assert!(matches!(
                store
                    .commit_run(&RunCommit {
                        lease: Some(lease_id),
                        ..missing_lease
                    })
                    .await
                    .expect("leased commit"),
                RunCommitResult::Recorded(_)
            ));

            let stale_run = RunId::from(format!("run-stale-{}", unique()));
            let stale_session = SessionId::from(format!("session-stale-{}", unique()));
            store
                .start_run(&RunStart {
                    run_id: stale_run.clone(),
                    session_id: stale_session.clone(),
                    agent_kind: "test".to_string(),
                    model: None,
                    client_run_key: None,
                    parent_run_id: None,
                    retry_of_run_id: None,
                    scope: scope.clone(),
                    metadata: serde_json::json!({}),
                    lease: Some(LeaseClaim::new(
                        WorkerId::from("worker-stale"),
                        LeaseId::from("lease-stale"),
                        Duration::from_millis(1),
                    )),
                })
                .await
                .expect("start stale");
            async_rt::time::sleep(Duration::from_millis(5)).await;
            let reaped = store
                .reap_stale(&WorkerId::from("reaper"), 8, |run, seq| {
                    RunEventEnvelope::new(
                        run.run_id.clone(),
                        run.session_id.clone(),
                        seq,
                        Payload::Fail {
                            error: "lease expired".to_string(),
                        },
                    )
                })
                .await
                .expect("reap stale");
            assert_eq!(reaped.len(), 1);
            assert_eq!(reaped[0].run_id, stale_run);
            assert_eq!(reaped[0].status, RunStatus::Error);

            let shared_session = SessionId::from(format!("session-shared-{}", unique()));
            let scope_a = serde_json::json!({"tenant": "tenant-a"});
            let scope_b = serde_json::json!({"tenant": "tenant-b"});
            let run_a = RunId::from(format!("run-a-{}", unique()));
            let run_b = RunId::from(format!("run-b-{}", unique()));
            assert!(matches!(
                store
                    .start_run(&RunStart {
                        run_id: run_a.clone(),
                        session_id: shared_session.clone(),
                        agent_kind: "test".to_string(),
                        model: None,
                        client_run_key: Some("same-key".to_string()),
                        parent_run_id: None,
                        retry_of_run_id: None,
                        scope: scope_a.clone(),
                        metadata: serde_json::json!({}),
                        lease: None,
                    })
                    .await
                    .expect("start scope a"),
                StoreStartResult::Created
            ));
            assert!(matches!(
                store
                    .start_run(&RunStart {
                        run_id: run_b.clone(),
                        session_id: shared_session.clone(),
                        agent_kind: "test".to_string(),
                        model: None,
                        client_run_key: Some("same-key".to_string()),
                        parent_run_id: None,
                        retry_of_run_id: None,
                        scope: scope_b.clone(),
                        metadata: serde_json::json!({}),
                        lease: None,
                    })
                    .await
                    .expect("start scope b"),
                StoreStartResult::Created
            ));
            assert_eq!(
                store
                    .find_idempotent_run(&scope_a, &shared_session, "same-key")
                    .await
                    .expect("idem a")
                    .expect("run a")
                    .run_id,
                run_a
            );
            assert_eq!(
                store
                    .find_idempotent_run(&scope_b, &shared_session, "same-key")
                    .await
                    .expect("idem b")
                    .expect("run b")
                    .run_id,
                run_b
            );
        });
    }

    #[test]
    fn scope_key_is_stable_for_unordered_maps() {
        let mut left = HashMap::new();
        left.insert("tenant", "demo");
        left.insert("shop", "north");
        let mut right = HashMap::new();
        right.insert("shop", "north");
        right.insert("tenant", "demo");

        assert_eq!(
            scope_key(&left).expect("left"),
            scope_key(&right).expect("right")
        );
    }

    async fn reset_schema(store: &PgStore<Event>) {
        let client = store.pool().get().await.expect("pool client");
        client
            .batch_execute(
                "DROP TABLE IF EXISTS typemach_run_events CASCADE;
                 DROP TABLE IF EXISTS typemach_runs CASCADE;
                 DROP TABLE IF EXISTS typemach_sessions CASCADE;
                 DROP TABLE IF EXISTS typemach_checkpoints CASCADE;",
            )
            .await
            .expect("reset schema");
    }

    fn pool(url: String) -> Pool {
        let mut cfg = deadpool_postgres::Config::new();
        cfg.url = Some(url);
        cfg.create_pool(Some(Runtime::Tokio1), NoTls)
            .expect("create pool")
    }

    fn unique() -> String {
        format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        )
    }

    fn block_on<F>(future: F) -> F::Output
    where
        F: std::future::Future,
    {
        async_rt::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(future)
    }
}
