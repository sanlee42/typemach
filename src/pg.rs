use std::marker::PhantomData;

use async_trait::async_trait;
use deadpool_postgres::Pool;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::checkpoint::{CheckpointRecord, CheckpointStore};
use crate::error::MachineError;
use crate::run::{RunId, SessionId, ThreadId, WorkerId};
use crate::runtime::Event;
use crate::store::{
    FinishRunResult, RunCommit, RunCommitResult, RunEvent, RunFinish, RunFinishRecord, RunLookup,
    RunStart, RunStatus, RunStore, RunTx, StoreStartResult,
};

type PgTypes<E, Scope, Data> = fn() -> (E, Scope, Data);
type BusyThread = (Option<WorkerId>, Option<RunId>);

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

    async fn running_thread(
        &self,
        thread_id: &ThreadId,
    ) -> Result<Option<BusyThread>, MachineError> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        running_thread_tx(&client, thread_id, None).await
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
                    thread_id TEXT NOT NULL,
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
                    lease_expires_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
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
        let mut client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let tx = client.transaction().await.map_err(store_db)?;
        if let Some((owner, run_id)) =
            running_thread_tx(&tx, &run.thread_id, Some(&run.run_id)).await?
        {
            tx.commit().await.map_err(store_db)?;
            return Err(MachineError::ThreadBusy { owner, run: run_id });
        }
        let insert = tx
            .execute(
                "INSERT INTO typemach_runs (
                    run_id, session_id, thread_id, scope_key, scope, agent_kind, model, client_run_key,
                    parent_run_id, retry_of_run_id, metadata, status, owner_id, lease_id,
                    lease_expires_at, attempt, updated_at
                 )
                 VALUES (
                    $1, $2, $3, $4, $5::text::jsonb, $6, $7, $8,
                    $9, $10, $11::text::jsonb, $12, $13, $14,
                    CASE
                        WHEN $14::text IS NULL THEN NULL
                        ELSE now() + ($15::double precision * interval '1 second')
                    END,
                    $16, now()
                 )",
                &[
                    &run.run_id.as_str(),
                    &run.session_id.as_str(),
                    &run.thread_id.as_str(),
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
            Ok(_) => {
                if let Some(claim) = &run.lease
                    && let Err(err) = claim_thread_tx(&tx, run, claim).await
                {
                    tx.rollback().await.map_err(store_db)?;
                    return Err(err);
                }
                tx.commit().await.map_err(store_db)?;
                Ok(StoreStartResult::Created)
            }
            Err(err) if is_unique_violation(&err) => {
                tx.rollback().await.map_err(store_db)?;
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
                if let Some((owner, run_id)) = self.running_thread(&run.thread_id).await? {
                    return Err(MachineError::ThreadBusy { owner, run: run_id });
                }
                Err(store_db(err))
            }
            Err(err) => {
                tx.rollback().await.map_err(store_db)?;
                Err(store_db(err))
            }
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
                "SELECT run_id, session_id, thread_id, status, finish_reason, cancel_requested, owner_id
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
                "SELECT run_id, session_id, thread_id, status, finish_reason, cancel_requested, owner_id
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

mod io;
mod lease;
#[cfg(test)]
mod tests;
mod tx;

use io::*;
