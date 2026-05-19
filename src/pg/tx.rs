use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::*;
use crate::store::RunTx;

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
                        lease_expires_at IS NOT NULL AND lease_expires_at <= now(),
                        thread_id
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
            check_thread_tx(&tx, &row, commit, checkpoint).await?;
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
            delete_thread_tx(&tx, &commit.run_id).await?;
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
