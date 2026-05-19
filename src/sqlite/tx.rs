use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio_rusqlite::rusqlite::{TransactionBehavior, params};

use super::*;
use crate::store::RunTx;

#[async_trait]
impl<E, Scope, Data> RunTx<E> for SqliteStore<E, Scope, Data>
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
        let commit = commit.clone();
        self.call(move |conn| {
            let scope_key = scope_key(&commit.scope)?;
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(store_db)?;
            let Some(row) = commit_row_tx(&tx, &commit.run_id, &scope_key)? else {
                tx.commit().map_err(store_db)?;
                return Ok(RunCommitResult::Skipped);
            };
            if row.session_id != commit.session_id {
                return Err(MachineError::InvalidRunEvent {
                    reason: "event session_id does not match target run".to_string(),
                });
            }
            if row.status.is_terminal() {
                if commit.finish.is_some()
                    && let Some(event) = terminal_event_tx::<E>(&tx, &commit.run_id)?
                {
                    tx.commit().map_err(store_db)?;
                    return Ok(RunCommitResult::Finished {
                        events: vec![event.clone()],
                        result: FinishRunResult::AlreadyFinished(event),
                    });
                }
                tx.commit().map_err(store_db)?;
                return Ok(RunCommitResult::Skipped);
            }
            check_sqlite_lease(&row, commit.lease.as_ref())?;
            if let Some(checkpoint) = &commit.checkpoint {
                check_thread_tx(&tx, &row, &commit, checkpoint)?;
                save_checkpoint_tx(&tx, checkpoint.thread_id.as_str(), &checkpoint.record)?;
            }
            let mut last_seq = last_seq_tx(&tx, &commit.run_id)?;
            for event in &commit.events {
                if event.seq() <= last_seq {
                    return Err(MachineError::InvalidRunEvent {
                        reason: "event seq must increase monotonically".to_string(),
                    });
                }
                insert_event_tx(&tx, event)?;
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
                let now = now_ms();
                let finish_data = json_text(&finish.data)?;
                tx.execute(
                    "UPDATE typemach_runs
                     SET status = ?2,
                         finished_at = ?3,
                         finish_reason = ?4,
                         error_code = ?5,
                         finish_data = ?6,
                         owner_id = NULL,
                         lease_id = NULL,
                         lease_expires_at = NULL,
                         updated_at = ?3
                     WHERE run_id = ?1",
                    params![
                        commit.run_id.as_str(),
                        finish.status.as_str(),
                        now,
                        finish.finish_reason,
                        finish.error_code,
                        finish_data,
                    ],
                )
                .map_err(store_db)?;
                delete_thread_tx(&tx, &commit.run_id)?;
                tx.commit().map_err(store_db)?;
                return Ok(RunCommitResult::Finished {
                    events: commit.events.clone(),
                    result: FinishRunResult::Finished(terminal_event),
                });
            }
            tx.commit().map_err(store_db)?;
            Ok(RunCommitResult::Recorded(commit.events.clone()))
        })
        .await
    }
}
