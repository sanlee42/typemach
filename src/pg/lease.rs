use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::*;
use crate::store::{Lease, RunLease};

#[async_trait]
impl<E, Scope, Data> RunLease<E> for PgStore<E, Scope, Data>
where
    E: RunEvent + Serialize + DeserializeOwned,
    Scope: Clone + Serialize + Send + Sync + 'static,
    Data: Clone + Serialize + Send + Sync + 'static,
{
    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<bool, MachineError> {
        let mut client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let tx = client.transaction().await.map_err(store_db)?;
        let ttl = ttl.as_secs_f64();
        let updated = tx
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
        if updated != 1 {
            tx.commit().await.map_err(store_db)?;
            return Ok(false);
        }
        let thread_updated = tx
            .execute(
                "UPDATE typemach_thread_leases
                 SET lease_expires_at = now() + ($4::double precision * interval '1 second'),
                     updated_at = now()
                 WHERE run_id = $1
                   AND owner_id = $2
                   AND lease_id = $3
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
        if thread_updated != 1 {
            tx.rollback().await.map_err(store_db)?;
            return Ok(false);
        }
        tx.commit().await.map_err(store_db)?;
        Ok(true)
    }

    async fn release(&self, lease: &Lease) -> Result<(), MachineError> {
        let mut client = self
            .pool
            .get()
            .await
            .map_err(|err| store_msg(format!("acquire pool client: {err}")))?;
        let tx = client.transaction().await.map_err(store_db)?;
        tx.execute(
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
        tx.execute(
            "DELETE FROM typemach_thread_leases
             WHERE run_id = $1 AND owner_id = $2 AND lease_id = $3",
            &[
                &lease.run.as_str(),
                &lease.owner.as_str(),
                &lease.id.as_str(),
            ],
        )
        .await
        .map_err(store_db)?;
        tx.commit().await.map_err(store_db)?;
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
                "SELECT run_id, session_id, thread_id, status, finish_reason, cancel_requested, owner_id,
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
            let seq: i64 = row.get(7);
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
            delete_thread_tx(&tx, &lookup.run_id).await?;
            lookup.status = RunStatus::Error;
            lookup.finish_reason = Some("lease_expired".to_string());
            lookup.owner = None;
            reaped.push(lookup);
        }
        tx.commit().await.map_err(store_db)?;
        Ok(reaped)
    }
}
