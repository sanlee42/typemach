use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio_rusqlite::rusqlite::{OptionalExtension, TransactionBehavior, params};

use super::*;
use crate::store::{Lease, RunLease};

#[async_trait]
impl<E, Scope, Data> RunLease<E> for SqliteStore<E, Scope, Data>
where
    E: RunEvent + Serialize + DeserializeOwned,
    Scope: Clone + Serialize + Send + Sync + 'static,
    Data: Clone + Serialize + Send + Sync + 'static,
{
    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<bool, MachineError> {
        let lease = lease.clone();
        self.call(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(store_db)?;
            let now = now_ms();
            let until = now + duration_ms(ttl);
            let updated = tx
                .execute(
                    "UPDATE typemach_runs
                     SET lease_expires_at = ?4, updated_at = ?5
                     WHERE run_id = ?1
                       AND owner_id = ?2
                       AND lease_id = ?3
                       AND status = 'running'
                       AND lease_expires_at > ?5",
                    params![
                        lease.run.as_str(),
                        lease.owner.as_str(),
                        lease.id.as_str(),
                        until,
                        now,
                    ],
                )
                .map_err(store_db)?;
            if updated != 1 {
                tx.commit().map_err(store_db)?;
                return Ok(false);
            }
            let thread_updated = tx
                .execute(
                    "UPDATE typemach_thread_leases
                     SET lease_expires_at = ?4, updated_at = ?5
                     WHERE run_id = ?1
                       AND owner_id = ?2
                       AND lease_id = ?3
                       AND lease_expires_at > ?5",
                    params![
                        lease.run.as_str(),
                        lease.owner.as_str(),
                        lease.id.as_str(),
                        until,
                        now,
                    ],
                )
                .map_err(store_db)?;
            if thread_updated != 1 {
                tx.rollback().map_err(store_db)?;
                return Ok(false);
            }
            tx.commit().map_err(store_db)?;
            Ok(true)
        })
        .await
    }

    async fn release(&self, lease: &Lease) -> Result<(), MachineError> {
        let lease = lease.clone();
        self.call(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(store_db)?;
            tx.execute(
                "UPDATE typemach_runs
                 SET owner_id = NULL,
                     lease_id = NULL,
                     lease_expires_at = NULL,
                     updated_at = ?4
                 WHERE run_id = ?1
                   AND owner_id = ?2
                   AND lease_id = ?3
                   AND status = 'running'",
                params![
                    lease.run.as_str(),
                    lease.owner.as_str(),
                    lease.id.as_str(),
                    now_ms(),
                ],
            )
            .map_err(store_db)?;
            tx.execute(
                "DELETE FROM typemach_thread_leases
                 WHERE run_id = ?1 AND owner_id = ?2 AND lease_id = ?3",
                params![lease.run.as_str(), lease.owner.as_str(), lease.id.as_str()],
            )
            .map_err(store_db)?;
            tx.commit().map_err(store_db)?;
            Ok(())
        })
        .await
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
        let rows = self
            .call(move |conn| {
                let limit = limit as i64;
                let tx = conn
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(store_db)?;
                let now = now_ms();
                let rows = {
                    let mut stmt = tx
                        .prepare(
                            "SELECT run_id, session_id, thread_id, status, finish_reason,
                                    cancel_requested, owner_id,
                                    COALESCE(
                                        (SELECT MAX(seq) FROM typemach_run_events event
                                         WHERE event.run_id = typemach_runs.run_id),
                                        0
                                    ) + 1 AS next_seq
                             FROM typemach_runs
                             WHERE status = 'running'
                               AND lease_expires_at IS NOT NULL
                               AND lease_expires_at <= ?1
                             ORDER BY lease_expires_at ASC
                             LIMIT ?2",
                        )
                        .map_err(store_db)?;
                    let rows = stmt
                        .query_map(params![now, limit], |row| {
                            Ok((lookup_row(row)?, row.get::<_, i64>(7)?))
                        })
                        .map_err(store_db)?;
                    let mut rows_out = Vec::new();
                    for row in rows {
                        let (raw, seq) = row.map_err(store_db)?;
                        rows_out.push((raw.into_lookup()?, seq));
                    }
                    rows_out
                };
                tx.commit().map_err(store_db)?;
                Ok(rows)
            })
            .await?;

        let mut events = Vec::with_capacity(rows.len());
        for (lookup, seq) in rows {
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
            events.push((lookup, event));
        }

        self.call(move |conn| {
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(store_db)?;
            let now = now_ms();
            let mut reaped = Vec::with_capacity(events.len());
            for (mut lookup, event) in events {
                let current_seq = tx
                    .query_row(
                        "SELECT COALESCE(
                            (SELECT MAX(seq) FROM typemach_run_events event
                             WHERE event.run_id = typemach_runs.run_id),
                            0
                         ) + 1
                         FROM typemach_runs
                         WHERE run_id = ?1
                           AND status = 'running'
                           AND lease_expires_at IS NOT NULL
                           AND lease_expires_at <= ?2",
                        params![lookup.run_id.as_str(), now],
                        |row| row.get::<_, i64>(0),
                    )
                    .optional()
                    .map_err(store_db)?;
                let Some(current_seq) = current_seq else {
                    continue;
                };
                if event.seq() != current_seq {
                    return Err(MachineError::InvalidRunEvent {
                        reason: "reap requires the next terminal event".to_string(),
                    });
                }
                insert_event_tx(&tx, &event)?;
                tx.execute(
                    "UPDATE typemach_runs
                     SET status = ?2,
                         finished_at = ?3,
                         finish_reason = ?4,
                         error_code = ?5,
                         finish_data = NULL,
                         owner_id = NULL,
                         lease_id = NULL,
                         lease_expires_at = NULL,
                         updated_at = ?3
                     WHERE run_id = ?1",
                    params![
                        lookup.run_id.as_str(),
                        RunStatus::Error.as_str(),
                        now_ms(),
                        "lease_expired",
                        "lease_lost",
                    ],
                )
                .map_err(store_db)?;
                delete_thread_tx(&tx, &lookup.run_id)?;
                lookup.status = RunStatus::Error;
                lookup.finish_reason = Some("lease_expired".to_string());
                lookup.owner = None;
                reaped.push(lookup);
            }
            tx.commit().map_err(store_db)?;
            Ok(reaped)
        })
        .await
    }
}
