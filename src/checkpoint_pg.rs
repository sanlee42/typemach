use crate::checkpoint::{CheckpointRecord, CheckpointSaver};
use crate::error::MachineError;
use async_trait::async_trait;
use deadpool_postgres::Pool;
use serde_json::Value;

/// PostgreSQL-backed checkpoint saver.
///
/// Schema:
///   CREATE TABLE IF NOT EXISTS typemach_checkpoints (
///     thread_id TEXT PRIMARY KEY,
///     version INTEGER NOT NULL,
///     state JSONB NOT NULL,
///     next_step JSONB NULL,
///     interrupted_step JSONB NULL,
///     interrupt JSONB NULL,
///     run_id TEXT NULL,
///     created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
///     updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
///   );
pub struct PostgresSaver {
    pool: Pool,
}

impl PostgresSaver {
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    pub async fn ensure_schema(&self) -> Result<(), MachineError> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| MachineError::CheckpointPool(format!("acquire pool client: {e}")))?;
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
                )",
            )
            .await
            .map_err(|e| MachineError::CheckpointPool(format!("create table: {e}")))?;
        client
            .batch_execute(
                "ALTER TABLE typemach_checkpoints
                   ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;
                 ALTER TABLE typemach_checkpoints
                   ADD COLUMN IF NOT EXISTS next_step JSONB NULL;
                 ALTER TABLE typemach_checkpoints
                   ADD COLUMN IF NOT EXISTS interrupted_step JSONB NULL;
                 ALTER TABLE typemach_checkpoints
                   ADD COLUMN IF NOT EXISTS run_id TEXT NULL;",
            )
            .await
            .map_err(|e| MachineError::CheckpointPool(format!("migrate table: {e}")))?;
        Ok(())
    }
}

#[async_trait]
impl CheckpointSaver for PostgresSaver {
    async fn save(
        &self,
        thread_id: &str,
        checkpoint: &CheckpointRecord,
    ) -> Result<(), MachineError> {
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
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| MachineError::CheckpointPool(format!("acquire pool client: {e}")))?;
        client
            .execute(
                "INSERT INTO typemach_checkpoints (
                    thread_id, version, state, next_step, interrupted_step, interrupt, run_id, updated_at
                 )
                 VALUES (
                    $1,
                    $2,
                    $3::text::jsonb,
                    $4::text::jsonb,
                    $5::text::jsonb,
                    $6::text::jsonb,
                    $7,
                    now()
                 )
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
            .map_err(|e| MachineError::CheckpointPool(format!("save checkpoint: {e}")))?;
        Ok(())
    }

    async fn load(&self, thread_id: &str) -> Result<Option<CheckpointRecord>, MachineError> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| MachineError::CheckpointPool(format!("acquire pool client: {e}")))?;
        let result = client
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
            .map_err(|e| MachineError::CheckpointPool(format!("load checkpoint: {e}")))?;
        match result {
            Some(row) => {
                let version: i32 = row.get(0);
                let raw: String = row.get(1);
                let raw_next_step: Option<String> = row.get(2);
                let raw_interrupted_step: Option<String> = row.get(3);
                let raw_interrupt: Option<String> = row.get(4);
                let run_id: Option<String> = row.get(5);
                let state: Value =
                    serde_json::from_str(&raw).map_err(MachineError::Deserialization)?;
                let next_step = raw_next_step
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()
                    .map_err(MachineError::Deserialization)?;
                let interrupted_step = raw_interrupted_step
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()
                    .map_err(MachineError::Deserialization)?;
                let interrupt = raw_interrupt
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()
                    .map_err(MachineError::Deserialization)?;
                Ok(Some(CheckpointRecord {
                    version: version as u32,
                    state,
                    next_step,
                    interrupted_step,
                    interrupt,
                    run_id,
                }))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deadpool_postgres::Runtime;
    use deadpool_postgres::tokio_postgres::NoTls;

    #[test]
    fn postgres_checkpoint_roundtrip_skips_without_test_database_url() {
        let Some(url) = std::env::var("TEST_DATABASE_URL").ok() else {
            return;
        };
        if !url.to_ascii_lowercase().contains("test") || url.ends_with("/postgres") {
            panic!("refusing to run typemach checkpoint test against non-test database");
        }

        block_on(async {
            let mut cfg = deadpool_postgres::Config::new();
            cfg.url = Some(url);
            let pool = cfg
                .create_pool(Some(Runtime::Tokio1), NoTls)
                .expect("create postgres pool");
            let saver = PostgresSaver::new(pool);
            saver.ensure_schema().await.expect("ensure schema");

            let thread_id = format!(
                "typemach-test-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system time")
                    .as_nanos()
            );
            let checkpoint = CheckpointRecord::interrupted(
                serde_json::json!({"value": 1}),
                serde_json::json!("Clarify"),
                serde_json::json!("resume"),
                "run-1",
            );
            saver
                .save(&thread_id, &checkpoint)
                .await
                .expect("save checkpoint");
            let loaded = saver
                .load(&thread_id)
                .await
                .expect("load checkpoint")
                .expect("checkpoint exists");
            assert_eq!(loaded.state, serde_json::json!({"value": 1}));
            assert_eq!(loaded.interrupted_step, Some(serde_json::json!("Clarify")));
            assert_eq!(loaded.interrupt, Some(serde_json::json!("resume")));
        });
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
