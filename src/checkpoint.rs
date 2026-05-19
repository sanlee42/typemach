use crate::error::MachineError;
use async_rt::sync::Mutex;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointRecord {
    #[serde(default = "checkpoint_version")]
    pub version: u32,
    pub state: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_step: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interrupted_step: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interrupt: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
}

impl CheckpointRecord {
    pub fn running(state: Value, next_step: Option<Value>, run_id: impl Into<String>) -> Self {
        Self {
            version: checkpoint_version(),
            state,
            next_step,
            interrupted_step: None,
            interrupt: None,
            run_id: Some(run_id.into()),
        }
    }

    pub fn interrupted(
        state: Value,
        interrupted_step: Value,
        interrupt: Value,
        run_id: impl Into<String>,
    ) -> Self {
        Self {
            version: checkpoint_version(),
            state,
            next_step: None,
            interrupted_step: Some(interrupted_step),
            interrupt: Some(interrupt),
            run_id: Some(run_id.into()),
        }
    }
}

fn checkpoint_version() -> u32 {
    1
}

#[async_trait]
pub trait CheckpointSaver: Send + Sync {
    async fn save(
        &self,
        thread_id: &str,
        checkpoint: &CheckpointRecord,
    ) -> Result<(), MachineError>;
    async fn load(&self, thread_id: &str) -> Result<Option<CheckpointRecord>, MachineError>;
}

#[async_trait]
pub trait CheckpointStore: Send + Sync {
    async fn load_checkpoint(
        &self,
        thread_id: &str,
    ) -> Result<Option<CheckpointRecord>, MachineError>;
}

#[async_trait]
impl<T> CheckpointStore for T
where
    T: CheckpointSaver + ?Sized,
{
    async fn load_checkpoint(
        &self,
        thread_id: &str,
    ) -> Result<Option<CheckpointRecord>, MachineError> {
        self.load(thread_id).await
    }
}

pub struct MemorySaver {
    checkpoints: Mutex<HashMap<String, CheckpointRecord>>,
}

impl MemorySaver {
    pub fn new() -> Self {
        Self {
            checkpoints: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for MemorySaver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CheckpointSaver for MemorySaver {
    async fn save(
        &self,
        thread_id: &str,
        checkpoint: &CheckpointRecord,
    ) -> Result<(), MachineError> {
        let mut guard: async_rt::sync::MutexGuard<'_, HashMap<String, CheckpointRecord>> =
            self.checkpoints.lock().await;
        guard.insert(thread_id.to_string(), checkpoint.clone());
        Ok(())
    }

    async fn load(&self, thread_id: &str) -> Result<Option<CheckpointRecord>, MachineError> {
        let guard: async_rt::sync::MutexGuard<'_, HashMap<String, CheckpointRecord>> =
            self.checkpoints.lock().await;
        Ok(guard.get(thread_id).cloned())
    }
}
