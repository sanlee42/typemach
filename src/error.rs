use thiserror::Error;

#[derive(Debug, Error)]
pub enum MachineError {
    #[error("checkpoint database error: {0}")]
    CheckpointDb(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("checkpoint pool exhausted: {0}")]
    CheckpointPool(String),

    #[error("store database error: {0}")]
    StoreDb(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("serialization failed: {0}")]
    Serialization(#[source] serde_json::Error),

    #[error("deserialization failed: {0}")]
    Deserialization(#[source] serde_json::Error),

    #[error("max steps exceeded ({max})")]
    MaxStepsExceeded { max: u32 },

    #[error("too many in-flight runs")]
    CapacityExceeded,

    #[error("run cancelled")]
    Cancelled,

    #[error("stream receiver closed")]
    StreamClosed,

    #[error("checkpoint has pending interrupt but no state")]
    NoCheckpointState,

    #[error("resume requested but checkpoint has no pending interrupt")]
    NoPendingInterrupt,

    #[error("checkpoint has pending interrupt but no interrupted step")]
    NoInterruptedStep,

    #[error("checkpoint has invalid interrupt payload: {reason}")]
    InvalidInterrupt { reason: String },

    #[error("checkpoint has invalid step payload: {reason}")]
    InvalidStep { reason: String },

    #[error("machine transition failed: {0}")]
    Transition(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl MachineError {
    pub fn transition<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Transition(Box::new(source))
    }
}
