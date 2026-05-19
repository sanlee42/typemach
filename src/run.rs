use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use crate::op::{Effect, EffectUpdate, ItemWrite, NoopRunOps, RunOps};

macro_rules! id_newtype {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_string())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

id_newtype!(RunId);
id_newtype!(SessionId);
id_newtype!(ThreadId);
id_newtype!(WorkerId);
id_newtype!(LeaseId);

/// Input for a single turn invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRequest<I> {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub command: RunCommand,
    pub input: I,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<Value>,
    pub runtime_limits: RuntimeLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunCommand {
    Start,
    Resume,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeLimits {
    pub max_steps: u32,
    pub allow_clarification: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_timeout: Option<Duration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_timeout: Option<Duration>,
}

impl RuntimeLimits {
    pub fn new(max_steps: u32) -> Self {
        Self {
            max_steps,
            allow_clarification: true,
            step_timeout: None,
            run_timeout: None,
        }
    }
}

/// Output from a completed or interrupted turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RunOutput<O, I> {
    Completed {
        trace: Vec<Value>,
        output: O,
        snapshot: Value,
    },
    Interrupted {
        interrupt: I,
        snapshot: Value,
    },
}

/// Runtime streaming configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamConfig {
    pub heartbeat_interval: Duration,
    pub channel_capacity: usize,
}

impl StreamConfig {
    pub fn channel_capacity(self) -> usize {
        self.channel_capacity.max(1)
    }

    pub fn heartbeat_interval(self) -> Duration {
        self.heartbeat_interval.max(Duration::from_millis(1))
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(15),
            channel_capacity: 32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepResult {
    Next,
    Interrupt,
    Complete,
}

#[derive(Debug)]
pub enum RunStreamEvent<Step, Signal, Output, Interrupt> {
    Started {
        run_id: RunId,
        session_id: SessionId,
        thread_id: ThreadId,
    },
    Heartbeat {
        run_id: RunId,
        session_id: SessionId,
        thread_id: ThreadId,
    },
    StepStarted {
        step: Step,
        step_count: u32,
    },
    StepFinished {
        step: Step,
        result: StepResult,
    },
    Signal {
        signal: Signal,
    },
    Interrupted {
        interrupt: Interrupt,
        snapshot: Value,
    },
    Completed {
        trace: Vec<Value>,
        output: Output,
        snapshot: Value,
    },
    Failed {
        error: crate::error::MachineError,
    },
    Cancelled,
}

pub struct RunEventReceiver<Step, Signal, Output, Interrupt> {
    pub(crate) receiver:
        async_rt::sync::mpsc::Receiver<RunStreamEvent<Step, Signal, Output, Interrupt>>,
}

impl<Step, Signal, Output, Interrupt> RunEventReceiver<Step, Signal, Output, Interrupt> {
    pub async fn next_event(&mut self) -> Option<RunStreamEvent<Step, Signal, Output, Interrupt>> {
        self.receiver.recv().await
    }

    pub async fn next_event_timeout(
        &mut self,
        deadline: Instant,
    ) -> Option<RunStreamEvent<Step, Signal, Output, Interrupt>> {
        async_rt::time::timeout_at(deadline.into(), self.receiver.recv())
            .await
            .unwrap_or_default()
    }
}

#[derive(Debug)]
pub(crate) struct RunCancel {
    cancelled: AtomicBool,
    notify: async_rt::sync::Notify,
}

impl RunCancel {
    pub(crate) fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            notify: async_rt::sync::Notify::new(),
        }
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    pub(crate) async fn cancelled(&self) {
        loop {
            let notified = self.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// Context passed to every node during graph execution.
#[derive(Clone)]
pub struct RunContext<I, Step, Signal, Output, Interrupt> {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub runtime_limits: RuntimeLimits,
    pub input: I,
    pub(crate) event_tx:
        Option<async_rt::sync::mpsc::Sender<RunStreamEvent<Step, Signal, Output, Interrupt>>>,
    pub(crate) cancel: Arc<RunCancel>,
    pub(crate) ops: Arc<dyn RunOps>,
    _stream_types: PhantomData<(Step, Output, Interrupt)>,
}

impl<I, Step, Signal, Output, Interrupt> RunContext<I, Step, Signal, Output, Interrupt> {
    pub(crate) fn new(
        run_id: RunId,
        session_id: SessionId,
        thread_id: ThreadId,
        runtime_limits: RuntimeLimits,
        input: I,
        event_tx: Option<
            async_rt::sync::mpsc::Sender<RunStreamEvent<Step, Signal, Output, Interrupt>>,
        >,
        cancel: Arc<RunCancel>,
    ) -> Self {
        Self {
            run_id,
            session_id,
            thread_id,
            runtime_limits,
            input,
            event_tx,
            cancel,
            ops: Arc::new(NoopRunOps),
            _stream_types: PhantomData,
        }
    }

    pub(crate) fn with_ops(mut self, ops: Arc<dyn RunOps>) -> Self {
        self.ops = ops;
        self
    }

    pub async fn reserve(
        &self,
        key: &str,
        kind: &str,
        request: Value,
    ) -> Result<Effect, crate::error::MachineError> {
        self.ops.reserve(&self.run_id, key, kind, request).await
    }

    pub async fn start(&self, key: &str) -> Result<Effect, crate::error::MachineError> {
        self.ops.start(&self.run_id, key).await
    }

    pub async fn done(
        &self,
        key: impl Into<String>,
        result: Value,
    ) -> Result<(), crate::error::MachineError> {
        self.ops
            .push_effect(&self.run_id, EffectUpdate::done(key, result))
            .await
    }

    pub async fn fail(
        &self,
        key: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), crate::error::MachineError> {
        self.ops
            .push_effect(&self.run_id, EffectUpdate::failed(key, code, message))
            .await
    }

    pub async fn unknown(
        &self,
        key: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), crate::error::MachineError> {
        self.ops
            .push_effect(&self.run_id, EffectUpdate::unknown(key, message))
            .await
    }

    pub async fn item(
        &self,
        key: impl Into<String>,
        kind: impl Into<String>,
        body: Value,
    ) -> Result<(), crate::error::MachineError> {
        self.ops
            .push_item(
                &self.run_id,
                ItemWrite {
                    key: key.into(),
                    kind: kind.into(),
                    body,
                },
            )
            .await
    }

    pub fn is_streaming(&self) -> bool {
        self.event_tx.is_some()
    }

    pub async fn emit(&self, signal: Signal) -> Result<(), crate::error::MachineError> {
        let Some(tx) = &self.event_tx else {
            return Ok(());
        };
        tx.send(RunStreamEvent::Signal { signal })
            .await
            .map_err(|_| crate::error::MachineError::StreamClosed)
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    pub fn cancelled(&self) -> impl Future<Output = ()> + '_ {
        self.cancel.cancelled()
    }
}
