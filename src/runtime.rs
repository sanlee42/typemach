use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::checkpoint::CheckpointSaver;
use crate::error::MachineError;
use crate::lifecycle::{
    AppendEventResult, RunLifecycle, RunSubscription, StartRunRejection, StartRunResult,
};
use crate::machine::Machine;
use crate::op::EntryWrite;
use crate::registry::{RunHandle, RunRegistry};
use crate::run::{
    LeaseId, RunEventReceiver, RunId, RunOutput, RunRequest, RunStreamEvent, SessionId, StepResult,
    StreamConfig, ThreadId, WorkerId,
};
use crate::runner::Runner;
use crate::store::{
    FinishRunResult, RunEventEnvelope, RunEventPayload, RunFinish, RunLookup, RunStart, RunStatus,
    RunStore,
};

mod tx;
pub use tx::TxRuntime;

pub type Event = RunEventEnvelope<Payload>;

pub type Rx<Step, Signal, Output, Interrupt> = RunEventReceiver<Step, Signal, Output, Interrupt>;

type StreamEvent<M> = RunStreamEvent<
    <M as Machine>::Step,
    <M as Machine>::Signal,
    <M as Machine>::Output,
    <M as Machine>::Interrupt,
>;

type MachineRx<M> = Rx<
    <M as Machine>::Step,
    <M as Machine>::Signal,
    <M as Machine>::Output,
    <M as Machine>::Interrupt,
>;

type MachineOutput<M> = RunOutput<<M as Machine>::Output, <M as Machine>::Interrupt>;

static NEXT_LEASE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct LeaseCfg {
    pub owner: WorkerId,
    pub ttl: Duration,
    pub renew: Duration,
}

impl LeaseCfg {
    pub fn new(owner: WorkerId) -> Self {
        Self {
            owner,
            ttl: Duration::from_secs(30),
            renew: Duration::from_secs(10),
        }
    }
}

impl Default for LeaseCfg {
    fn default() -> Self {
        Self::new(new_worker_id())
    }
}

#[derive(Debug, Clone)]
pub struct Start<Scope = Value> {
    pub scope: Scope,
    pub kind: String,
    pub model: Option<String>,
    pub key: Option<String>,
    pub parent: Option<RunId>,
    pub retry_of: Option<RunId>,
    pub meta: Value,
    pub entries: Vec<EntryWrite>,
    pub token: Option<String>,
}

impl<Scope> Start<Scope> {
    pub fn new(scope: Scope, kind: impl Into<String>) -> Self {
        Self {
            scope,
            kind: kind.into(),
            model: None,
            key: None,
            parent: None,
            retry_of: None,
            meta: Value::Object(Default::default()),
            entries: Vec::new(),
            token: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StartResult<T> {
    Started(T),
    Existing(RunLookup),
    Rejected(StartRunRejection),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Payload {
    Start {
        thread_id: ThreadId,
    },
    Beat {
        thread_id: ThreadId,
    },
    StepStart {
        step: Value,
        n: u32,
    },
    StepDone {
        step: Value,
        result: StepResult,
    },
    Signal {
        signal: Value,
    },
    Done {
        trace: Vec<Value>,
        output: Value,
        snapshot: Value,
    },
    Interrupt {
        interrupt: Value,
        snapshot: Value,
    },
    Fail {
        error: String,
    },
    Cancel,
}

impl RunEventPayload for Payload {
    fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Done { .. } | Self::Interrupt { .. } | Self::Fail { .. } | Self::Cancel
        )
    }
}

pub struct Runtime<M, C, S>
where
    M: Machine,
    C: CheckpointSaver,
    S: RunStore<Event>,
{
    runner: Runner<M, C>,
    life: RunLifecycle<Event, S>,
}

impl<M, C, S> Clone for Runtime<M, C, S>
where
    M: Machine,
    C: CheckpointSaver,
    S: RunStore<Event>,
{
    fn clone(&self) -> Self {
        Self {
            runner: self.runner.clone(),
            life: self.life.clone(),
        }
    }
}

impl<M, C, S> Runtime<M, C, S>
where
    M: Machine,
    C: CheckpointSaver + 'static,
    S: RunStore<Event> + 'static,
    S::FinishData: Default,
{
    pub fn new(machine: M, ck: Arc<C>, store: Arc<S>, max: usize) -> Self {
        Self {
            runner: Runner::new(machine, ck),
            life: RunLifecycle::new(RunRegistry::new(), store, max),
        }
    }

    pub fn runner(&self) -> &Runner<M, C> {
        &self.runner
    }

    pub fn life(&self) -> &RunLifecycle<Event, S> {
        &self.life
    }

    pub fn store(&self) -> &S {
        self.life.store()
    }

    pub async fn stream(
        &self,
        req: RunRequest<M::Input>,
        start: Start<S::Scope>,
        cfg: StreamConfig,
    ) -> Result<StartResult<MachineRx<M>>, MachineError>
    where
        M::Input: Serialize,
    {
        self.life
            .ensure_session(Some(req.session_id.clone()), &start.scope)
            .await?;

        let scope = start.scope.clone();
        let input = Some(to_json(&req.input)?);
        let run = RunStart {
            run_id: req.run_id.clone(),
            session_id: req.session_id.clone(),
            thread_id: req.thread_id.clone(),
            agent_kind: start.kind,
            model: start.model,
            client_run_key: start.key,
            parent_run_id: start.parent,
            retry_of_run_id: start.retry_of,
            scope: start.scope,
            metadata: start.meta,
            input,
            entries: start.entries,
            lease: None,
        };
        let token = start.token.unwrap_or_else(|| req.run_id.to_string());
        match self
            .life
            .start_run(run, RunHandle::new(token), None)
            .await?
        {
            StartRunResult::Started => {}
            StartRunResult::Existing(existing) => return Ok(StartResult::Existing(existing)),
            StartRunResult::NotRegistered(rejection) => {
                return Ok(StartResult::Rejected(rejection));
            }
        }

        let run_id = req.run_id.clone();
        let session_id = req.session_id.clone();
        let raw = self.runner.stream(req, cfg);
        let (tx, receiver) = async_rt::sync::mpsc::channel(cfg.channel_capacity());
        let runner = self.runner.clone();
        let life = self.life.clone();
        async_rt::spawn(async move {
            bridge::<M, C, S>(runner, life, raw, tx, run_id, session_id, scope).await;
        });

        Ok(StartResult::Started(RunEventReceiver { receiver }))
    }

    pub async fn invoke(
        &self,
        req: RunRequest<M::Input>,
        start: Start<S::Scope>,
    ) -> Result<StartResult<MachineOutput<M>>, MachineError>
    where
        M::Input: Serialize,
    {
        let stream = self.stream(req, start, StreamConfig::default()).await?;
        let StartResult::Started(mut rx) = stream else {
            return Ok(stream.map_started(|_| unreachable!()));
        };

        loop {
            match rx.next_event().await {
                Some(RunStreamEvent::Completed {
                    trace,
                    output,
                    snapshot,
                }) => {
                    return Ok(StartResult::Started(RunOutput::Completed {
                        trace,
                        output,
                        snapshot,
                    }));
                }
                Some(RunStreamEvent::Interrupted {
                    interrupt,
                    snapshot,
                }) => {
                    return Ok(StartResult::Started(RunOutput::Interrupted {
                        interrupt,
                        snapshot,
                    }));
                }
                Some(RunStreamEvent::Failed { error }) => return Err(error),
                Some(RunStreamEvent::Cancelled) => return Err(MachineError::Cancelled),
                Some(_) => {}
                None => return Err(MachineError::StreamClosed),
            }
        }
    }

    pub async fn cancel(&self, run_id: &RunId, scope: &S::Scope) -> Result<bool, MachineError> {
        let active = self.life.request_cancel(run_id, scope).await?.is_some();
        self.runner.cancel_run(run_id).await;
        Ok(active)
    }

    pub async fn subscribe(
        &self,
        run_id: &RunId,
        scope: &S::Scope,
        after_seq: i64,
    ) -> Result<RunSubscription<Event>, MachineError> {
        self.life.subscribe(run_id, scope, after_seq).await
    }
}

impl<T> StartResult<T> {
    fn map_started<U>(self, f: impl FnOnce(T) -> U) -> StartResult<U> {
        match self {
            Self::Started(value) => StartResult::Started(f(value)),
            Self::Existing(existing) => StartResult::Existing(existing),
            Self::Rejected(rejection) => StartResult::Rejected(rejection),
        }
    }
}

fn new_worker_id() -> WorkerId {
    WorkerId::from(unique_id("worker"))
}

fn new_lease_id() -> LeaseId {
    LeaseId::from(unique_id("lease"))
}

fn unique_id(prefix: &str) -> String {
    let n = NEXT_LEASE_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("{prefix}-{}-{nanos}-{n}", std::process::id())
}

async fn bridge<M, C, S>(
    runner: Runner<M, C>,
    life: RunLifecycle<Event, S>,
    mut raw: MachineRx<M>,
    tx: async_rt::sync::mpsc::Sender<StreamEvent<M>>,
    run_id: RunId,
    session_id: SessionId,
    scope: S::Scope,
) where
    M: Machine,
    C: CheckpointSaver + 'static,
    S: RunStore<Event> + 'static,
    S::FinishData: Default,
{
    loop {
        match raw.receiver.try_recv() {
            Ok(event) => {
                if forward_event::<M, C, S>(
                    &runner,
                    &life,
                    &tx,
                    &run_id,
                    &session_id,
                    &scope,
                    event,
                )
                .await
                {
                    return;
                }
                continue;
            }
            Err(async_rt::sync::mpsc::error::TryRecvError::Empty) => {}
            Err(async_rt::sync::mpsc::error::TryRecvError::Disconnected) => {
                cancel_and_finish::<M, C, S>(
                    &runner,
                    &life,
                    &run_id,
                    &session_id,
                    &scope,
                    "stream_closed",
                )
                .await;
                return;
            }
        }

        async_rt::select! {
            event = raw.next_event() => {
                let Some(event) = event else {
                    cancel_and_finish::<M, C, S>(
                        &runner,
                        &life,
                        &run_id,
                        &session_id,
                        &scope,
                        "stream_closed",
                    )
                    .await;
                    return;
                };
                if forward_event::<M, C, S>(
                    &runner,
                    &life,
                    &tx,
                    &run_id,
                    &session_id,
                    &scope,
                    event,
                )
                .await
                {
                    return;
                }
            }
            _ = tx.closed() => {
                cancel_and_finish::<M, C, S>(
                    &runner,
                    &life,
                    &run_id,
                    &session_id,
                    &scope,
                    "receiver_closed",
                )
                .await;
                return;
            }
        }
    }
}

async fn forward_event<M, C, S>(
    runner: &Runner<M, C>,
    life: &RunLifecycle<Event, S>,
    tx: &async_rt::sync::mpsc::Sender<StreamEvent<M>>,
    run_id: &RunId,
    session_id: &SessionId,
    scope: &S::Scope,
    event: StreamEvent<M>,
) -> bool
where
    M: Machine,
    C: CheckpointSaver + 'static,
    S: RunStore<Event> + 'static,
    S::FinishData: Default,
{
    let record = record_event::<M, S>(life, run_id, session_id, scope, &event).await;
    let terminal = matches!(record, Ok(true));

    if let Err(error) = record {
        runner.cancel_run(run_id).await;
        finish_error::<S>(life, run_id, session_id, scope, &error).await;
        let _ = tx.send(RunStreamEvent::Failed { error }).await;
        return true;
    }

    if tx.send(event).await.is_err() {
        if !terminal {
            cancel_and_finish::<M, C, S>(
                runner,
                life,
                run_id,
                session_id,
                scope,
                "receiver_closed",
            )
            .await;
        }
        return true;
    }

    terminal
}

enum EventRecord {
    Append(Payload),
    Finish {
        payload: Payload,
        status: RunStatus,
        reason: &'static str,
        code: Option<String>,
    },
}

async fn record_event<M, S>(
    life: &RunLifecycle<Event, S>,
    run_id: &RunId,
    session_id: &SessionId,
    scope: &S::Scope,
    event: &StreamEvent<M>,
) -> Result<bool, MachineError>
where
    M: Machine,
    S: RunStore<Event>,
    S::FinishData: Default,
{
    match event_record::<M>(event)? {
        EventRecord::Append(payload) => {
            match life
                .append_event(run_id, session_id, scope, payload)
                .await?
            {
                AppendEventResult::Recorded(_) => Ok(false),
                AppendEventResult::Skipped => Err(MachineError::RunNotFound),
            }
        }
        EventRecord::Finish {
            payload,
            status,
            reason,
            code,
        } => {
            let finish = RunFinish {
                run_id: run_id.clone(),
                session_id: session_id.clone(),
                scope: scope.clone(),
                status,
                finish_reason: reason.to_string(),
                error_code: code,
                entries: Vec::new(),
                data: S::FinishData::default(),
            };
            let result = life.finish_run(finish, payload).await?;
            match result {
                FinishRunResult::Finished(_) | FinishRunResult::AlreadyFinished(_) => Ok(true),
            }
        }
    }
}

fn event_record<M>(event: &StreamEvent<M>) -> Result<EventRecord, MachineError>
where
    M: Machine,
{
    Ok(match event {
        RunStreamEvent::Started { thread_id, .. } => EventRecord::Append(Payload::Start {
            thread_id: thread_id.clone(),
        }),
        RunStreamEvent::Heartbeat { thread_id, .. } => EventRecord::Append(Payload::Beat {
            thread_id: thread_id.clone(),
        }),
        RunStreamEvent::StepStarted { step, step_count } => {
            EventRecord::Append(Payload::StepStart {
                step: to_json(step)?,
                n: *step_count,
            })
        }
        RunStreamEvent::StepFinished { step, result } => EventRecord::Append(Payload::StepDone {
            step: to_json(step)?,
            result: *result,
        }),
        RunStreamEvent::Signal { signal } => EventRecord::Append(Payload::Signal {
            signal: to_json(signal)?,
        }),
        RunStreamEvent::Completed {
            trace,
            output,
            snapshot,
        } => EventRecord::Finish {
            payload: Payload::Done {
                trace: trace.clone(),
                output: to_json(output)?,
                snapshot: snapshot.clone(),
            },
            status: RunStatus::Completed,
            reason: "completed",
            code: None,
        },
        RunStreamEvent::Interrupted {
            interrupt,
            snapshot,
        } => EventRecord::Finish {
            payload: Payload::Interrupt {
                interrupt: to_json(interrupt)?,
                snapshot: snapshot.clone(),
            },
            status: RunStatus::Interrupted,
            reason: "interrupted",
            code: None,
        },
        RunStreamEvent::Failed { error } => EventRecord::Finish {
            payload: Payload::Fail {
                error: error.to_string(),
            },
            status: RunStatus::Error,
            reason: "failed",
            code: Some(error_code(error).to_string()),
        },
        RunStreamEvent::Cancelled => EventRecord::Finish {
            payload: Payload::Cancel,
            status: RunStatus::Cancelled,
            reason: "cancelled",
            code: None,
        },
    })
}

fn to_json<T>(value: &T) -> Result<Value, MachineError>
where
    T: Serialize,
{
    serde_json::to_value(value).map_err(MachineError::Serialization)
}

async fn cancel_and_finish<M, C, S>(
    runner: &Runner<M, C>,
    life: &RunLifecycle<Event, S>,
    run_id: &RunId,
    session_id: &SessionId,
    scope: &S::Scope,
    reason: &str,
) where
    M: Machine,
    C: CheckpointSaver + 'static,
    S: RunStore<Event>,
    S::FinishData: Default,
{
    let _ = life.request_cancel(run_id, scope).await;
    runner.cancel_run(run_id).await;
    let finish = RunFinish {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        scope: scope.clone(),
        status: RunStatus::Cancelled,
        finish_reason: reason.to_string(),
        error_code: None,
        entries: Vec::new(),
        data: S::FinishData::default(),
    };
    let _ = life.finish_run(finish, Payload::Cancel).await;
}

async fn finish_error<S>(
    life: &RunLifecycle<Event, S>,
    run_id: &RunId,
    session_id: &SessionId,
    scope: &S::Scope,
    error: &MachineError,
) where
    S: RunStore<Event>,
    S::FinishData: Default,
{
    let finish = RunFinish {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        scope: scope.clone(),
        status: RunStatus::Error,
        finish_reason: "failed".to_string(),
        error_code: Some(error_code(error).to_string()),
        entries: Vec::new(),
        data: S::FinishData::default(),
    };
    let _ = life
        .finish_run(
            finish,
            Payload::Fail {
                error: error.to_string(),
            },
        )
        .await;
}

fn error_code(error: &MachineError) -> &'static str {
    match error {
        MachineError::CheckpointDb(_) => "checkpoint_db",
        MachineError::CheckpointPool(_) => "checkpoint_pool",
        MachineError::StoreDb(_) => "store_db",
        MachineError::Serialization(_) => "serialization",
        MachineError::Deserialization(_) => "deserialization",
        MachineError::MaxStepsExceeded { .. } => "max_steps_exceeded",
        MachineError::CapacityExceeded => "capacity_exceeded",
        MachineError::RunAlreadyActive => "run_already_active",
        MachineError::LeaseLost => "lease_lost",
        MachineError::NotOwner { .. } => "not_owner",
        MachineError::ThreadBusy { .. } => "thread_busy",
        MachineError::RuntimeOpUnavailable => "runtime_op_unavailable",
        MachineError::EffectConflict => "effect_conflict",
        MachineError::EffectPending => "effect_pending",
        MachineError::EffectNotFound => "effect_not_found",
        MachineError::ItemConflict => "item_conflict",
        MachineError::EntryConflict => "entry_conflict",
        MachineError::StartConflict => "start_conflict",
        MachineError::InputConflict => "input_conflict",
        MachineError::InvalidPageLimit => "invalid_page_limit",
        MachineError::StepTimeout => "step_timeout",
        MachineError::RunTimeout => "run_timeout",
        MachineError::RunNotFound => "run_not_found",
        MachineError::InvalidRunEvent { .. } => "invalid_run_event",
        MachineError::Cancelled => "cancelled",
        MachineError::StreamClosed => "stream_closed",
        MachineError::NoCheckpointState => "no_checkpoint_state",
        MachineError::NoPendingInterrupt => "no_pending_interrupt",
        MachineError::NoInterruptedStep => "no_interrupted_step",
        MachineError::InvalidInterrupt { .. } => "invalid_interrupt",
        MachineError::InvalidStep { .. } => "invalid_step",
        MachineError::Transition(_) => "transition",
    }
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod runtime_tests;
