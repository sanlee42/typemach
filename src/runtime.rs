use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::checkpoint::CheckpointSaver;
use crate::error::MachineError;
use crate::lifecycle::{
    AppendEventResult, RunLifecycle, RunSubscription, StartRunRejection, StartRunResult,
};
use crate::machine::Machine;
use crate::registry::{RunHandle, RunRegistry};
use crate::run::{
    RunEventReceiver, RunId, RunOutput, RunRequest, RunStreamEvent, SessionId, StepResult,
    StreamConfig, ThreadId,
};
use crate::runner::Runner;
use crate::store::{
    FinishRunResult, RunEventEnvelope, RunEventPayload, RunFinish, RunLookup, RunStart, RunStatus,
    RunStore,
};

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

#[derive(Debug, Clone)]
pub struct Start<Scope = Value> {
    pub scope: Scope,
    pub kind: String,
    pub model: Option<String>,
    pub key: Option<String>,
    pub parent: Option<RunId>,
    pub retry_of: Option<RunId>,
    pub meta: Value,
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
    ) -> Result<StartResult<MachineRx<M>>, MachineError> {
        self.life
            .ensure_session(Some(req.session_id.clone()), &start.scope)
            .await?;

        let scope = start.scope.clone();
        let run = RunStart {
            run_id: req.run_id.clone(),
            session_id: req.session_id.clone(),
            agent_kind: start.kind,
            model: start.model,
            client_run_key: start.key,
            parent_run_id: start.parent,
            retry_of_run_id: start.retry_of,
            scope: start.scope,
            metadata: start.meta,
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
    ) -> Result<StartResult<MachineOutput<M>>, MachineError> {
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
mod tests {
    use super::*;
    use crate::checkpoint::MemorySaver;
    use crate::machine::{ResumeAction, Transition};
    use crate::run::{RunCommand, RuntimeLimits};
    use crate::store::{MemoryRunStore, RunStore};
    use async_trait::async_trait;
    use serde_json::json;
    use std::time::{Duration, Instant};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    enum Step {
        Start,
        Done,
        Loop,
        Slow,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct State {
        value: u32,
    }

    #[derive(Debug, Clone)]
    struct Input {
        mode: Mode,
    }

    #[derive(Debug, Clone)]
    enum Mode {
        Complete,
        Interrupt,
        Loop,
        Slow,
    }

    struct TestMachine;

    #[async_trait]
    impl Machine for TestMachine {
        type Step = Step;
        type State = State;
        type Input = Input;
        type Signal = String;
        type Output = String;
        type Interrupt = String;

        fn start_step(&self) -> Self::Step {
            Step::Start
        }

        fn resume_action(&self, _interrupt: &Self::Interrupt) -> ResumeAction<Self::Step> {
            ResumeAction::ReenterInterruptedStep
        }

        fn new_state(
            &self,
            _input: &Self::Input,
            _previous: Option<&Self::State>,
            _snapshot: Option<&Value>,
        ) -> Result<Self::State, MachineError> {
            Ok(State { value: 0 })
        }

        fn apply_resume_input(
            &self,
            _state: &mut Self::State,
            _input: &Self::Input,
        ) -> Result<(), MachineError> {
            Ok(())
        }

        async fn transition(
            &self,
            step: Self::Step,
            state: &mut Self::State,
            ctx: &crate::run::RunContext<
                Self::Input,
                Self::Step,
                Self::Signal,
                Self::Output,
                Self::Interrupt,
            >,
        ) -> Result<Transition<Self::Step, Self::Interrupt, Self::Output>, MachineError> {
            match (&ctx.input.mode, step) {
                (Mode::Complete, Step::Start) => {
                    ctx.emit("signal-1".to_string()).await?;
                    state.value += 1;
                    Ok(Transition::Next(Step::Done))
                }
                (Mode::Complete, Step::Done) => {
                    Ok(Transition::Complete(format!("value={}", state.value)))
                }
                (Mode::Interrupt, Step::Start) => Ok(Transition::Interrupt("answer?".to_string())),
                (Mode::Loop, Step::Start) | (Mode::Loop, Step::Loop) => {
                    Ok(Transition::Next(Step::Loop))
                }
                (Mode::Slow, Step::Start) => Ok(Transition::Next(Step::Slow)),
                (Mode::Slow, Step::Slow) => {
                    async_rt::time::sleep(Duration::from_secs(5)).await;
                    Ok(Transition::Complete("slow".to_string()))
                }
                _ => Ok(Transition::Complete("done".to_string())),
            }
        }
    }

    fn runtime(max: usize) -> Runtime<TestMachine, MemorySaver, MemoryRunStore<Event>> {
        Runtime::new(
            TestMachine,
            Arc::new(MemorySaver::new()),
            Arc::new(MemoryRunStore::<Event>::new()),
            max,
        )
    }

    fn request(run_id: &str, mode: Mode) -> RunRequest<Input> {
        RunRequest {
            run_id: RunId::from(run_id),
            session_id: SessionId::from("session-1"),
            thread_id: ThreadId::from(format!("thread-{run_id}")),
            command: RunCommand::Start,
            input: Input { mode },
            snapshot: None,
            runtime_limits: RuntimeLimits {
                max_steps: 5,
                allow_clarification: true,
            },
        }
    }

    fn start(key: Option<&str>) -> Start {
        let mut start = Start::new(scope(), "test");
        start.key = key.map(str::to_string);
        start
    }

    fn scope() -> Value {
        json!({"tenant": "demo"})
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

    #[test]
    fn stream_persists_and_forwards_completed_events() {
        block_on(async {
            let rt = runtime(8);
            let run_id = RunId::from("run-stream");
            let result = rt
                .stream(
                    request(run_id.as_str(), Mode::Complete),
                    start(None),
                    StreamConfig::default(),
                )
                .await
                .expect("stream");
            let StartResult::Started(mut rx) = result else {
                panic!("expected started");
            };

            let mut completed = None;
            while let Some(event) = rx.next_event().await {
                if let RunStreamEvent::Completed { output, .. } = event {
                    completed = Some(output);
                    break;
                }
            }

            assert_eq!(completed.as_deref(), Some("value=1"));
            let events = rt
                .store()
                .list_events(&run_id, &scope(), 0)
                .await
                .expect("events");
            assert!(matches!(
                events.first().map(|event| &event.payload),
                Some(Payload::Start { .. })
            ));
            assert!(matches!(
                events.last().map(|event| &event.payload),
                Some(Payload::Done { .. })
            ));
            let lookup = rt
                .store()
                .lookup_run(&run_id, &scope())
                .await
                .expect("lookup")
                .expect("run");
            assert_eq!(lookup.status, RunStatus::Completed);
        });
    }

    #[test]
    fn invoke_returns_output_and_records_events() {
        block_on(async {
            let rt = runtime(8);
            let run_id = RunId::from("run-invoke");
            let result = rt
                .invoke(request(run_id.as_str(), Mode::Complete), start(None))
                .await
                .expect("invoke");
            let StartResult::Started(output) = result else {
                panic!("expected started");
            };
            match output {
                RunOutput::Completed { output, .. } => assert_eq!(output, "value=1"),
                RunOutput::Interrupted { .. } => panic!("expected completed"),
            }
            let events = rt
                .store()
                .list_events(&run_id, &scope(), 0)
                .await
                .expect("events");
            assert!(
                events
                    .iter()
                    .any(|event| matches!(event.payload, Payload::Signal { .. }))
            );
        });
    }

    #[test]
    fn interrupted_run_gets_interrupted_status() {
        block_on(async {
            let rt = runtime(8);
            let run_id = RunId::from("run-interrupt");
            let result = rt
                .invoke(request(run_id.as_str(), Mode::Interrupt), start(None))
                .await
                .expect("invoke");
            let StartResult::Started(output) = result else {
                panic!("expected started");
            };
            match output {
                RunOutput::Interrupted { interrupt, .. } => assert_eq!(interrupt, "answer?"),
                RunOutput::Completed { .. } => panic!("expected interrupt"),
            }
            let lookup = rt
                .store()
                .lookup_run(&run_id, &scope())
                .await
                .expect("lookup")
                .expect("run");
            assert_eq!(lookup.status, RunStatus::Interrupted);
        });
    }

    #[test]
    fn failed_run_records_error_terminal() {
        block_on(async {
            let rt = runtime(8);
            let run_id = RunId::from("run-fail");
            let mut req = request(run_id.as_str(), Mode::Loop);
            req.runtime_limits.max_steps = 1;
            let err = rt
                .invoke(req, start(None))
                .await
                .expect_err("invoke should fail");
            assert!(matches!(err, MachineError::MaxStepsExceeded { max: 1 }));

            let lookup = rt
                .store()
                .lookup_run(&run_id, &scope())
                .await
                .expect("lookup")
                .expect("run");
            assert_eq!(lookup.status, RunStatus::Error);
            let terminal = rt
                .store()
                .terminal_event(&run_id, &scope())
                .await
                .expect("terminal")
                .expect("terminal event");
            assert!(matches!(terminal.payload, Payload::Fail { .. }));
        });
    }

    #[test]
    fn cancel_stops_active_run_and_records_cancelled() {
        block_on(async {
            let rt = runtime(8);
            let run_id = RunId::from("run-cancel");
            let result = rt
                .stream(
                    request(run_id.as_str(), Mode::Slow),
                    start(None),
                    StreamConfig::default(),
                )
                .await
                .expect("stream");
            let StartResult::Started(mut rx) = result else {
                panic!("expected started");
            };

            assert!(rt.cancel(&run_id, &scope()).await.expect("cancel"));
            let mut cancelled = false;
            let deadline = Instant::now() + Duration::from_secs(1);
            while let Some(event) = rx.next_event_timeout(deadline).await {
                if matches!(event, RunStreamEvent::Cancelled) {
                    cancelled = true;
                    break;
                }
            }
            assert!(cancelled);
            let lookup = rt
                .store()
                .lookup_run(&run_id, &scope())
                .await
                .expect("lookup")
                .expect("run");
            assert_eq!(lookup.status, RunStatus::Cancelled);
        });
    }

    #[test]
    fn dropped_receiver_cancels_and_finishes_run() {
        block_on(async {
            let rt = runtime(8);
            let run_id = RunId::from("run-drop");
            let result = rt
                .stream(
                    request(run_id.as_str(), Mode::Slow),
                    start(None),
                    StreamConfig::default(),
                )
                .await
                .expect("stream");
            let StartResult::Started(rx) = result else {
                panic!("expected started");
            };
            drop(rx);

            let deadline = Instant::now() + Duration::from_secs(1);
            loop {
                if let Some(lookup) = rt
                    .store()
                    .lookup_run(&run_id, &scope())
                    .await
                    .expect("lookup")
                    && lookup.status == RunStatus::Cancelled
                {
                    break;
                }
                assert!(Instant::now() < deadline);
                async_rt::time::sleep(Duration::from_millis(10)).await;
            }
        });
    }

    #[test]
    fn dropped_receiver_cancels_quiet_transition_without_waiting_for_heartbeat() {
        block_on(async {
            let rt = runtime(8);
            let run_id = RunId::from("run-drop-quiet");
            let result = rt
                .stream(
                    request(run_id.as_str(), Mode::Slow),
                    start(None),
                    StreamConfig {
                        heartbeat_interval: Duration::from_secs(30),
                        channel_capacity: 32,
                    },
                )
                .await
                .expect("stream");
            let StartResult::Started(mut rx) = result else {
                panic!("expected started");
            };

            let deadline = Instant::now() + Duration::from_secs(1);
            loop {
                let event = rx
                    .next_event_timeout(deadline)
                    .await
                    .expect("quiet step should start");
                if matches!(
                    event,
                    RunStreamEvent::StepStarted {
                        step: Step::Slow,
                        ..
                    }
                ) {
                    break;
                }
            }
            drop(rx);

            let deadline = Instant::now() + Duration::from_secs(1);
            loop {
                if let Some(lookup) = rt
                    .store()
                    .lookup_run(&run_id, &scope())
                    .await
                    .expect("lookup")
                    && lookup.status == RunStatus::Cancelled
                {
                    break;
                }
                assert!(Instant::now() < deadline);
                async_rt::time::sleep(Duration::from_millis(10)).await;
            }
        });
    }

    #[test]
    fn idempotent_key_returns_existing_run() {
        block_on(async {
            let rt = runtime(8);
            let first_id = RunId::from("run-idem-1");
            let result = rt
                .invoke(
                    request(first_id.as_str(), Mode::Complete),
                    start(Some("key-1")),
                )
                .await
                .expect("first invoke");
            assert!(matches!(result, StartResult::Started(_)));

            let result = rt
                .invoke(request("run-idem-2", Mode::Complete), start(Some("key-1")))
                .await
                .expect("second invoke");
            let StartResult::Existing(existing) = result else {
                panic!("expected existing");
            };
            assert_eq!(existing.run_id, first_id);
        });
    }

    #[test]
    fn capacity_rejection_does_not_persist_or_poison_key() {
        block_on(async {
            let rt = runtime(1);
            let blocker_id = RunId::from("run-capacity-blocker");
            let result = rt
                .stream(
                    request(blocker_id.as_str(), Mode::Slow),
                    start(None),
                    StreamConfig::default(),
                )
                .await
                .expect("stream");
            let StartResult::Started(mut blocker_rx) = result else {
                panic!("expected blocker to start");
            };

            let rejected_id = RunId::from("run-capacity-rejected");
            let result = rt
                .stream(
                    request(rejected_id.as_str(), Mode::Complete),
                    start(Some("capacity-key")),
                    StreamConfig::default(),
                )
                .await
                .expect("stream");
            assert!(matches!(
                result,
                StartResult::Rejected(StartRunRejection::CapacityExceeded)
            ));

            let lookup = rt
                .store()
                .lookup_run(&rejected_id, &scope())
                .await
                .expect("lookup");
            assert!(lookup.is_none());

            assert!(rt.cancel(&blocker_id, &scope()).await.expect("cancel"));
            let deadline = Instant::now() + Duration::from_secs(1);
            while let Some(event) = blocker_rx.next_event_timeout(deadline).await {
                if matches!(event, RunStreamEvent::Cancelled) {
                    break;
                }
            }

            let retry_id = RunId::from("run-capacity-retry");
            let result = rt
                .invoke(
                    request(retry_id.as_str(), Mode::Complete),
                    start(Some("capacity-key")),
                )
                .await
                .expect("retry");
            assert!(matches!(result, StartResult::Started(_)));
        });
    }

    #[test]
    fn subscribe_respects_scope() {
        block_on(async {
            let rt = runtime(8);
            let run_id = RunId::from("run-scope");
            let result = rt
                .invoke(request(run_id.as_str(), Mode::Complete), start(None))
                .await
                .expect("invoke");
            assert!(matches!(result, StartResult::Started(_)));

            let wrong_scope = json!({"tenant": "other"});
            let sub = rt
                .subscribe(&run_id, &wrong_scope, 0)
                .await
                .expect("subscribe");
            assert!(matches!(sub, RunSubscription::Missing));
        });
    }
}
