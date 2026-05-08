use crate::checkpoint::{CheckpointRecord, CheckpointSaver};
use crate::error::MachineError;
use crate::machine::{Machine, MachineState, ResumeAction, Transition};
use crate::run::{
    RunCancel, RunCommand, RunContext, RunEventReceiver, RunId, RunOutput, RunRequest,
    RunStreamEvent, StepResult, StreamConfig, ThreadId,
};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::info;

type StreamSender<M> = async_rt::sync::mpsc::Sender<
    RunStreamEvent<
        <M as Machine>::Step,
        <M as Machine>::Signal,
        <M as Machine>::Output,
        <M as Machine>::Interrupt,
    >,
>;

type MachineRunContext<M> = RunContext<
    <M as Machine>::Input,
    <M as Machine>::Step,
    <M as Machine>::Signal,
    <M as Machine>::Output,
    <M as Machine>::Interrupt,
>;

pub struct Runner<M, CK>
where
    M: Machine,
    CK: CheckpointSaver,
{
    machine: Arc<M>,
    checkpointer: Arc<CK>,
    thread_locks: Arc<async_rt::sync::Mutex<HashMap<ThreadId, Arc<async_rt::sync::Mutex<()>>>>>,
    cancelled_runs: Arc<async_rt::sync::Mutex<HashSet<RunId>>>,
    active_runs: Arc<async_rt::sync::Mutex<HashMap<RunId, Arc<RunCancel>>>>,
}

impl<M, CK> Clone for Runner<M, CK>
where
    M: Machine,
    CK: CheckpointSaver,
{
    fn clone(&self) -> Self {
        Self {
            machine: Arc::clone(&self.machine),
            checkpointer: Arc::clone(&self.checkpointer),
            thread_locks: Arc::clone(&self.thread_locks),
            cancelled_runs: Arc::clone(&self.cancelled_runs),
            active_runs: Arc::clone(&self.active_runs),
        }
    }
}

impl<M, CK> Runner<M, CK>
where
    M: Machine,
    CK: CheckpointSaver + 'static,
{
    pub fn new(machine: M, checkpointer: Arc<CK>) -> Self {
        Self {
            machine: Arc::new(machine),
            checkpointer,
            thread_locks: Arc::new(async_rt::sync::Mutex::new(HashMap::new())),
            cancelled_runs: Arc::new(async_rt::sync::Mutex::new(HashSet::new())),
            active_runs: Arc::new(async_rt::sync::Mutex::new(HashMap::new())),
        }
    }

    pub fn machine(&self) -> &M {
        &self.machine
    }

    pub fn checkpointer(&self) -> &CK {
        &self.checkpointer
    }

    pub async fn cancel_run(&self, run_id: &RunId) {
        self.cancelled_runs.lock().await.insert(run_id.clone());
        if let Some(active) = self.active_runs.lock().await.get(run_id).cloned() {
            active.cancel();
        }
    }

    pub async fn is_cancelled(&self, run_id: &RunId) -> bool {
        self.cancelled_runs.lock().await.contains(run_id)
    }

    pub async fn invoke(
        &self,
        request: RunRequest<M::Input>,
    ) -> Result<RunOutput<M::Output, M::Interrupt>, MachineError> {
        let run_id = request.run_id.clone();
        let control = Arc::new(RunCancel::new());
        self.register_active(&run_id, Arc::clone(&control)).await;
        let result = self.invoke_inner(request, control).await;
        self.unregister_active(&run_id, None).await;
        result
    }

    pub fn stream(
        &self,
        request: RunRequest<M::Input>,
        config: StreamConfig,
    ) -> RunEventReceiver<M::Step, M::Signal, M::Output, M::Interrupt> {
        let (tx, receiver) = async_rt::sync::mpsc::channel(config.channel_capacity());
        let runner = self.clone();
        async_rt::spawn(async move {
            let run_id = request.run_id.clone();
            let control = Arc::new(RunCancel::new());
            runner.register_active(&run_id, Arc::clone(&control)).await;
            runner.run_stream_driver(request, config, tx, control).await;
            runner.unregister_active(&run_id, None).await;
        });
        RunEventReceiver { receiver }
    }

    async fn invoke_inner(
        &self,
        request: RunRequest<M::Input>,
        control: Arc<RunCancel>,
    ) -> Result<RunOutput<M::Output, M::Interrupt>, MachineError> {
        let thread_id = request.thread_id.clone();
        let _lock = self.thread_lock(&thread_id).await;

        if self.is_cancelled(&request.run_id).await {
            control.cancel();
            return Err(MachineError::Cancelled);
        }

        let checkpoint = self.checkpointer.load(thread_id.as_str()).await?;
        let (mut state, mut current_step) = self.restore_or_start(&request, checkpoint)?;
        let ctx = RunContext::new(
            request.run_id.clone(),
            request.session_id.clone(),
            thread_id.clone(),
            request.runtime_limits.clone(),
            request.input.clone(),
            None,
            control,
        );
        let mut trace = Vec::new();
        let mut step_count = 0u32;

        loop {
            step_count += 1;
            if step_count > request.runtime_limits.max_steps {
                return Err(MachineError::MaxStepsExceeded {
                    max: request.runtime_limits.max_steps,
                });
            }

            info!(
                run_id = %request.run_id,
                step = ?current_step,
                step_count,
                "executing machine step"
            );

            match self
                .transition_or_cancel(current_step.clone(), &mut state, &ctx)
                .await?
            {
                Transition::Next(next) => {
                    self.save_running(&request.run_id, &thread_id, &state, Some(&next))
                        .await?;
                    trace.push(step_trace(&current_step, "next"));
                    current_step = next;
                }
                Transition::Interrupt(interrupt) => {
                    self.save_interrupted(
                        &request.run_id,
                        &thread_id,
                        &state,
                        &current_step,
                        &interrupt,
                    )
                    .await?;
                    trace.push(step_trace(&current_step, "interrupt"));
                    return Ok(RunOutput::Interrupted {
                        interrupt,
                        snapshot: state.to_json()?,
                    });
                }
                Transition::Complete(output) => {
                    self.save_running(&request.run_id, &thread_id, &state, None)
                        .await?;
                    trace.push(step_trace(&current_step, "complete"));
                    return Ok(RunOutput::Completed {
                        trace,
                        output,
                        snapshot: state.to_json()?,
                    });
                }
            }
        }
    }

    async fn run_stream_driver(
        &self,
        request: RunRequest<M::Input>,
        config: StreamConfig,
        tx: StreamSender<M>,
        control: Arc<RunCancel>,
    ) {
        if send_event(
            &tx,
            RunStreamEvent::Started {
                run_id: request.run_id.clone(),
                session_id: request.session_id.clone(),
                thread_id: request.thread_id.clone(),
            },
        )
        .await
        .is_err()
        {
            return;
        }

        let result = self
            .run_stream_inner(&request, config, &tx, Arc::clone(&control))
            .await;
        match result {
            Ok(StreamTerminal::Completed {
                trace,
                output,
                snapshot,
            }) => {
                let _ = send_event(
                    &tx,
                    RunStreamEvent::Completed {
                        trace,
                        output,
                        snapshot,
                    },
                )
                .await;
            }
            Ok(StreamTerminal::Interrupted {
                interrupt,
                snapshot,
            }) => {
                let _ = send_event(
                    &tx,
                    RunStreamEvent::Interrupted {
                        interrupt,
                        snapshot,
                    },
                )
                .await;
            }
            Ok(StreamTerminal::Cancelled) => {
                let _ = send_event(&tx, RunStreamEvent::Cancelled).await;
            }
            Ok(StreamTerminal::ReceiverClosed) => {}
            Err(MachineError::Cancelled) => {
                let _ = send_event(&tx, RunStreamEvent::Cancelled).await;
            }
            Err(error) => {
                let _ = send_event(&tx, RunStreamEvent::Failed { error }).await;
            }
        }
    }

    async fn run_stream_inner(
        &self,
        request: &RunRequest<M::Input>,
        config: StreamConfig,
        tx: &StreamSender<M>,
        control: Arc<RunCancel>,
    ) -> Result<StreamTerminal<M::Output, M::Interrupt>, MachineError> {
        let thread_id = request.thread_id.clone();
        let _lock = match self
            .stream_thread_lock(request, tx, config.heartbeat_interval(), &control)
            .await
        {
            StreamLock::Acquired(guard) => guard,
            StreamLock::Cancelled => return Ok(StreamTerminal::Cancelled),
            StreamLock::ReceiverClosed => return Ok(StreamTerminal::ReceiverClosed),
        };

        if self.is_cancelled(&request.run_id).await {
            control.cancel();
            return Ok(StreamTerminal::Cancelled);
        }

        let checkpoint = self.checkpointer.load(thread_id.as_str()).await?;
        let (mut state, mut current_step) = self.restore_or_start(request, checkpoint)?;
        let ctx = RunContext::new(
            request.run_id.clone(),
            request.session_id.clone(),
            thread_id.clone(),
            request.runtime_limits.clone(),
            request.input.clone(),
            Some(tx.clone()),
            control,
        );
        let mut trace = Vec::new();
        let mut step_count = 0u32;

        loop {
            step_count += 1;
            if step_count > request.runtime_limits.max_steps {
                return Err(MachineError::MaxStepsExceeded {
                    max: request.runtime_limits.max_steps,
                });
            }

            info!(
                run_id = %request.run_id,
                step = ?current_step,
                step_count,
                "streaming machine step"
            );

            if send_event(
                tx,
                RunStreamEvent::StepStarted {
                    step: current_step.clone(),
                    step_count,
                },
            )
            .await
            .is_err()
            {
                return Ok(StreamTerminal::ReceiverClosed);
            }

            let transition = self
                .stream_transition(
                    current_step.clone(),
                    &mut state,
                    &ctx,
                    tx,
                    config.heartbeat_interval(),
                )
                .await?;

            match transition {
                StreamTransition::Next(next) => {
                    self.save_running(&request.run_id, &thread_id, &state, Some(&next))
                        .await?;
                    trace.push(step_trace(&current_step, "next"));
                    if send_event(
                        tx,
                        RunStreamEvent::StepFinished {
                            step: current_step,
                            result: StepResult::Next,
                        },
                    )
                    .await
                    .is_err()
                    {
                        return Ok(StreamTerminal::ReceiverClosed);
                    }
                    current_step = next;
                }
                StreamTransition::Interrupt(interrupt) => {
                    self.save_interrupted(
                        &request.run_id,
                        &thread_id,
                        &state,
                        &current_step,
                        &interrupt,
                    )
                    .await?;
                    trace.push(step_trace(&current_step, "interrupt"));
                    if send_event(
                        tx,
                        RunStreamEvent::StepFinished {
                            step: current_step,
                            result: StepResult::Interrupt,
                        },
                    )
                    .await
                    .is_err()
                    {
                        return Ok(StreamTerminal::ReceiverClosed);
                    }
                    return Ok(StreamTerminal::Interrupted {
                        interrupt,
                        snapshot: state.to_json()?,
                    });
                }
                StreamTransition::Complete(output) => {
                    self.save_running(&request.run_id, &thread_id, &state, None)
                        .await?;
                    trace.push(step_trace(&current_step, "complete"));
                    if send_event(
                        tx,
                        RunStreamEvent::StepFinished {
                            step: current_step,
                            result: StepResult::Complete,
                        },
                    )
                    .await
                    .is_err()
                    {
                        return Ok(StreamTerminal::ReceiverClosed);
                    }
                    return Ok(StreamTerminal::Completed {
                        trace,
                        output,
                        snapshot: state.to_json()?,
                    });
                }
                StreamTransition::Cancelled => return Ok(StreamTerminal::Cancelled),
                StreamTransition::ReceiverClosed => return Ok(StreamTerminal::ReceiverClosed),
            }
        }
    }

    async fn stream_thread_lock(
        &self,
        request: &RunRequest<M::Input>,
        tx: &StreamSender<M>,
        heartbeat_interval: std::time::Duration,
        control: &RunCancel,
    ) -> StreamLock {
        let lock = self.thread_lock_arc(&request.thread_id).await.lock_owned();
        async_rt::pin!(lock);
        let heartbeat = async_rt::time::sleep(heartbeat_interval);
        async_rt::pin!(heartbeat);

        loop {
            async_rt::select! {
                guard = &mut lock => {
                    return StreamLock::Acquired(guard);
                }
                _ = control.cancelled() => {
                    return StreamLock::Cancelled;
                }
                _ = tx.closed() => {
                    return StreamLock::ReceiverClosed;
                }
                _ = &mut heartbeat => {
                    if send_event(
                        tx,
                        RunStreamEvent::Heartbeat {
                            run_id: request.run_id.clone(),
                            session_id: request.session_id.clone(),
                            thread_id: request.thread_id.clone(),
                        },
                    )
                    .await
                    .is_err()
                    {
                        return StreamLock::ReceiverClosed;
                    }
                    heartbeat.as_mut().reset(async_rt::time::Instant::now() + heartbeat_interval);
                }
            }
        }
    }

    async fn transition_or_cancel(
        &self,
        step: M::Step,
        state: &mut M::State,
        ctx: &MachineRunContext<M>,
    ) -> Result<Transition<M::Step, M::Interrupt, M::Output>, MachineError> {
        async_rt::select! {
            result = self.machine.transition(step, state, ctx) => result,
            _ = ctx.cancelled() => Err(MachineError::Cancelled),
        }
    }

    async fn stream_transition(
        &self,
        step: M::Step,
        state: &mut M::State,
        ctx: &MachineRunContext<M>,
        tx: &StreamSender<M>,
        heartbeat_interval: std::time::Duration,
    ) -> Result<StreamTransition<M::Step, M::Output, M::Interrupt>, MachineError> {
        let transition = self.machine.transition(step, state, ctx);
        async_rt::pin!(transition);
        let heartbeat = async_rt::time::sleep(heartbeat_interval);
        async_rt::pin!(heartbeat);

        loop {
            async_rt::select! {
                result = &mut transition => {
                    return result.map(|transition| match transition {
                        Transition::Next(next) => StreamTransition::Next(next),
                        Transition::Interrupt(interrupt) => StreamTransition::Interrupt(interrupt),
                        Transition::Complete(output) => StreamTransition::Complete(output),
                    });
                }
                _ = &mut heartbeat => {
                    if send_event(
                        tx,
                        RunStreamEvent::Heartbeat {
                            run_id: ctx.run_id.clone(),
                            session_id: ctx.session_id.clone(),
                            thread_id: ctx.thread_id.clone(),
                        },
                    )
                    .await
                    .is_err()
                    {
                        return Ok(StreamTransition::ReceiverClosed);
                    }
                    heartbeat.as_mut().reset(async_rt::time::Instant::now() + heartbeat_interval);
                }
                _ = ctx.cancelled() => {
                    return Ok(StreamTransition::Cancelled);
                }
                _ = tx.closed() => {
                    return Ok(StreamTransition::ReceiverClosed);
                }
            }
        }
    }

    async fn register_active(&self, run_id: &RunId, control: Arc<RunCancel>) {
        let cancelled = self.cancelled_runs.lock().await.contains(run_id);
        if cancelled {
            control.cancel();
        }
        self.active_runs
            .lock()
            .await
            .insert(run_id.clone(), Arc::clone(&control));
        if self.cancelled_runs.lock().await.contains(run_id) {
            control.cancel();
        }
    }

    async fn unregister_active(&self, run_id: &RunId, _control: Option<&Arc<RunCancel>>) {
        self.active_runs.lock().await.remove(run_id);
        self.cancelled_runs.lock().await.remove(run_id);
    }

    #[cfg(test)]
    async fn active_run_count(&self) -> usize {
        self.active_runs.lock().await.len()
    }

    #[cfg(test)]
    async fn cancelled_run_count(&self) -> usize {
        self.cancelled_runs.lock().await.len()
    }

    fn restore_or_start(
        &self,
        request: &RunRequest<M::Input>,
        checkpoint: Option<CheckpointRecord>,
    ) -> Result<(M::State, M::Step), MachineError> {
        let Some(record) = checkpoint else {
            if request.command == RunCommand::Resume {
                return Err(MachineError::NoPendingInterrupt);
            }
            let state = self
                .machine
                .new_state(&request.input, None, request.snapshot.as_ref())?;
            return Ok((state, self.machine.start_step()));
        };

        let previous = M::State::from_json(&record.state)?;
        if let Some(raw_interrupt) = record.interrupt {
            if request.command != RunCommand::Resume {
                let state = self.machine.new_state(
                    &request.input,
                    Some(&previous),
                    request.snapshot.as_ref(),
                )?;
                return Ok((state, self.machine.start_step()));
            }
            let interrupt =
                serde_json::from_value::<M::Interrupt>(raw_interrupt).map_err(|err| {
                    MachineError::InvalidInterrupt {
                        reason: err.to_string(),
                    }
                })?;
            let mut state = previous;
            self.machine
                .apply_resume_input(&mut state, &request.input)?;
            let step = match self.machine.resume_action(&interrupt) {
                ResumeAction::ReenterInterruptedStep => {
                    let raw_step = record
                        .interrupted_step
                        .ok_or(MachineError::NoInterruptedStep)?;
                    serde_json::from_value::<M::Step>(raw_step).map_err(|err| {
                        MachineError::InvalidStep {
                            reason: err.to_string(),
                        }
                    })?
                }
                ResumeAction::JumpTo(step) => step,
            };
            return Ok((state, step));
        }

        if request.command == RunCommand::Resume {
            return Err(MachineError::NoPendingInterrupt);
        }

        let state =
            self.machine
                .new_state(&request.input, Some(&previous), request.snapshot.as_ref())?;
        let same_run_checkpoint = record.run_id.as_deref() == Some(request.run_id.as_str());
        let step = if same_run_checkpoint {
            match record.next_step {
                Some(raw_step) => serde_json::from_value::<M::Step>(raw_step).map_err(|err| {
                    MachineError::InvalidStep {
                        reason: err.to_string(),
                    }
                })?,
                None => self.machine.start_step(),
            }
        } else {
            self.machine.start_step()
        };
        Ok((state, step))
    }

    async fn save_running(
        &self,
        run_id: &RunId,
        thread_id: &ThreadId,
        state: &M::State,
        next_step: Option<&M::Step>,
    ) -> Result<(), MachineError> {
        let next_step = next_step
            .map(serde_json::to_value)
            .transpose()
            .map_err(MachineError::Serialization)?;
        self.checkpointer
            .save(
                thread_id.as_str(),
                &CheckpointRecord::running(state.to_json()?, next_step, run_id.as_str()),
            )
            .await
    }

    async fn save_interrupted(
        &self,
        run_id: &RunId,
        thread_id: &ThreadId,
        state: &M::State,
        interrupted_step: &M::Step,
        interrupt: &M::Interrupt,
    ) -> Result<(), MachineError> {
        let interrupted_step =
            serde_json::to_value(interrupted_step).map_err(MachineError::Serialization)?;
        let interrupt = serde_json::to_value(interrupt).map_err(MachineError::Serialization)?;
        self.checkpointer
            .save(
                thread_id.as_str(),
                &CheckpointRecord::interrupted(
                    state.to_json()?,
                    interrupted_step,
                    interrupt,
                    run_id.as_str(),
                ),
            )
            .await
    }

    async fn thread_lock(&self, thread_id: &ThreadId) -> async_rt::sync::OwnedMutexGuard<()> {
        self.thread_lock_arc(thread_id).await.lock_owned().await
    }

    async fn thread_lock_arc(&self, thread_id: &ThreadId) -> Arc<async_rt::sync::Mutex<()>> {
        let mut locks = self.thread_locks.lock().await;
        let entry = locks
            .entry(thread_id.clone())
            .or_insert_with(|| Arc::new(async_rt::sync::Mutex::new(())));
        let arc = Arc::clone(entry);
        drop(locks);
        arc
    }
}

fn step_trace<Step>(step: &Step, result: &str) -> Value
where
    Step: std::fmt::Debug,
{
    serde_json::json!({
        "step": format!("{step:?}"),
        "result": result,
    })
}

enum StreamTransition<Step, Output, Interrupt> {
    Next(Step),
    Interrupt(Interrupt),
    Complete(Output),
    Cancelled,
    ReceiverClosed,
}

enum StreamTerminal<Output, Interrupt> {
    Completed {
        trace: Vec<Value>,
        output: Output,
        snapshot: Value,
    },
    Interrupted {
        interrupt: Interrupt,
        snapshot: Value,
    },
    Cancelled,
    ReceiverClosed,
}

enum StreamLock {
    Acquired(async_rt::sync::OwnedMutexGuard<()>),
    Cancelled,
    ReceiverClosed,
}

async fn send_event<Step, Signal, Output, Interrupt>(
    tx: &async_rt::sync::mpsc::Sender<RunStreamEvent<Step, Signal, Output, Interrupt>>,
    event: RunStreamEvent<Step, Signal, Output, Interrupt>,
) -> Result<(), ()> {
    tx.send(event).await.map_err(|_| ())
}

#[cfg(test)]
#[path = "runner_tests.rs"]
mod runner_tests;
