use super::*;
use crate::checkpoint::{CheckpointRecord, CheckpointSaver, MemorySaver};
use crate::machine::{ResumeAction, Transition};
use crate::op::{Effect, EffectStatus, Entry, EntryQuery, EntryWrite, Item, Page, Vis};
use crate::run::{LeaseId, RunCommand, RuntimeLimits};
use crate::store::{
    Lease, LeaseClaim, MemoryRunStore, RunCommit, RunCommitResult, RunFinishRecord, RunLease,
    RunStore, RunTx, StoreStartResult,
};
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
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

#[derive(Debug, Clone, Serialize)]
struct Input {
    mode: Mode,
}

#[derive(Debug, Clone, Serialize)]
enum Mode {
    Complete,
    Interrupt,
    Loop,
    Slow,
    Ops,
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
            (Mode::Ops, Step::Start) => {
                ctx.reserve("effect-a", "tool", json!({"arg": 1})).await?;
                ctx.start("effect-a").await?;
                ctx.done("effect-a", json!({"ok": true})).await?;
                ctx.item("item-a", "artifact", json!({"value": 7})).await?;
                Ok(Transition::Next(Step::Done))
            }
            (Mode::Ops, Step::Done) => Ok(Transition::Complete("ops".to_string())),
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

fn tx_runtime(max: usize) -> TxRuntime<TestMachine, TestTxStore> {
    TxRuntime::new(TestMachine, Arc::new(TestTxStore::new()), max)
}

#[derive(Clone)]
struct TestTxStore {
    runs: MemoryRunStore<Event>,
    checkpoints: Arc<async_rt::sync::Mutex<HashMap<String, CheckpointRecord>>>,
}

impl TestTxStore {
    fn new() -> Self {
        Self {
            runs: MemoryRunStore::<Event>::new(),
            checkpoints: Arc::new(async_rt::sync::Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl CheckpointSaver for TestTxStore {
    async fn save(
        &self,
        thread_id: &str,
        checkpoint: &CheckpointRecord,
    ) -> Result<(), MachineError> {
        self.checkpoints
            .lock()
            .await
            .insert(thread_id.to_string(), checkpoint.clone());
        Ok(())
    }

    async fn load(&self, thread_id: &str) -> Result<Option<CheckpointRecord>, MachineError> {
        Ok(self.checkpoints.lock().await.get(thread_id).cloned())
    }
}

#[async_trait]
impl RunStore<Event> for TestTxStore {
    type Scope = Value;
    type FinishData = ();

    async fn ensure_session(
        &self,
        session_id: Option<SessionId>,
        scope: &Self::Scope,
    ) -> Result<SessionId, MachineError> {
        self.runs.ensure_session(session_id, scope).await
    }

    async fn start_run(
        &self,
        run: &RunStart<Self::Scope>,
    ) -> Result<StoreStartResult, MachineError> {
        self.runs.start_run(run).await
    }

    async fn lookup_run(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
    ) -> Result<Option<RunLookup>, MachineError> {
        self.runs.lookup_run(run_id, scope).await
    }

    async fn finish_run(
        &self,
        finish: &RunFinishRecord<Event, Self::FinishData, Self::Scope>,
    ) -> Result<FinishRunResult<Event>, MachineError> {
        self.runs.finish_run(finish).await
    }

    async fn terminal_event(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
    ) -> Result<Option<Event>, MachineError> {
        self.runs.terminal_event(run_id, scope).await
    }

    async fn find_idempotent_run(
        &self,
        scope: &Self::Scope,
        session_id: &SessionId,
        key: &str,
    ) -> Result<Option<RunLookup>, MachineError> {
        self.runs.find_idempotent_run(scope, session_id, key).await
    }

    async fn check_run_start(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
        input: Option<&Value>,
        entries: &[EntryWrite],
    ) -> Result<(), MachineError> {
        self.runs
            .check_run_start(run_id, scope, input, entries)
            .await
    }

    async fn mark_cancelled(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
    ) -> Result<(), MachineError> {
        self.runs.mark_cancelled(run_id, scope).await
    }

    async fn record_event(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
        event: &Event,
    ) -> Result<bool, MachineError> {
        self.runs.record_event(run_id, scope, event).await
    }

    async fn list_events(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
        after_seq: i64,
        limit: usize,
    ) -> Result<Page<Event>, MachineError> {
        self.runs.list_events(run_id, scope, after_seq, limit).await
    }
}

#[async_trait]
impl RunTx<Event> for TestTxStore {
    async fn commit_run(
        &self,
        commit: &RunCommit<Event, Self::FinishData, Self::Scope>,
    ) -> Result<RunCommitResult<Event>, MachineError> {
        if let Some(checkpoint) = &commit.checkpoint {
            self.save(checkpoint.thread_id.as_str(), &checkpoint.record)
                .await?;
        }
        self.runs.commit_run(commit).await
    }

    async fn reserve_effect(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
        lease: Option<&LeaseId>,
        key: &str,
        kind: &str,
        request: serde_json::Value,
    ) -> Result<Effect, MachineError> {
        self.runs
            .reserve_effect(run_id, scope, lease, key, kind, request)
            .await
    }

    async fn start_effect(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
        lease: Option<&LeaseId>,
        key: &str,
    ) -> Result<Effect, MachineError> {
        self.runs.start_effect(run_id, scope, lease, key).await
    }

    async fn list_items(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
        limit: usize,
    ) -> Result<Vec<Item>, MachineError> {
        self.runs.list_items(run_id, scope, limit).await
    }

    async fn list_effects(
        &self,
        run_id: &RunId,
        scope: &Self::Scope,
        limit: usize,
    ) -> Result<Vec<Effect>, MachineError> {
        self.runs.list_effects(run_id, scope, limit).await
    }

    async fn list_entries(
        &self,
        query: EntryQuery<'_, Self::Scope>,
    ) -> Result<Page<Entry>, MachineError> {
        self.runs.list_entries(query).await
    }

    async fn latest_entry(
        &self,
        scope: &Self::Scope,
        session_id: &SessionId,
        thread_id: Option<&ThreadId>,
        kind: &str,
        vis: Option<Vis>,
    ) -> Result<Option<Entry>, MachineError> {
        self.runs
            .latest_entry(scope, session_id, thread_id, kind, vis)
            .await
    }
}

#[async_trait]
impl RunLease<Event> for TestTxStore {
    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<bool, MachineError> {
        self.runs.renew(lease, ttl).await
    }

    async fn release(&self, lease: &Lease) -> Result<(), MachineError> {
        self.runs.release(lease).await
    }

    async fn reap_stale<F>(
        &self,
        owner: &WorkerId,
        limit: usize,
        build_event: F,
    ) -> Result<Vec<RunLookup>, MachineError>
    where
        F: FnMut(&RunLookup, i64) -> Event + Send,
    {
        self.runs.reap_stale(owner, limit, build_event).await
    }
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
            step_timeout: None,
            run_timeout: None,
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

#[path = "runtime_tests/control.rs"]
mod control;
#[path = "runtime_tests/stream.rs"]
mod stream;
