use super::*;
use crate::checkpoint::{CheckpointRecord, MemorySaver};
use crate::machine::Transition;
use crate::run::{
    RunCommand, RunId, RunStreamEvent, RuntimeLimits, SessionId, StreamConfig, ThreadId,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Step {
    Start,
    Finish,
    Loop,
    Resume,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestState {
    value: u32,
    answer: Option<String>,
}

#[derive(Debug, Clone)]
struct TestInput {
    value: u32,
    answer: Option<String>,
    mode: TestMode,
    drop_flag: Option<Arc<AtomicBool>>,
}

#[derive(Debug, Clone)]
enum TestMode {
    Complete,
    Loop,
    Interrupt,
    Emit,
    Slow,
}

#[derive(Debug, Clone)]
struct TestMachine;

#[async_trait]
impl Machine for TestMachine {
    type Step = Step;
    type State = TestState;
    type Input = TestInput;
    type Signal = String;
    type Output = String;
    type Interrupt = String;

    fn start_step(&self) -> Self::Step {
        Step::Start
    }

    fn resume_action(&self, _interrupt: &Self::Interrupt) -> ResumeAction<Self::Step> {
        ResumeAction::JumpTo(Step::Resume)
    }

    fn new_state(
        &self,
        input: &Self::Input,
        previous: Option<&Self::State>,
        _snapshot: Option<&Value>,
    ) -> Result<Self::State, MachineError> {
        Ok(TestState {
            value: previous.map(|state| state.value).unwrap_or(0) + input.value,
            answer: None,
        })
    }

    fn apply_resume_input(
        &self,
        state: &mut Self::State,
        input: &Self::Input,
    ) -> Result<(), MachineError> {
        state.answer = input.answer.clone();
        Ok(())
    }

    async fn transition(
        &self,
        step: Self::Step,
        state: &mut Self::State,
        _ctx: &RunContext<Self::Input, Self::Step, Self::Signal, Self::Output, Self::Interrupt>,
    ) -> Result<Transition<Self::Step, Self::Interrupt, Self::Output>, MachineError> {
        match (step, &_ctx.input.mode) {
            (Step::Start, TestMode::Complete) => {
                state.value += 1;
                Ok(Transition::Next(Step::Finish))
            }
            (Step::Finish, TestMode::Complete) => {
                Ok(Transition::Complete(format!("value={}", state.value)))
            }
            (Step::Start, TestMode::Loop) | (Step::Loop, TestMode::Loop) => {
                Ok(Transition::Next(Step::Loop))
            }
            (Step::Start, TestMode::Interrupt) => Ok(Transition::Interrupt("answer?".to_string())),
            (Step::Start, TestMode::Emit) => {
                _ctx.emit("signal-1".to_string()).await?;
                Ok(Transition::Complete("done".to_string()))
            }
            (Step::Start, TestMode::Slow) => {
                let _guard = _ctx.input.drop_flag.as_ref().map(|flag| DropNotice {
                    dropped: Arc::clone(flag),
                });
                async_rt::time::sleep(std::time::Duration::from_secs(5)).await;
                Ok(Transition::Complete("slow".to_string()))
            }
            (Step::Resume, _) => Ok(Transition::Complete(
                state.answer.clone().unwrap_or_default(),
            )),
            _ => Ok(Transition::Complete("done".to_string())),
        }
    }
}

struct DropNotice {
    dropped: Arc<AtomicBool>,
}

impl Drop for DropNotice {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

fn request(mode: TestMode) -> RunRequest<TestInput> {
    RunRequest {
        run_id: RunId::from("run-1"),
        session_id: SessionId::from("session-1"),
        thread_id: ThreadId::from("thread-1"),
        command: RunCommand::Start,
        input: TestInput {
            value: 1,
            answer: None,
            mode,
            drop_flag: None,
        },
        snapshot: None,
        runtime_limits: RuntimeLimits {
            max_steps: 5,
            allow_clarification: true,
        },
    }
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
fn typed_step_execution_completes() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let output = runner
            .invoke(request(TestMode::Complete))
            .await
            .expect("run should complete");

        match output {
            RunOutput::Completed {
                output, snapshot, ..
            } => {
                assert_eq!(output, "value=2");
                assert_eq!(snapshot["value"], 2);
            }
            RunOutput::Interrupted { .. } => panic!("expected completed"),
        }
    });
}

#[test]
fn max_steps_are_enforced() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let mut req = request(TestMode::Loop);
        req.runtime_limits.max_steps = 2;
        let err = runner.invoke(req).await.expect_err("run should fail");
        assert!(matches!(err, MachineError::MaxStepsExceeded { max: 2 }));
    });
}

#[test]
fn cancelled_run_stops_before_execution() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let req = request(TestMode::Complete);
        runner.cancel_run(&req.run_id).await;
        let err = runner.invoke(req).await.expect_err("run should cancel");
        assert!(matches!(err, MachineError::Cancelled));
    });
}

#[test]
fn cancelled_run_marker_is_cleared_after_run_exits() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let req = request(TestMode::Complete);
        runner.cancel_run(&req.run_id).await;
        let err = runner.invoke(req).await.expect_err("run should cancel");
        assert!(matches!(err, MachineError::Cancelled));
        assert_eq!(runner.cancelled_run_count().await, 0);
    });
}

#[test]
fn interrupt_checkpoint_resumes_to_completion() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));

        let first = runner
            .invoke(request(TestMode::Interrupt))
            .await
            .expect("run should interrupt");
        match first {
            RunOutput::Interrupted {
                interrupt,
                snapshot,
            } => {
                assert_eq!(interrupt, "answer?");
                assert_eq!(snapshot["value"], 1);
            }
            RunOutput::Completed { .. } => panic!("expected interrupt"),
        }

        let mut second = request(TestMode::Complete);
        second.command = RunCommand::Resume;
        second.input.answer = Some("resumed".to_string());
        let output = runner.invoke(second).await.expect("resume should complete");
        match output {
            RunOutput::Completed {
                output, snapshot, ..
            } => {
                assert_eq!(output, "resumed");
                assert_eq!(snapshot["answer"], "resumed");
            }
            RunOutput::Interrupted { .. } => panic!("expected completed"),
        }
    });
}

#[test]
fn checkpoint_scope_uses_explicit_thread_id() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let mut first = request(TestMode::Complete);
        first.session_id = SessionId::from("same-session");
        first.thread_id = ThreadId::from("thread-a");
        let mut second = request(TestMode::Complete);
        second.session_id = SessionId::from("same-session");
        second.thread_id = ThreadId::from("thread-b");

        let out_a = runner.invoke(first).await.expect("thread a");
        let out_b = runner.invoke(second).await.expect("thread b");
        match (out_a, out_b) {
            (
                RunOutput::Completed { snapshot: a, .. },
                RunOutput::Completed { snapshot: b, .. },
            ) => {
                assert_eq!(a["value"], 2);
                assert_eq!(b["value"], 2);
            }
            _ => panic!("expected completed outputs"),
        }
    });
}

#[test]
fn new_run_starts_at_start_step_when_thread_has_previous_running_checkpoint() {
    block_on(async {
        let saver = Arc::new(MemorySaver::new());
        saver
            .save(
                "thread-a",
                &CheckpointRecord::running(
                    serde_json::to_value(TestState {
                        value: 100,
                        answer: None,
                    })
                    .expect("state json"),
                    Some(serde_json::to_value(Step::Finish).expect("step json")),
                    "old-run",
                ),
            )
            .await
            .expect("save checkpoint");
        let runner = Runner::new(TestMachine, saver);

        let mut req = request(TestMode::Complete);
        req.run_id = RunId::from("new-run");
        req.thread_id = ThreadId::from("thread-a");
        let output = runner.invoke(req).await.expect("new run should complete");
        match output {
            RunOutput::Completed { output, .. } => {
                assert_eq!(output, "value=102");
            }
            RunOutput::Interrupted { .. } => panic!("expected completed"),
        }
    });
}

#[test]
fn resume_requires_pending_interrupt() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let mut req = request(TestMode::Complete);
        req.command = RunCommand::Resume;
        let err = runner.invoke(req).await.expect_err("resume should fail");
        assert!(matches!(err, MachineError::NoPendingInterrupt));
    });
}

#[test]
fn stream_started_and_heartbeat_arrive_before_slow_transition_finishes() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let mut req = request(TestMode::Slow);
        req.run_id = RunId::from("run-slow-heartbeat");
        let mut stream = runner.stream(
            req,
            StreamConfig {
                heartbeat_interval: std::time::Duration::from_millis(10),
                channel_capacity: 8,
            },
        );

        let first = stream.next_event().await.expect("started");
        assert!(matches!(first, RunStreamEvent::Started { .. }));
        let mut saw_heartbeat = false;
        for _ in 0..5 {
            let event = stream.next_event().await.expect("stream event");
            if matches!(event, RunStreamEvent::Heartbeat { .. }) {
                saw_heartbeat = true;
                break;
            }
        }
        assert!(saw_heartbeat, "expected heartbeat while transition runs");
    });
}

#[test]
fn stream_heartbeats_and_cancels_while_waiting_for_thread_lock() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let mut first = request(TestMode::Slow);
        first.run_id = RunId::from("run-lock-holder");
        let mut first_stream = runner.stream(
            first,
            StreamConfig {
                heartbeat_interval: std::time::Duration::from_millis(10),
                channel_capacity: 8,
            },
        );
        while let Some(event) = first_stream.next_event().await {
            if matches!(event, RunStreamEvent::StepStarted { .. }) {
                break;
            }
        }

        let mut second = request(TestMode::Slow);
        second.run_id = RunId::from("run-waiting-lock");
        let mut second_stream = runner.stream(
            second.clone(),
            StreamConfig {
                heartbeat_interval: std::time::Duration::from_millis(10),
                channel_capacity: 8,
            },
        );
        assert!(matches!(
            second_stream.next_event().await,
            Some(RunStreamEvent::Started { .. })
        ));
        let mut saw_wait_heartbeat = false;
        for _ in 0..5 {
            let event = second_stream
                .next_event()
                .await
                .expect("second stream event");
            if matches!(event, RunStreamEvent::Heartbeat { .. }) {
                saw_wait_heartbeat = true;
                break;
            }
        }
        assert!(
            saw_wait_heartbeat,
            "expected heartbeat while waiting for lock"
        );

        runner.cancel_run(&second.run_id).await;
        let mut saw_cancelled = false;
        while let Some(event) = second_stream.next_event().await {
            if matches!(event, RunStreamEvent::Cancelled) {
                saw_cancelled = true;
                break;
            }
        }
        assert!(saw_cancelled, "waiting stream did not cancel promptly");
        drop(first_stream);
    });
}

#[test]
fn stream_signal_is_ordered_before_terminal() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let mut stream = runner.stream(
            request(TestMode::Emit),
            StreamConfig {
                heartbeat_interval: std::time::Duration::from_secs(1),
                channel_capacity: 8,
            },
        );
        let mut saw_signal = false;
        while let Some(event) = stream.next_event().await {
            match event {
                RunStreamEvent::Signal { signal } => {
                    assert_eq!(signal, "signal-1");
                    saw_signal = true;
                }
                RunStreamEvent::Completed { output, .. } => {
                    assert!(saw_signal, "signal must arrive before terminal");
                    assert_eq!(output, "done");
                    break;
                }
                _ => {}
            }
        }
    });
}

#[test]
fn stream_cancel_during_transition_drops_slow_future() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let dropped = Arc::new(AtomicBool::new(false));
        let mut req = request(TestMode::Slow);
        req.run_id = RunId::from("run-cancel-slow");
        req.input.drop_flag = Some(Arc::clone(&dropped));
        let mut stream = runner.stream(
            req.clone(),
            StreamConfig {
                heartbeat_interval: std::time::Duration::from_secs(1),
                channel_capacity: 8,
            },
        );

        while let Some(event) = stream.next_event().await {
            if matches!(event, RunStreamEvent::StepStarted { .. }) {
                break;
            }
        }
        runner.cancel_run(&req.run_id).await;
        while let Some(event) = stream.next_event().await {
            if matches!(event, RunStreamEvent::Cancelled) {
                break;
            }
        }
        assert!(
            dropped.load(Ordering::SeqCst),
            "transition future was not dropped"
        );
    });
}

#[test]
fn stream_receiver_drop_stops_driver_and_cleans_active_run() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let dropped = Arc::new(AtomicBool::new(false));
        let mut req = request(TestMode::Slow);
        req.run_id = RunId::from("run-drop-receiver");
        req.input.drop_flag = Some(Arc::clone(&dropped));
        let mut stream = runner.stream(
            req,
            StreamConfig {
                heartbeat_interval: std::time::Duration::from_secs(1),
                channel_capacity: 8,
            },
        );
        while let Some(event) = stream.next_event().await {
            if matches!(event, RunStreamEvent::StepStarted { .. }) {
                break;
            }
        }
        drop(stream);
        for _ in 0..20 {
            if runner.active_run_count().await == 0 {
                break;
            }
            async_rt::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(runner.active_run_count().await, 0);
        assert!(
            dropped.load(Ordering::SeqCst),
            "transition future was not dropped"
        );
    });
}

#[test]
fn stream_interrupt_can_resume_with_invoke_checkpoint() {
    block_on(async {
        let runner = Runner::new(TestMachine, Arc::new(MemorySaver::new()));
        let mut stream = runner.stream(
            request(TestMode::Interrupt),
            StreamConfig {
                heartbeat_interval: std::time::Duration::from_secs(1),
                channel_capacity: 8,
            },
        );
        let mut interrupted = false;
        while let Some(event) = stream.next_event().await {
            if let RunStreamEvent::Interrupted { interrupt, .. } = event {
                assert_eq!(interrupt, "answer?");
                interrupted = true;
                break;
            }
        }
        assert!(interrupted);

        let mut second = request(TestMode::Complete);
        second.command = RunCommand::Resume;
        second.input.answer = Some("resumed-from-stream".to_string());
        let output = runner.invoke(second).await.expect("resume should complete");
        match output {
            RunOutput::Completed { output, .. } => {
                assert_eq!(output, "resumed-from-stream");
            }
            RunOutput::Interrupted { .. } => panic!("expected completed"),
        }
    });
}
