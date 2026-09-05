use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::*;
use tokio::time::sleep;
use typemach::CheckpointSaver;
use typemach_agent::{Artifact, ToolDisposition};

const RECEIPT: &str = "The allocation plan is ready below.";
type Event = RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>;

#[derive(Clone, Copy)]
enum PresentationMode {
    Single,
    Concurrent,
    SequentialFirst,
    SequentialBoth,
}

#[derive(Clone)]
struct PresentingTools {
    mode: PresentationMode,
    calls: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
}

impl PresentingTools {
    fn new(mode: PresentationMode) -> Self {
        Self {
            mode,
            calls: Arc::new(AtomicUsize::new(0)),
            active: Arc::new(AtomicUsize::new(0)),
            max_active: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl ToolRegistry for PresentingTools {
    async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok(vec![AgentToolSpec {
            name: "publish_plan".to_string(),
            description: "publish an authoritative plan".to_string(),
            input_schema: json!({ "type": "object" }),
            output_schema: Value::Null,
            metadata: Value::Null,
            annotations: ToolAnnotations {
                read_only: !matches!(
                    self.mode,
                    PresentationMode::SequentialFirst | PresentationMode::SequentialBoth
                ),
                ..ToolAnnotations::default()
            },
        }])
    }

    async fn call_tool(&self, request: ToolCallRequest) -> Result<ToolResult, AgentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        raise_max(&self.max_active, active);
        if matches!(self.mode, PresentationMode::Concurrent) {
            let delay = if request.tool_use.id == "tool-1" {
                40
            } else {
                10
            };
            sleep(Duration::from_millis(delay)).await;
        }
        self.active.fetch_sub(1, Ordering::SeqCst);

        let result = ToolResult::ok(
            &request.tool_use,
            json!({ "id": request.tool_use.id, "status": "ready" }),
        )
        .with_artifacts(vec![Artifact {
            tool_use_id: request.tool_use.id.clone(),
            title: format!("Plan {}", request.tool_use.id),
            kind: "markdown".to_string(),
            content: format!("Authoritative output for {}", request.tool_use.id),
            source: None,
            window: None,
            updated_at: None,
        }])?;

        let presents = match self.mode {
            PresentationMode::Single => true,
            PresentationMode::Concurrent => request.tool_use.id == "tool-2",
            PresentationMode::SequentialFirst => request.tool_use.id == "tool-1",
            PresentationMode::SequentialBoth => true,
        };
        if presents {
            result.present(RECEIPT)
        } else {
            Ok(result)
        }
    }
}

fn tool_turn(ids: &[&str]) -> ModelResponse {
    ModelResponse {
        stop_reason: Some(StopReason::ToolUse),
        ..tool_response(
            "",
            ids.iter()
                .map(|id| ToolUse {
                    id: (*id).to_string(),
                    name: "publish_plan".to_string(),
                    input: json!({}),
                    raw: None,
                })
                .collect(),
        )
    }
}

fn presentation_input(max_tool_calls: u32) -> AgentRunInput {
    AgentRunInput {
        messages: vec![AgentMessage::user_text("Build the plan")],
        context: Value::Null,
        budget: AgentBudget {
            max_model_turns: 1,
            max_tool_calls,
        },
        human_input: None,
        system_suffix: None,
    }
}

#[test]
fn legacy_results_continue_and_present_requires_successful_text() {
    let legacy = json!({
        "tool_use_id": "tool-1",
        "name": "publish_plan",
        "content": { "status": "ready" },
        "is_error": false
    });
    let result: ToolResult = serde_json::from_value(legacy.clone()).expect("legacy result");
    assert_eq!(result.disposition, ToolDisposition::Continue);
    assert_eq!(
        serde_json::to_value(result).expect("serialize result"),
        legacy
    );

    let tool_use = ToolUse {
        id: "tool-1".to_string(),
        name: "publish_plan".to_string(),
        input: Value::Null,
        raw: None,
    };
    assert!(
        ToolResult::ok(&tool_use, Value::Null)
            .present(" \n")
            .is_err()
    );
    assert!(
        ToolResult::error(&tool_use, "failed")
            .present(RECEIPT)
            .is_err()
    );
}

#[tokio::test]
async fn presented_tool_finishes_after_its_artifact_without_another_model_call() {
    let model = ScriptedModel::new([tool_turn(&["tool-1"])]);
    let tools = PresentingTools::new(PresentationMode::Single);
    let runner = build_agent_runner(MemorySaver::default(), model.clone(), tools, AllowAllTools);
    let events =
        collect(runner.stream(request(presentation_input(1)), StreamConfig::default())).await;

    let output = completed(&events);
    assert_eq!(output.answer, RECEIPT);
    assert_eq!(output.finish_reason, FinishReason::Stop);
    assert_eq!(output.artifacts.len(), 1);
    assert_eq!(model.requests().len(), 1);
    assert_tool_recorded_before_receipt(&events, "tool-1");
    assert_eq!(final_receipts(&events), [RECEIPT]);

    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    assert!(checkpoint.next_step.is_none());
    let state: AgentState = serde_json::from_value(checkpoint.state.clone()).expect("agent state");
    assert_eq!(state.model_turns, 1);
    assert_eq!(state.tool_calls, 1);
    assert_eq!(state.answer, RECEIPT);
    assert!(state.pending_tools.is_empty());
    assert!(tool_results(&state).all(|result| result.disposition == ToolDisposition::Continue));
    assert!(!checkpoint.state.to_string().contains("disposition"));
}

#[tokio::test]
async fn concurrent_batch_records_every_result_before_presenting_once() {
    let model = ScriptedModel::new([tool_turn(&["tool-1", "tool-2"])]);
    let tools = PresentingTools::new(PresentationMode::Concurrent);
    let max_active = Arc::clone(&tools.max_active);
    let runner = build_agent_runner(MemorySaver::default(), model.clone(), tools, AllowAllTools);
    let events =
        collect(runner.stream(request(presentation_input(2)), StreamConfig::default())).await;

    let output = completed(&events);
    assert_eq!(max_active.load(Ordering::SeqCst), 2);
    assert_eq!(model.requests().len(), 1);
    assert_eq!(output.answer, RECEIPT);
    assert_eq!(tool_result_ids(&events), ["tool-1", "tool-2"]);
    assert_eq!(
        output
            .artifacts
            .iter()
            .map(|artifact| artifact.tool_use_id.as_str())
            .collect::<Vec<_>>(),
        ["tool-1", "tool-2"]
    );
    for tool_use_id in ["tool-1", "tool-2"] {
        assert_tool_recorded_before_receipt(&events, tool_use_id);
    }
    assert_eq!(final_receipts(&events), [RECEIPT]);

    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    assert!(checkpoint.next_step.is_none());
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
    assert_eq!(state.model_turns, 1);
    assert_eq!(state.tool_calls, 2);
    assert_eq!(state.answer, RECEIPT);
    assert_eq!(tool_results(&state).count(), 2);
    assert!(tool_results(&state).all(|result| result.disposition == ToolDisposition::Continue));
}

#[tokio::test]
async fn sequential_batch_finishes_only_after_calls_following_present() {
    let model = ScriptedModel::new([tool_turn(&["tool-1", "tool-2"])]);
    let tools = PresentingTools::new(PresentationMode::SequentialFirst);
    let calls = Arc::clone(&tools.calls);
    let runner = build_agent_runner(MemorySaver::default(), model.clone(), tools, AllowAllTools);
    let events =
        collect(runner.stream(request(presentation_input(2)), StreamConfig::default())).await;

    let output = completed(&events);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(model.requests().len(), 1);
    assert_eq!(output.answer, RECEIPT);
    assert_eq!(tool_result_ids(&events), ["tool-1", "tool-2"]);
    assert_eq!(
        output
            .artifacts
            .iter()
            .map(|artifact| artifact.tool_use_id.as_str())
            .collect::<Vec<_>>(),
        ["tool-1", "tool-2"]
    );
    for tool_use_id in ["tool-1", "tool-2"] {
        assert_tool_recorded_before_receipt(&events, tool_use_id);
    }
    assert_eq!(final_receipts(&events), [RECEIPT]);

    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    assert!(checkpoint.next_step.is_none());
    assert!(
        !checkpoint
            .state
            .to_string()
            .contains("interrupted before completion")
    );
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
    assert_eq!(state.tool_calls, 2);
    assert_eq!(tool_use_ids(&state), ["tool-1", "tool-2"]);
    let results = tool_results(&state).collect::<Vec<_>>();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| {
        !result.is_error
            && result.content["status"] == "ready"
            && result.disposition == ToolDisposition::Continue
    }));
}

#[tokio::test]
async fn sequential_batch_rejects_multiple_presentations_after_recording_both() {
    let model = ScriptedModel::new([tool_turn(&["tool-1", "tool-2"])]);
    let tools = PresentingTools::new(PresentationMode::SequentialBoth);
    let calls = Arc::clone(&tools.calls);
    let runner = build_agent_runner(MemorySaver::default(), model.clone(), tools, AllowAllTools);
    let events =
        collect(runner.stream(request(presentation_input(2)), StreamConfig::default())).await;

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(model.requests().len(), 1);
    assert_eq!(tool_result_ids(&events), ["tool-1", "tool-2"]);
    assert!(final_receipts(&events).is_empty());
    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Failed { error }
            if error.to_string().contains("invalid tool result")
                && error.to_string().contains("more than one final answer")
    )));
}

fn signal_position(events: &[Event], predicate: impl Fn(&AgentSignal) -> bool) -> usize {
    events
        .iter()
        .position(|event| matches!(event, RunStreamEvent::Signal { signal } if predicate(signal)))
        .expect("signal position")
}

fn raise_max(max: &AtomicUsize, value: usize) {
    let mut current = max.load(Ordering::SeqCst);
    while value > current {
        match max.compare_exchange(current, value, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return,
            Err(next) => current = next,
        }
    }
}

fn tool_result_ids(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal: AgentSignal::ToolResult { tool_use_id, .. },
            } => Some(tool_use_id.clone()),
            _ => None,
        })
        .collect()
}

fn final_receipt_position(events: &[Event]) -> usize {
    signal_position(events, |signal| {
        matches!(signal, AgentSignal::AssistantMessageDelta { .. })
    })
}

fn assert_tool_recorded_before_receipt(events: &[Event], tool_use_id: &str) {
    let result = signal_position(events, |signal| {
        matches!(
            signal,
            AgentSignal::ToolResult { tool_use_id: result_id, .. } if result_id == tool_use_id
        )
    });
    let artifact = signal_position(events, |signal| {
        matches!(
            signal,
            AgentSignal::Artifact { artifact } if artifact.tool_use_id == tool_use_id
        )
    });
    let completed = signal_position(events, |signal| {
        matches!(
            signal,
            AgentSignal::ToolCompleted { tool_use_id: completed_id, .. }
                if completed_id == tool_use_id
        )
    });
    assert!(result < artifact);
    assert!(artifact < completed);
    assert!(completed < final_receipt_position(events));
}

fn final_receipts(events: &[Event]) -> Vec<&str> {
    events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal: AgentSignal::AssistantMessageDelta { delta, .. },
            } => Some(delta.as_str()),
            _ => None,
        })
        .collect()
}

fn tool_results(state: &AgentState) -> impl Iterator<Item = &ToolResult> {
    state
        .messages
        .iter()
        .flat_map(|message| match message {
            AgentMessage::User { content } | AgentMessage::Assistant { content } => content,
        })
        .filter_map(|block| match block {
            ContentBlock::ToolResult(result) => Some(result),
            _ => None,
        })
}

fn tool_use_ids(state: &AgentState) -> Vec<&str> {
    state
        .messages
        .iter()
        .flat_map(|message| match message {
            AgentMessage::User { content } | AgentMessage::Assistant { content } => content,
        })
        .filter_map(|block| match block {
            ContentBlock::ToolUse(tool_use) => Some(tool_use.id.as_str()),
            _ => None,
        })
        .collect()
}
