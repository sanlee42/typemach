use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use typemach::CheckpointSaver;
use typemach_agent::Artifact;

#[derive(Clone)]
struct ArtifactTools {
    artifacts: Vec<Artifact>,
    calls: Arc<AtomicUsize>,
}

impl ArtifactTools {
    fn new(artifacts: Vec<Artifact>) -> Self {
        Self {
            artifacts,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl ToolRegistry for ArtifactTools {
    async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok(["metric_point", "ask_user"]
            .into_iter()
            .map(|name| AgentToolSpec {
                name: name.to_string(),
                description: name.to_string(),
                input_schema: json!({ "type": "object" }),
                output_schema: Value::Null,
                metadata: Value::Null,
                annotations: ToolAnnotations::default(),
            })
            .collect())
    }

    async fn call_tool(&self, request: ToolCallRequest) -> Result<ToolResult, AgentError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        ToolResult::ok(&request.tool_use, json!({ "value": 42 }))
            .with_artifacts(self.artifacts.clone())
    }
}

fn artifact(tool_use_id: &str, title: &str, kind: &str, content: &str) -> Artifact {
    Artifact {
        tool_use_id: tool_use_id.to_string(),
        title: title.to_string(),
        kind: kind.to_string(),
        content: content.to_string(),
        source: None,
        window: None,
        updated_at: None,
    }
}

#[tokio::test]
async fn checkpoint_resume_keeps_external_artifacts_exactly_once_and_in_order() {
    let expected = vec![
        artifact("tool-1", "Orders", "table", "42"),
        artifact("tool-1", "Review", "markdown", "Orders are stable."),
    ];
    let tools = ArtifactTools::new(expected.clone());
    let calls = Arc::clone(&tools.calls);
    let model = ScriptedModel::new([
        ModelResponse {
            outcome: Some(ModelOutcome::ToolCalls {
                text: String::new(),
                calls: vec![
                    ToolUse {
                        id: "tool-1".to_string(),
                        name: "metric_point".to_string(),
                        input: json!({}),
                        raw: None,
                    },
                    ToolUse {
                        id: "ask-1".to_string(),
                        name: "ask_user".to_string(),
                        input: json!({ "question": "Continue?" }),
                        raw: None,
                    },
                ],
            }),
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse {
            outcome: Some(ModelOutcome::FinalAnswer {
                text: "Done.".to_string(),
            }),
            stop_reason: Some(StopReason::EndTurn),
            ..ModelResponse::default()
        },
    ]);
    let runner = build_agent_runner(MemorySaver::default(), model, tools, AllowAllTools);
    let input = AgentRunInput {
        messages: vec![AgentMessage::user_text("Read the metric")],
        context: Value::Null,
        budget: AgentBudget::default(),
        human_input: None,
        system_suffix: None,
    };

    let first = collect(runner.stream(request(input.clone()), StreamConfig::default())).await;
    assert!(first.iter().any(|event| matches!(
        event,
        RunStreamEvent::Interrupted { interrupt, .. } if interrupt.tool_use_id == "ask-1"
    )));
    assert_eq!(artifact_signals(&first), expected);
    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
    assert_eq!(state.artifacts, expected);

    let second = collect(runner.stream(
        RunRequest {
            command: RunCommand::Resume,
            input: AgentRunInput {
                messages: Vec::new(),
                human_input: Some(HumanInputAnswer {
                    tool_use_id: "ask-1".to_string(),
                    answer: "Yes".to_string(),
                }),
                ..input
            },
            ..request(AgentRunInput {
                messages: Vec::new(),
                context: Value::Null,
                budget: AgentBudget::default(),
                human_input: None,
                system_suffix: None,
            })
        },
        StreamConfig::default(),
    ))
    .await;

    assert!(artifact_signals(&second).is_empty());
    let output = completed(&second);
    assert_eq!(output.artifacts, expected);
    assert!(
        state
            .messages
            .iter()
            .chain(&output.messages)
            .flat_map(|message| match message {
                AgentMessage::User { content } | AgentMessage::Assistant { content } => content,
            })
            .filter_map(|block| match block {
                ContentBlock::ToolResult(result) => Some(result),
                _ => None,
            })
            .all(|result| result.artifacts.is_empty())
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn invalid_external_artifact_batch_fails_before_any_artifact_is_published() {
    let tool_use = ToolUse {
        id: "tool-1".to_string(),
        name: "metric_point".to_string(),
        input: json!({}),
        raw: None,
    };
    for invalid in [
        artifact("tool-1", " ", "markdown", "body"),
        artifact("tool-1", "title", "markdown", " "),
        artifact("tool-1", "title", "chart", "body"),
        artifact("other-tool", "title", "table", "body"),
    ] {
        let error = ToolResult::ok(&tool_use, Value::Null)
            .with_artifacts(vec![invalid])
            .expect_err("invalid artifact");
        assert!(matches!(error, AgentError::InvalidToolResult(_)));
    }

    let valid = artifact("tool-1", "Orders", "table", "42");
    let invalid = artifact("tool-1", "Review", "chart", "body");
    let mut result = ToolResult::ok(&tool_use, json!({ "value": 42 }));
    result.artifacts = vec![valid, invalid];
    let tools = InvalidArtifactTools(result);
    let model = ScriptedModel::new([ModelResponse {
        outcome: Some(ModelOutcome::ToolCalls {
            text: String::new(),
            calls: vec![tool_use],
        }),
        stop_reason: Some(StopReason::ToolUse),
        ..ModelResponse::default()
    }]);
    let runner = build_agent_runner(MemorySaver::default(), model, tools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Read the metric")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    assert!(events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Failed { error }
            if error.to_string().contains("invalid tool result")
                && error.to_string().contains("artifact 1")
    )));
    assert!(artifact_signals(&events).is_empty());
    assert!(!events.iter().any(|event| matches!(
        event,
        RunStreamEvent::Signal {
            signal: AgentSignal::ToolResult { .. } | AgentSignal::ToolCompleted { .. },
        }
    )));
}

struct InvalidArtifactTools(ToolResult);

#[async_trait]
impl ToolRegistry for InvalidArtifactTools {
    async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok(vec![AgentToolSpec {
            name: "metric_point".to_string(),
            description: "read metric point".to_string(),
            input_schema: json!({ "type": "object" }),
            output_schema: Value::Null,
            metadata: Value::Null,
            annotations: ToolAnnotations::default(),
        }])
    }

    async fn call_tool(&self, _request: ToolCallRequest) -> Result<ToolResult, AgentError> {
        Ok(self.0.clone())
    }
}

fn artifact_signals(
    events: &[RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>],
) -> Vec<Artifact> {
    events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal: AgentSignal::Artifact { artifact },
            } => Some(artifact.clone()),
            _ => None,
        })
        .collect()
}
