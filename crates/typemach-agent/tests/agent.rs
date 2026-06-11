use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};
use typemach::{
    MemorySaver, RunCommand, RunId, RunRequest, RunStreamEvent, RuntimeLimits, SessionId,
    StreamConfig, ThreadId,
};
use typemach_agent::{
    AgentBudget, AgentError, AgentEventReceiver, AgentMessage, AgentModel, AgentRunInput,
    AgentRunOutput, AgentSignal, AgentStep, AgentToolSpec, AllowAllTools, AskUserQuestion,
    ContentBlock, ContextPolicy, FinishReason, HumanInputAnswer, ModelRequest, ModelResponse,
    ModelStream, StopReason, TerminalAction, ToolAnnotations, ToolCallRequest, ToolRegistry,
    ToolResult, ToolUse, build_agent_runner, build_agent_runner_with_context_policy,
};

#[derive(Clone, Default)]
struct ScriptedModel {
    responses: Arc<Mutex<VecDeque<ModelResponse>>>,
    requests: Arc<Mutex<Vec<ModelRequest>>>,
}

impl ScriptedModel {
    fn new(responses: impl IntoIterator<Item = ModelResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses.into_iter().collect())),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn requests(&self) -> Vec<ModelRequest> {
        self.requests.lock().expect("requests lock").clone()
    }
}

#[async_trait]
impl AgentModel for ScriptedModel {
    async fn next_step(
        &self,
        request: ModelRequest,
        stream: ModelStream,
    ) -> Result<ModelResponse, AgentError> {
        self.requests.lock().expect("requests lock").push(request);
        let response = self
            .responses
            .lock()
            .expect("responses lock")
            .pop_front()
            .ok_or_else(|| AgentError::Model("script exhausted".to_string()))?;
        for delta in &response.deltas {
            stream.delta(delta.clone())?;
        }
        Ok(ModelResponse {
            deltas: Vec::new(),
            ..response
        })
    }
}

#[derive(Default)]
struct FakeTools;

#[async_trait]
impl ToolRegistry for FakeTools {
    async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok(vec![
            AgentToolSpec {
                name: "metric_point".to_string(),
                description: "read metric point".to_string(),
                input_schema: json!({ "type": "object" }),
                output_schema: Value::Null,
                metadata: Value::Null,
                annotations: ToolAnnotations::default(),
            },
            AgentToolSpec {
                name: "ask_user".to_string(),
                description: "ask user".to_string(),
                input_schema: json!({ "type": "object" }),
                output_schema: Value::Null,
                metadata: Value::Null,
                annotations: ToolAnnotations::default(),
            },
            AgentToolSpec {
                name: "emit_artifact".to_string(),
                description: "emit artifact".to_string(),
                input_schema: json!({ "type": "object" }),
                output_schema: Value::Null,
                metadata: Value::Null,
                annotations: ToolAnnotations::default(),
            },
            AgentToolSpec {
                name: "report".to_string(),
                description: "finish with report".to_string(),
                input_schema: json!({ "type": "object" }),
                output_schema: Value::Null,
                metadata: Value::Null,
                annotations: ToolAnnotations {
                    terminal: true,
                    ..ToolAnnotations::default()
                },
            },
        ])
    }

    async fn call_tool(&self, request: ToolCallRequest) -> Result<ToolResult, AgentError> {
        Ok(ToolResult::ok(
            &request.tool_use,
            json!({ "value": 42, "ds": "2026-06-08" }),
        ))
    }
}

#[derive(Default)]
struct LargeTools;

#[async_trait]
impl ToolRegistry for LargeTools {
    async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok(vec![AgentToolSpec {
            name: "large_evidence".to_string(),
            description: "read large evidence".to_string(),
            input_schema: json!({ "type": "object" }),
            output_schema: Value::Null,
            metadata: Value::Null,
            annotations: ToolAnnotations::default(),
        }])
    }

    async fn call_tool(&self, request: ToolCallRequest) -> Result<ToolResult, AgentError> {
        Ok(ToolResult::ok(
            &request.tool_use,
            json!({ "rows": ["x".repeat(128)] }),
        ))
    }
}

fn request(input: AgentRunInput) -> RunRequest<AgentRunInput> {
    RunRequest {
        run_id: RunId::from("run-1"),
        session_id: SessionId::from("session-1"),
        thread_id: ThreadId::from("thread-1"),
        command: RunCommand::Start,
        input,
        snapshot: None,
        runtime_limits: RuntimeLimits::new(32),
    }
}

async fn collect(
    mut rx: AgentEventReceiver,
) -> Vec<RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>> {
    let mut events = Vec::new();
    while let Some(event) = rx.next_event().await {
        let terminal = matches!(
            event,
            RunStreamEvent::Completed { .. }
                | RunStreamEvent::Interrupted { .. }
                | RunStreamEvent::Failed { .. }
                | RunStreamEvent::Cancelled
        );
        events.push(event);
        if terminal {
            break;
        }
    }
    events
}

#[tokio::test]
async fn model_tool_result_model_loop_completes() {
    let model = ScriptedModel::new([
        ModelResponse {
            tool_uses: vec![ToolUse {
                id: "tool-1".to_string(),
                name: "metric_point".to_string(),
                input: json!({ "metric_id": "paid_order_count", "ds": "2026-06-08" }),
                raw: None,
            }],
            ..ModelResponse::default()
        },
        ModelResponse {
            deltas: vec!["订单量是 42。".to_string()],
            final_text: Some(String::new()),
            ..ModelResponse::default()
        },
    ]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
        AllowAllTools,
    );
    let rx = runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("昨天订单量")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
        }),
        StreamConfig::default(),
    );
    let events = collect(rx).await;
    assert!(events.iter().any(|event| matches!(
      event,
      RunStreamEvent::Signal {
        signal: AgentSignal::ToolStarted { name, .. },
      } if name == "metric_point"
    )));
    assert!(events.iter().any(|event| matches!(
      event,
      RunStreamEvent::Signal {
        signal: AgentSignal::ToolResult { name, content, .. },
      } if name == "metric_point" && content["value"] == 42
    )));
    let completed = completed(&events);
    assert_eq!(completed.answer, "订单量是 42。");
    assert_eq!(completed.finish_reason, FinishReason::Stop);
}

#[tokio::test]
async fn final_text_without_stream_deltas_is_emitted_and_persisted() {
    let model = ScriptedModel::new([ModelResponse {
        final_text: Some("订单量是 42。".to_string()),
        ..ModelResponse::default()
    }]);
    let runner = build_agent_runner(MemorySaver::default(), model, FakeTools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("昨天订单量")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
        }),
        StreamConfig::default(),
    ))
    .await;
    assert!(events.iter().any(|event| matches!(
      event,
      RunStreamEvent::Signal {
        signal: AgentSignal::AssistantDelta { delta, .. },
      } if delta == "订单量是 42。"
    )));
    let completed = completed(&events);
    assert_eq!(completed.answer, "订单量是 42。");
    assert!(matches!(
        completed.messages.last(),
        Some(AgentMessage::Assistant { content })
            if content == &vec![ContentBlock::Text {
                text: "订单量是 42。".to_string()
            }]
    ));
}

#[tokio::test]
async fn ask_user_interrupts_and_resume_continues() {
    let model = ScriptedModel::new([
        ModelResponse {
            tool_uses: vec![ToolUse {
                id: "ask-1".to_string(),
                name: "ask_user".to_string(),
                input: json!({ "question": "看哪个日期？", "fields": { "type": "choice" } }),
                raw: None,
            }],
            ..ModelResponse::default()
        },
        ModelResponse {
            deltas: vec!["按 2026-06-08 查看，订单量是 42。".to_string()],
            final_text: Some(String::new()),
            ..ModelResponse::default()
        },
    ]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
        AllowAllTools,
    );
    let mut first = runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("订单量")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
        }),
        StreamConfig::default(),
    );
    let interrupted = loop {
        match first.next_event().await.expect("event") {
            RunStreamEvent::Interrupted { interrupt, .. } => break interrupt,
            RunStreamEvent::Failed { error } => panic!("failed: {error}"),
            _ => {}
        }
    };
    assert_eq!(interrupted.question, "看哪个日期？");

    let resume = RunRequest {
        command: RunCommand::Resume,
        input: AgentRunInput {
            messages: Vec::new(),
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: Some(HumanInputAnswer {
                tool_use_id: "ask-1".to_string(),
                answer: "2026-06-08".to_string(),
            }),
        },
        ..request(AgentRunInput {
            messages: Vec::new(),
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
        })
    };
    let events = collect(runner.stream(resume, StreamConfig::default())).await;
    let completed = completed(&events);
    assert_eq!(completed.answer, "按 2026-06-08 查看，订单量是 42。");
    let requests = model.requests();
    let second = requests.get(1).expect("second model request");
    assert!(second.messages.iter().any(|message| matches!(
        message,
        AgentMessage::User { content }
            if content.iter().any(|block| matches!(
                block,
                ContentBlock::ToolResult(result)
                    if result.tool_use_id == "ask-1"
                        && result.content == json!({ "answer": "2026-06-08" })
            ))
    )));
}

#[tokio::test]
async fn emit_artifact_is_signalled_and_not_required_in_answer() {
    let model = ScriptedModel::new([
        ModelResponse {
            tool_uses: vec![ToolUse {
                id: "artifact-1".to_string(),
                name: "emit_artifact".to_string(),
                input: json!({
                  "title": "经营复盘",
                  "type": "markdown",
                  "content": "# 经营复盘"
                }),
                raw: None,
            }],
            ..ModelResponse::default()
        },
        ModelResponse {
            deltas: vec!["复盘已生成。".to_string()],
            final_text: Some(String::new()),
            ..ModelResponse::default()
        },
    ]);
    let runner = build_agent_runner(MemorySaver::default(), model, FakeTools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("生成复盘")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
        }),
        StreamConfig::default(),
    ))
    .await;
    assert!(events.iter().any(|event| matches!(
      event,
      RunStreamEvent::Signal {
        signal: AgentSignal::Artifact { artifact },
      } if artifact.title == "经营复盘"
    )));
    let completed = completed(&events);
    assert_eq!(completed.answer, "复盘已生成。");
    assert_eq!(completed.artifacts.len(), 1);
}

#[tokio::test]
async fn reasoning_blocks_are_persisted_without_polluting_answer() {
    let model = ScriptedModel::new([
        ModelResponse {
            content: vec![
                ContentBlock::Thinking {
                    text: "I should inspect the metric.".to_string(),
                    signature: Some("sig-1".to_string()),
                },
                ContentBlock::ToolUse(ToolUse {
                    id: "tool-1".to_string(),
                    name: "metric_point".to_string(),
                    input: json!({ "metric_id": "paid_order_count", "ds": "2026-06-08" }),
                    raw: Some(json!({ "id": "tool-1", "index": 0 })),
                }),
            ],
            stop_reason: Some(StopReason::ToolUse),
            response_id: Some("msg-1".to_string()),
            raw: Some(json!({ "id": "msg-1" })),
            ..ModelResponse::default()
        },
        ModelResponse {
            content: vec![ContentBlock::Text {
                text: "订单量是 42。".to_string(),
            }],
            stop_reason: Some(StopReason::EndTurn),
            ..ModelResponse::default()
        },
    ]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
        AllowAllTools,
    );
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("昨天订单量")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
        }),
        StreamConfig::default(),
    ))
    .await;
    let completed = completed(&events);
    assert_eq!(completed.answer, "订单量是 42。");
    assert!(matches!(
        completed.messages.get(1),
        Some(AgentMessage::Assistant { content })
            if matches!(
                content.first(),
                Some(ContentBlock::Thinking {
                    signature: Some(sig),
                    ..
                }) if sig == "sig-1"
            )
    ));
}

#[tokio::test]
async fn terminal_tool_completes_without_dispatching_registry_tool() {
    let model = ScriptedModel::new([ModelResponse {
        tool_uses: vec![ToolUse {
            id: "term-1".to_string(),
            name: "report".to_string(),
            input: json!({ "message": "当前没有足够证据继续。" }),
            raw: None,
        }],
        stop_reason: Some(StopReason::ToolUse),
        ..ModelResponse::default()
    }]);
    let runner = build_agent_runner(MemorySaver::default(), model, FakeTools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("生成报告")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
        }),
        StreamConfig::default(),
    ))
    .await;
    assert!(events.iter().any(|event| matches!(
      event,
      RunStreamEvent::Signal {
        signal: AgentSignal::Terminal { action },
      } if action.name == "report"
    )));
    let completed = completed(&events);
    assert_eq!(completed.finish_reason, FinishReason::Terminal);
    assert_eq!(completed.answer, "当前没有足够证据继续。");
    assert!(matches!(
        completed.terminal.as_ref(),
        Some(TerminalAction { name, .. }) if name == "report"
    ));
}

#[tokio::test]
async fn compacted_prompt_window_does_not_drop_persisted_messages() {
    let model = ScriptedModel::new([ModelResponse {
        final_text: Some("继续。".to_string()),
        ..ModelResponse::default()
    }]);
    let context_policy = ContextPolicy {
        compact_at_tokens: 1,
        max_input_tokens: 128,
        recent_turns: 2,
        ..ContextPolicy::default()
    };
    let runner = build_agent_runner_with_context_policy(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
        AllowAllTools,
        context_policy,
    );
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![
                AgentMessage::user_text("turn 1"),
                AgentMessage::assistant_text("turn 2"),
                AgentMessage::user_text("turn 3"),
                AgentMessage::assistant_text("turn 4"),
                AgentMessage::user_text("turn 5"),
            ],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
        }),
        StreamConfig::default(),
    ))
    .await;
    assert!(events.iter().any(|event| matches!(
      event,
      RunStreamEvent::Signal {
        signal: AgentSignal::ContextCompacted { compaction },
      } if compaction.archive.message_count == 3
    )));
    let requests = model.requests();
    let request = requests.first().expect("model request");
    assert_eq!(request.messages.len(), 3);
    assert!(matches!(
        request.messages.first(),
        Some(AgentMessage::User { content })
            if matches!(content.first(), Some(ContentBlock::ConversationDigest(_)))
    ));
    let completed = completed(&events);
    assert_eq!(completed.messages.len(), 6);
    assert!(completed.digest.is_some());
}

#[tokio::test]
async fn large_tool_result_is_archived_before_next_prompt() {
    let model = ScriptedModel::new([
        ModelResponse {
            tool_uses: vec![ToolUse {
                id: "tool-1".to_string(),
                name: "large_evidence".to_string(),
                input: json!({}),
                raw: None,
            }],
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse {
            final_text: Some("已读取。".to_string()),
            ..ModelResponse::default()
        },
    ]);
    let context_policy = ContextPolicy {
        max_tool_result_bytes: 32,
        ..ContextPolicy::default()
    };
    let runner = build_agent_runner_with_context_policy(
        MemorySaver::default(),
        model.clone(),
        LargeTools,
        AllowAllTools,
        context_policy,
    );
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("读取大证据")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
        }),
        StreamConfig::default(),
    ))
    .await;
    assert!(events.iter().any(|event| matches!(
      event,
      RunStreamEvent::Signal {
        signal: AgentSignal::ToolResult { content, .. },
      } if content["rows"][0].as_str().is_some_and(|value| value.len() == 128)
    )));
    assert!(events.iter().any(|event| matches!(
      event,
      RunStreamEvent::Signal {
        signal: AgentSignal::ToolResultArchived { archive },
      } if archive.name == "large_evidence" && archive.byte_count > 32
    )));
    let requests = model.requests();
    let second = requests.get(1).expect("second model request");
    assert!(second.messages.iter().any(|message| matches!(
        message,
        AgentMessage::User { content }
            if content.iter().any(|block| matches!(
                block,
                ContentBlock::ToolResult(result)
                    if result.content["archived"] == true
                        && result.content["sha256"].as_str().is_some()
            ))
    )));
    let completed = completed(&events);
    assert_eq!(completed.tool_result_archives.len(), 1);
}

#[test]
fn content_block_serde_shape_is_flat_and_defaults_annotations() {
    let tool_use = ContentBlock::ToolUse(ToolUse {
        id: "tool-1".to_string(),
        name: "metric_point".to_string(),
        input: json!({ "metric_id": "paid_order_count" }),
        raw: None,
    });
    assert_eq!(
        serde_json::to_value(tool_use).expect("serialize tool use"),
        json!({
            "type": "tool_use",
            "id": "tool-1",
            "name": "metric_point",
            "input": { "metric_id": "paid_order_count" }
        })
    );
    let annotations: ToolAnnotations =
        serde_json::from_value(json!({ "terminal": true })).expect("annotations");
    assert!(annotations.read_only);
    assert!(annotations.terminal);
}

fn completed(
    events: &[RunStreamEvent<AgentStep, AgentSignal, AgentRunOutput, AskUserQuestion>],
) -> &AgentRunOutput {
    events
        .iter()
        .find_map(|event| match event {
            RunStreamEvent::Completed { output, .. } => Some(output),
            _ => None,
        })
        .expect("completed output")
}
