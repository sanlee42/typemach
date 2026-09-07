use super::*;
use typemach_agent::{ResultId, RetainedResult};

#[derive(Clone, Default)]
struct RetainingTools {
    calls: Arc<Mutex<Vec<ToolCallRequest>>>,
}

#[async_trait]
impl ToolRegistry for RetainingTools {
    async fn list_tools(&self, _context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok([
            "metric_point",
            "inspect",
            "prime",
            "left",
            "right",
            "ask_user",
        ]
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
        self.calls.lock().expect("calls").push(request.clone());
        if request.tool_use.name == "inspect" {
            return Ok(ToolResult::ok(&request.tool_use, json!({ "ok": true })));
        }
        ToolResult::ok(
            &request.tool_use,
            json!({ "result_ref": request.tool_use.id }),
        )
        .retain(
            json!({ "private_value": format!("private-{}", request.tool_use.name) }),
            json!({ "scope": request.tool_use.name }),
        )
    }
}

fn retained(id: &str, value: &str) -> RetainedResult {
    RetainedResult::new(
        ResultId::new(id).expect("result id"),
        json!({ "private_value": value }),
        json!({ "scope": value }),
    )
}

fn ids(results: &[RetainedResult]) -> Vec<&str> {
    results.iter().map(|result| result.id().as_str()).collect()
}

#[tokio::test]
async fn retained_results_are_host_only_and_follow_turn_lifecycle() {
    let model = ScriptedModel::new([
        tool_response(
            "",
            vec![
                ToolUse {
                    id: "metric-1".to_string(),
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
        ),
        tool_response(
            "",
            vec![ToolUse {
                id: "inspect-resume".to_string(),
                name: "inspect".to_string(),
                input: json!({}),
                raw: None,
            }],
        ),
        final_response("resumed"),
        tool_response(
            "",
            vec![ToolUse {
                id: "inspect-fresh".to_string(),
                name: "inspect".to_string(),
                input: json!({}),
                raw: None,
            }],
        ),
        final_response("fresh"),
    ]);
    let tools = RetainingTools::default();
    let calls = Arc::clone(&tools.calls);
    let runner = build_agent_runner(MemorySaver::default(), model.clone(), tools, AllowAllTools);
    let initial = AgentRunInput {
        messages: vec![AgentMessage::user_text("inspect page")],
        context: json!({ "grant": "initial" }),
        retained_results: vec![retained("page", "original-page")],
        budget: AgentBudget::default(),
        human_input: None,
        system_suffix: None,
    };

    let first = collect(runner.stream(request(initial), StreamConfig::default())).await;
    assert!(first.iter().any(|event| matches!(
        event,
        RunStreamEvent::Interrupted { interrupt, .. } if interrupt.tool_use_id == "ask-1"
    )));
    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("state");
    assert_eq!(ids(&state.retained_results), ["page", "metric-1"]);

    let resume = RunRequest {
        command: RunCommand::Resume,
        input: AgentRunInput {
            messages: Vec::new(),
            context: json!({ "grant": "refreshed" }),
            retained_results: vec![retained("replacement", "must-not-replace")],
            budget: AgentBudget::default(),
            human_input: Some(HumanInputAnswer {
                tool_use_id: "ask-1".to_string(),
                answer: "yes".to_string(),
            }),
            system_suffix: None,
        },
        ..request(AgentRunInput {
            messages: Vec::new(),
            context: Value::Null,
            retained_results: Vec::new(),
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        })
    };
    let resumed = collect(runner.stream(resume, StreamConfig::default())).await;
    assert_eq!(completed(&resumed).answer, "resumed");

    let fresh = AgentRunInput {
        messages: vec![AgentMessage::user_text("new turn")],
        context: json!({ "grant": "new-turn" }),
        retained_results: vec![retained("new-page", "new-page")],
        budget: AgentBudget::default(),
        human_input: None,
        system_suffix: None,
    };
    let fresh_events = collect(runner.stream(request(fresh), StreamConfig::default())).await;
    let output = completed(&fresh_events);
    assert_eq!(output.answer, "fresh");

    let calls = calls.lock().expect("calls");
    assert_eq!(ids(&calls[0].retained_results), ["page"]);
    assert_eq!(ids(&calls[1].retained_results), ["page", "metric-1"]);
    assert_eq!(calls[1].context, json!({ "grant": "refreshed" }));
    assert_eq!(ids(&calls[2].retained_results), ["new-page"]);
    assert_eq!(calls[2].context, json!({ "grant": "new-turn" }));

    let model_json = serde_json::to_string(&model.requests()).expect("model requests");
    let output_json = serde_json::to_string(output).expect("run output");
    assert!(!model_json.contains("retained_results"));
    assert!(!output_json.contains("retained_results"));
    let signal_json = serde_json::to_string(
        &first
            .iter()
            .chain(&resumed)
            .chain(&fresh_events)
            .filter_map(|event| match event {
                RunStreamEvent::Signal {
                    signal: AgentSignal::ToolResult { content, .. },
                } => Some(content),
                _ => None,
            })
            .collect::<Vec<_>>(),
    )
    .expect("tool result signals");
    for private in [
        "original-page",
        "private-metric_point",
        "must-not-replace",
        "new-page",
    ] {
        assert!(!model_json.contains(private));
        assert!(!output_json.contains(private));
        assert!(!signal_json.contains(private));
    }
}

#[tokio::test]
async fn concurrent_siblings_receive_the_same_prior_result_snapshot() {
    let model = ScriptedModel::new([
        tool_response(
            "",
            vec![ToolUse {
                id: "prime-1".to_string(),
                name: "prime".to_string(),
                input: json!({}),
                raw: None,
            }],
        ),
        tool_response(
            "",
            vec![
                ToolUse {
                    id: "left-1".to_string(),
                    name: "left".to_string(),
                    input: json!({}),
                    raw: None,
                },
                ToolUse {
                    id: "right-1".to_string(),
                    name: "right".to_string(),
                    input: json!({}),
                    raw: None,
                },
            ],
        ),
        final_response("done"),
    ]);
    let tools = RetainingTools::default();
    let calls = Arc::clone(&tools.calls);
    let runner = build_agent_runner(MemorySaver::default(), model, tools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("parallel")],
            context: Value::Null,
            retained_results: vec![retained("page", "page")],
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;
    completed(&events);

    let calls = calls.lock().expect("calls");
    assert_eq!(ids(&calls[0].retained_results), ["page"]);
    for call in &calls[1..] {
        assert_eq!(ids(&call.retained_results), ["page", "prime-1"]);
    }
}
