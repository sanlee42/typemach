use super::*;

#[derive(Clone, Default)]
struct Catalog {
    calls: Arc<Mutex<Vec<String>>>,
}

fn spec(name: &str, description: &str) -> AgentToolSpec {
    AgentToolSpec {
        name: name.into(),
        description: description.into(),
        input_schema: json!({ "type": "object", "properties": { "secret_schema": { "type": "string" } } }),
        output_schema: Value::Null,
        metadata: Value::Null,
        annotations: ToolAnnotations::default(),
    }
}

#[async_trait]
impl ToolRegistry for Catalog {
    async fn list_tools(&self, _: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        Ok(vec![spec("ask_user", "Ask a question")])
    }

    async fn list_deferred_tools(&self, context: &Value) -> Result<Vec<AgentToolSpec>, AgentError> {
        if context == &json!("revoked") {
            return Ok(Vec::new());
        }
        Ok(vec![
            spec("weekly_refunds", "Weekly refund totals"),
            spec("daily_sales", "Daily sales totals"),
        ])
    }

    async fn call_tool(&self, request: ToolCallRequest) -> Result<ToolResult, AgentError> {
        assert_ne!(request.tool_use.name, "tool_search");
        self.calls
            .lock()
            .unwrap()
            .push(request.tool_use.name.clone());
        Ok(ToolResult::ok(&request.tool_use, json!({ "ok": true })))
    }
}

fn call(id: &str, name: &str, input: Value) -> ToolUse {
    ToolUse {
        id: id.into(),
        name: name.into(),
        input,
        raw: None,
    }
}

fn search(id: &str, query: &str) -> ModelResponse {
    tool_response("", vec![call(id, "tool_search", json!({ "query": query }))])
}

fn input() -> AgentRunInput {
    AgentRunInput {
        messages: vec![AgentMessage::user_text("Find evidence")],
        context: Value::Null,
        budget: AgentBudget::default(),
        human_input: None,
        system_suffix: None,
    }
}

fn names(request: &ModelRequest) -> Vec<&str> {
    request
        .tools
        .iter()
        .map(|spec| spec.name.as_str())
        .collect()
}

#[tokio::test]
async fn default_registry_keeps_all_direct_tools_without_search() {
    let model = ScriptedModel::new([final_response("Done")]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
        AllowAllTools,
    );
    completed(&collect(runner.stream(request(input()), StreamConfig::default())).await);
    assert_eq!(
        model.requests()[0].tools,
        FakeTools.list_tools(&Value::Null).await.unwrap()
    );
}

#[tokio::test]
async fn discovery_unions_names_and_denies_calls_issued_before_loading() {
    let model = ScriptedModel::new([
        tool_response(
            "",
            vec![
                call(
                    "search-1",
                    "tool_search",
                    json!({ "query": "refund weekly" }),
                ),
                call("too-early", "weekly_refunds", json!({})),
            ],
        ),
        search("search-2", "daily sales"),
        tool_response("", vec![call("loaded", "weekly_refunds", json!({}))]),
        final_response("Done"),
    ]);
    let registry = Catalog::default();
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        registry.clone(),
        AllowAllTools,
    );
    let events = collect(runner.stream(request(input()), StreamConfig::default())).await;
    completed(&events);
    let requests = model.requests();
    assert_eq!(names(&requests[0]), ["ask_user", "tool_search"]);
    assert_eq!(
        names(&requests[1]),
        ["ask_user", "weekly_refunds", "tool_search"]
    );
    assert_eq!(
        names(&requests[2]),
        ["ask_user", "weekly_refunds", "daily_sales", "tool_search"]
    );
    assert_eq!(*registry.calls.lock().unwrap(), ["weekly_refunds"]);
    assert!(events.iter().any(|event| matches!(event,
        RunStreamEvent::Signal { signal: AgentSignal::ToolResult { tool_use_id, is_error: true, .. } }
            if tool_use_id == "too-early"
    )));
    let results = events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal: AgentSignal::ToolResult { name, content, .. },
            } if name == "tool_search" => Some(content.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        results,
        [
            json!({ "loaded_tools": ["weekly_refunds"] }),
            json!({ "loaded_tools": ["daily_sales"] })
        ]
    );
}

async fn interrupted_run() -> (typemach::CheckpointRecord, ScriptedModel) {
    let model = ScriptedModel::new([
        search("search-1", "refund"),
        tool_response(
            "",
            vec![
                call("ask-1", "ask_user", json!({ "question": "Continue?" })),
                call("pending-1", "weekly_refunds", json!({})),
            ],
        ),
    ]);
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        Catalog::default(),
        AllowAllTools,
    );
    let events = collect(runner.stream(request(input()), StreamConfig::default())).await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, RunStreamEvent::Interrupted { .. }))
    );
    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .unwrap()
        .unwrap();
    let state: AgentState = serde_json::from_value(checkpoint.state.clone()).unwrap();
    assert_eq!(
        state
            .loaded_deferred_tools
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>(),
        ["weekly_refunds"]
    );
    (checkpoint, model)
}

async fn resume_checkpoint(revoked: bool) {
    let (checkpoint, _) = interrupted_run().await;
    let saver = MemorySaver::default();
    saver.save("thread-1", &checkpoint).await.unwrap();
    let model = ScriptedModel::new([final_response("Resumed"), final_response("Fresh turn")]);
    let registry = Catalog::default();
    let runner = build_agent_runner(saver, model.clone(), registry.clone(), AllowAllTools);
    let mut resume = request(input());
    resume.command = RunCommand::Resume;
    resume.input.messages.clear();
    resume.input.human_input = Some(HumanInputAnswer {
        tool_use_id: "ask-1".into(),
        answer: "Yes".into(),
    });
    if revoked {
        resume.input.context = json!("revoked");
    }
    completed(&collect(runner.stream(resume, StreamConfig::default())).await);
    let state: AgentState = serde_json::from_value(
        runner
            .checkpointer()
            .load("thread-1")
            .await
            .unwrap()
            .unwrap()
            .state,
    )
    .unwrap();
    assert_eq!(state.loaded_deferred_tools.len(), usize::from(!revoked));
    assert_eq!(registry.calls.lock().unwrap().len(), usize::from(!revoked));
    let expected = if revoked {
        vec!["ask_user"]
    } else {
        vec!["ask_user", "weekly_refunds", "tool_search"]
    };
    assert_eq!(names(&model.requests()[0]), expected);
    let mut fresh = request(input());
    fresh.run_id = RunId::from("run-2");
    completed(&collect(runner.stream(fresh, StreamConfig::default())).await);
    assert_eq!(names(&model.requests()[1]), ["ask_user", "tool_search"]);
}

#[tokio::test]
async fn checkpoint_restart_preserves_loaded_tools_and_next_turn_resets() {
    resume_checkpoint(false).await;
}

#[tokio::test]
async fn changed_resume_context_prunes_loaded_and_pending_tools() {
    resume_checkpoint(true).await;
}
