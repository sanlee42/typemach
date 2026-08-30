use super::*;

#[derive(Clone, Default)]
struct FailFirstFinalModel {
    calls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<ModelRequest>>>,
}

#[async_trait]
impl AgentModel for FailFirstFinalModel {
    async fn next_step(
        &self,
        request: ModelRequest,
        _stream: ModelStream,
    ) -> Result<ModelResponse, AgentError> {
        self.requests.lock().expect("requests lock").push(request);
        match self.calls.fetch_add(1, Ordering::Relaxed) {
            0 => Ok(ModelResponse {
                outcome: Some(ModelOutcome::FinalAnswer {
                    text: "Planning is complete.".to_string(),
                }),
                stop_reason: Some(StopReason::EndTurn),
                ..ModelResponse::default()
            }),
            1 => Err(AgentError::Model("final transport failed".to_string())),
            _ => Ok(ModelResponse {
                outcome: Some(ModelOutcome::FinalAnswer {
                    text: "Recovered final answer.".to_string(),
                }),
                stop_reason: Some(StopReason::EndTurn),
                ..ModelResponse::default()
            }),
        }
    }
}

#[tokio::test]
async fn retrying_the_same_run_resumes_final_without_replaying_planning() {
    let model = FailFirstFinalModel::default();
    let runner = build_agent_runner(
        MemorySaver::default(),
        model.clone(),
        FakeTools,
        AllowAllTools,
    );
    let run = request(AgentRunInput {
        messages: vec![AgentMessage::user_text("Answer directly")],
        context: Value::Null,
        budget: AgentBudget {
            max_model_turns: 4,
            max_tool_calls: 4,
        },
        human_input: None,
        system_suffix: None,
    });
    let first = collect(runner.stream(run.clone(), StreamConfig::default())).await;
    assert!(
        first
            .iter()
            .any(|event| matches!(event, RunStreamEvent::Failed { .. }))
    );
    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    assert_eq!(
        checkpoint.next_step,
        Some(serde_json::to_value(AgentStep::FinalAnswer).expect("serialize step"))
    );

    let second = collect(runner.stream(run, StreamConfig::default())).await;
    assert_eq!(completed(&second).answer, "Recovered final answer.");
    let requests = model.requests.lock().expect("requests lock");
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests[0].tool_choice,
        Some(typemach_agent::ToolChoice::Auto)
    );
    assert!(!requests[0].tools.is_empty());
    for request in &requests[1..] {
        assert_eq!(request.tool_choice, Some(typemach_agent::ToolChoice::None));
        assert!(request.tools.is_empty());
        let prompt = serde_json::to_string(&request.messages).expect("serialize prompt");
        assert!(!prompt.contains("Planning is complete."));
    }
}
