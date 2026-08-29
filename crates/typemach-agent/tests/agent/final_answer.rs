use super::*;
use typemach::CheckpointSaver;
use typemach_agent::AgentState;

#[tokio::test]
async fn answer_is_only_the_current_turn_final_model_step() {
    let model = ScriptedModel::new([
        ModelResponse {
            final_text: Some("I will inspect the metric first.".to_string()),
            tool_uses: vec![ToolUse {
                id: "tool-1".to_string(),
                name: "metric_point".to_string(),
                input: json!({ "metric_id": "paid_order_count", "ds": "2026-06-08" }),
                raw: None,
            }],
            stop_reason: Some(StopReason::ToolUse),
            ..ModelResponse::default()
        },
        ModelResponse {
            deltas: vec!["Order count is 42.".to_string()],
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
            messages: vec![
                AgentMessage::user_text("Previous question"),
                AgentMessage::assistant_text("Old answer must not enter the current answer."),
                AgentMessage::user_text("Yesterday's order count"),
            ],
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
        RunStreamEvent::Signal {
            signal: AgentSignal::ToolResult { tool_use_id, .. },
        } if tool_use_id == "tool-1"
    )));
    assert_eq!(model.requests().len(), 2);
    assert_eq!(completed(&events).answer, "Order count is 42.");
}

#[tokio::test]
async fn terminal_tool_preserves_draft_state_without_an_extra_message_delta() {
    let model = ScriptedModel::new([ModelResponse {
        deltas: vec!["Draft before terminal action.".to_string()],
        tool_uses: vec![ToolUse {
            id: "term-1".to_string(),
            name: "report".to_string(),
            input: json!({ "message": "Terminal message must not be appended." }),
            raw: None,
        }],
        stop_reason: Some(StopReason::ToolUse),
        ..ModelResponse::default()
    }]);
    let runner = build_agent_runner(MemorySaver::default(), model, FakeTools, AllowAllTools);
    let events = collect(runner.stream(
        request(AgentRunInput {
            messages: vec![AgentMessage::user_text("Create a report")],
            context: Value::Null,
            budget: AgentBudget::default(),
            human_input: None,
            system_suffix: None,
        }),
        StreamConfig::default(),
    ))
    .await;

    let deltas = events
        .iter()
        .filter_map(|event| match event {
            RunStreamEvent::Signal {
                signal: AgentSignal::AssistantDelta { delta, .. },
            } => Some(delta.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(deltas, ["Draft before terminal action."]);
    assert!(completed(&events).answer.is_empty());

    let checkpoint = runner
        .checkpointer()
        .load("thread-1")
        .await
        .expect("load checkpoint")
        .expect("checkpoint");
    let state: AgentState = serde_json::from_value(checkpoint.state).expect("agent state");
    assert_eq!(state.answer, "Draft before terminal action.");
}
