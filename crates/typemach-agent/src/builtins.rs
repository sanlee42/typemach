use serde_json::Value;

use crate::{AgentError, Artifact, AskUserQuestion, TerminalAction, ToolUse};

pub(super) fn ask_user_question(tool_use: &ToolUse) -> Result<AskUserQuestion, AgentError> {
    let question = tool_use
        .input
        .get("question")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            AgentError::InvalidBuiltInTool("ask_user requires non-empty question".to_string())
        })?
        .to_string();
    Ok(AskUserQuestion {
        tool_use_id: tool_use.id.clone(),
        question,
        fields: tool_use.input.get("fields").cloned().unwrap_or(Value::Null),
    })
}

pub(super) fn is_terminal_tool(tool_use: &ToolUse, spec: Option<&crate::AgentToolSpec>) -> bool {
    spec.is_some_and(|spec| spec.annotations.terminal)
        || matches!(
            tool_use.name.as_str(),
            "report" | "reject" | "terminal" | "planner.report" | "planner.reject"
        )
}

pub(super) fn terminal_action(tool_use: &ToolUse) -> TerminalAction {
    TerminalAction {
        tool_use_id: tool_use.id.clone(),
        name: tool_use.name.clone(),
        input: tool_use.input.clone(),
    }
}

pub(super) fn agent_builtin(tool_use: &ToolUse) -> bool {
    matches!(
        tool_use.name.as_str(),
        "ask_user" | "emit_artifact" | "tool_search"
    )
}

pub(super) fn artifact_from_tool(tool_use: &ToolUse) -> Result<Artifact, AgentError> {
    let title = required_string(&tool_use.input, "title")?;
    let content = required_string(&tool_use.input, "content")?;
    let kind = required_string(&tool_use.input, "type")?;
    if !matches!(kind.as_str(), "markdown" | "table") {
        return Err(AgentError::InvalidBuiltInTool(
            "type must be markdown or table".to_string(),
        ));
    }
    Ok(Artifact {
        tool_use_id: tool_use.id.clone(),
        title,
        kind,
        content,
        source: optional_source(&tool_use.input)?,
        window: optional_string(&tool_use.input, "window"),
        updated_at: optional_string(&tool_use.input, "updated_at"),
    })
}

fn optional_string(input: &Value, name: &str) -> Option<String> {
    input
        .get(name)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn optional_source(input: &Value) -> Result<Option<String>, AgentError> {
    let Some(source) = input.get("source") else {
        return Ok(None);
    };
    source
        .as_str()
        .map(str::trim)
        .filter(|source| !source.is_empty())
        .map(str::to_owned)
        .map(Some)
        .ok_or_else(|| {
            AgentError::InvalidBuiltInTool("source must be a non-empty string".to_string())
        })
}

fn required_string(input: &Value, name: &str) -> Result<String, AgentError> {
    input
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| AgentError::InvalidBuiltInTool(format!("missing non-empty {name}")))
}
