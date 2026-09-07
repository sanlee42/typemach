use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::AgentError;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String")]
pub struct ResultId(String);

impl ResultId {
    pub fn new(value: impl Into<String>) -> Result<Self, AgentError> {
        Self::try_from(value.into()).map_err(AgentError::InvalidToolResult)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ResultId {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.trim().is_empty() {
            return Err("retained result id must not be empty".to_string());
        }
        Ok(Self(value))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetainedResult {
    id: ResultId,
    value: Value,
    #[serde(default)]
    authorization: Value,
}

impl RetainedResult {
    pub fn new(id: ResultId, value: Value, authorization: Value) -> Self {
        Self {
            id,
            value,
            authorization,
        }
    }

    pub fn id(&self) -> &ResultId {
        &self.id
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn authorization(&self) -> &Value {
        &self.authorization
    }
}

pub(crate) fn validate(results: &[RetainedResult]) -> Result<(), AgentError> {
    let mut ids = BTreeSet::new();
    for result in results {
        if !ids.insert(result.id()) {
            return Err(duplicate_id(result.id()));
        }
    }
    Ok(())
}

pub(crate) fn push(
    results: &mut Vec<RetainedResult>,
    result: RetainedResult,
) -> Result<(), AgentError> {
    if results.iter().any(|existing| existing.id() == result.id()) {
        return Err(duplicate_id(result.id()));
    }
    results.push(result);
    Ok(())
}

fn duplicate_id(id: &ResultId) -> AgentError {
    AgentError::InvalidToolResult(format!("duplicate retained result id: {}", id.as_str()))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::{ToolResult, ToolUse};

    #[test]
    fn handoff_rejects_invalid_ids_and_errors() {
        assert!(ResultId::new(" ").is_err());
        assert!(serde_json::from_value::<ResultId>(json!("")).is_err());
        let valid = ResultId::new("result-1").expect("valid id");
        assert_eq!(
            serde_json::from_value::<ResultId>(
                serde_json::to_value(&valid).expect("serialize result id")
            )
            .expect("deserialize result id"),
            valid
        );

        let tool_use = ToolUse {
            id: "tool-1".to_string(),
            name: "metric_point".to_string(),
            input: Value::Null,
            raw: None,
        };
        let retained = RetainedResult::new(
            ResultId::new("other-tool").expect("valid id"),
            json!({ "value": 42 }),
            Value::Null,
        );
        let mut wrong_id = ToolResult::ok(&tool_use, Value::Null);
        wrong_id.retained = Some(retained.clone());
        assert!(matches!(
            wrong_id.validate(),
            Err(AgentError::InvalidToolResult(_))
        ));

        let mut error = ToolResult::error(&tool_use, "failed");
        error.retained = Some(retained);
        assert!(matches!(
            error.validate(),
            Err(AgentError::InvalidToolResult(_))
        ));
    }
}
