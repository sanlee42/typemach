use serde::{Deserialize, Serialize};

use crate::{AgentToolSpec, ToolUse};

#[derive(Debug, Clone, PartialEq)]
pub struct PendingToolCall {
    pub tool_use: ToolUse,
    pub(crate) authorization: PendingAuthorization,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PendingAuthorization {
    Legacy,
    Unlisted,
    Listed(AgentToolSpec),
}

impl PendingToolCall {
    pub fn new(tool_use: ToolUse, spec: Option<AgentToolSpec>) -> Self {
        Self {
            tool_use,
            authorization: spec
                .map_or(PendingAuthorization::Unlisted, PendingAuthorization::Listed),
        }
    }

    /// The spec authorized when this call was issued or revalidated from a legacy checkpoint.
    pub fn spec(&self) -> Option<&AgentToolSpec> {
        match &self.authorization {
            PendingAuthorization::Listed(spec) => Some(spec),
            PendingAuthorization::Legacy | PendingAuthorization::Unlisted => None,
        }
    }
}

impl Serialize for PendingToolCall {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.authorization == PendingAuthorization::Legacy {
            return self.tool_use.serialize(serializer);
        }
        #[derive(Serialize)]
        struct Current<'a> {
            tool_use: &'a ToolUse,
            #[serde(skip_serializing_if = "Option::is_none")]
            spec: Option<&'a AgentToolSpec>,
        }
        Current {
            tool_use: &self.tool_use,
            spec: self.spec(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PendingToolCall {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Shape {
            Current {
                tool_use: ToolUse,
                #[serde(default)]
                spec: Option<AgentToolSpec>,
            },
            Legacy(ToolUse),
        }
        match Shape::deserialize(deserializer)? {
            Shape::Current { tool_use, spec } => Ok(Self::new(tool_use, spec)),
            Shape::Legacy(tool_use) => Ok(Self {
                tool_use,
                authorization: PendingAuthorization::Legacy,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn checkpoint_round_trip_distinguishes_legacy_from_current_unlisted() {
        let tool = json!({ "id": "call-1", "name": "read_totals", "input": {} });
        let legacy: PendingToolCall = serde_json::from_value(tool.clone()).unwrap();
        let current: PendingToolCall = serde_json::from_value(json!({ "tool_use": tool })).unwrap();
        assert!(matches!(legacy.authorization, PendingAuthorization::Legacy));
        assert!(matches!(
            current.authorization,
            PendingAuthorization::Unlisted
        ));
        for pending in [legacy, current] {
            let decoded: PendingToolCall =
                serde_json::from_value(serde_json::to_value(&pending).unwrap()).unwrap();
            assert_eq!(decoded, pending);
        }
    }
}
