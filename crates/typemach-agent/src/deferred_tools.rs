use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::pending_tool::PendingAuthorization;
use crate::{
    AgentError, AgentToolSpec, PendingToolCall, PermissionDecision, ToolAnnotations,
    ToolPermissionPolicy, ToolRegistry, ToolResult, ToolUse,
};

pub(crate) const SEARCH_NAME: &str = "tool_search";
const DEFAULT_LIMIT: usize = 5;
const MAX_LIMIT: usize = 20;

/// A deferred tool name persisted in the current run's discovery state.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DeferredToolName(String);

impl DeferredToolName {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub(crate) struct ToolCatalog {
    pub visible: Vec<AgentToolSpec>,
    deferred: Vec<AgentToolSpec>,
}

impl ToolCatalog {
    pub async fn read(
        registry: &impl ToolRegistry,
        context: &Value,
        loaded: &mut BTreeSet<DeferredToolName>,
    ) -> Result<Self, AgentError> {
        let direct = registry.list_tools(context).await?;
        let deferred = registry.list_deferred_tools(context).await?;
        Self::new(direct, deferred, loaded)
    }

    fn new(
        mut direct: Vec<AgentToolSpec>,
        deferred: Vec<AgentToolSpec>,
        loaded: &mut BTreeSet<DeferredToolName>,
    ) -> Result<Self, AgentError> {
        let mut names = BTreeSet::new();
        for spec in direct.iter().chain(&deferred) {
            if spec.name == SEARCH_NAME {
                return Err(AgentError::ReservedToolName(spec.name.clone()));
            }
            if !names.insert(&spec.name) {
                return Err(AgentError::DuplicateToolName(spec.name.clone()));
            }
        }
        loaded.retain(|name| deferred.iter().any(|spec| spec.name == name.as_str()));
        direct.extend(
            deferred
                .iter()
                .filter(|spec| loaded.contains(&DeferredToolName(spec.name.clone())))
                .cloned(),
        );
        if !deferred.is_empty() {
            direct.push(search_spec());
        }
        Ok(Self {
            visible: direct,
            deferred,
        })
    }

    /// Revalidate listed calls and resolve legacy calls against current catalogs.
    /// Calls already classified as unlisted cannot gain authorization here.
    pub fn refresh_pending(
        &mut self,
        pending: &mut PendingToolCall,
        loaded: &mut BTreeSet<DeferredToolName>,
    ) {
        match pending.authorization {
            PendingAuthorization::Unlisted => return,
            PendingAuthorization::Legacy => {
                if let Some(spec) = self
                    .deferred
                    .iter()
                    .find(|spec| spec.name == pending.tool_use.name)
                    && loaded.insert(DeferredToolName(spec.name.clone()))
                {
                    self.visible.push(spec.clone());
                }
            }
            PendingAuthorization::Listed(_) => {}
        }
        pending.authorization = self
            .visible
            .iter()
            .find(|spec| spec.name == pending.tool_use.name)
            .cloned()
            .map_or(PendingAuthorization::Unlisted, PendingAuthorization::Listed);
    }

    pub fn search(&self, call: &ToolUse, loaded: &mut BTreeSet<DeferredToolName>) -> ToolResult {
        match search_input(&call.input) {
            Ok(input) => {
                let names = matching_names(&self.deferred, &input.query, input.limit);
                loaded.extend(names.iter().cloned());
                ToolResult::ok(call, json!({ "loaded_tools": names }))
            }
            Err(error) => ToolResult::error(call, error.to_string()),
        }
    }
}

pub(crate) fn permission(
    policy: &impl ToolPermissionPolicy,
    pending: &PendingToolCall,
    context: &Value,
) -> PermissionDecision {
    if pending.spec().is_none() {
        return PermissionDecision::Deny("tool is not currently loaded or authorized".into());
    }
    policy.check(&pending.tool_use, pending.spec(), context)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchInput {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

const fn default_limit() -> usize {
    DEFAULT_LIMIT
}

fn search_input(value: &Value) -> Result<SearchInput, AgentError> {
    let input: SearchInput = serde_json::from_value(value.clone())
        .map_err(|error| AgentError::InvalidBuiltInTool(error.to_string()))?;
    if !input.query.chars().any(char::is_alphanumeric) || !(1..=MAX_LIMIT).contains(&input.limit) {
        return Err(AgentError::InvalidBuiltInTool(format!(
            "tool_search requires search terms and limit between 1 and {MAX_LIMIT}"
        )));
    }
    Ok(input)
}

fn matching_names(tools: &[AgentToolSpec], query: &str, limit: usize) -> Vec<DeferredToolName> {
    let query = query.to_lowercase();
    let terms: BTreeSet<_> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .collect();
    let mut matches = tools
        .iter()
        .filter_map(|tool| {
            let name = tool.name.to_lowercase();
            let description = tool.description.to_lowercase();
            let score = terms
                .iter()
                .filter(|term| name.contains(**term) || description.contains(**term))
                .count();
            (score > 0).then_some((score, &tool.name))
        })
        .collect::<Vec<_>>();
    matches.sort_by(|(a_score, a_name), (b_score, b_name)| {
        b_score.cmp(a_score).then(a_name.cmp(b_name))
    });
    matches
        .into_iter()
        .take(limit)
        .map(|(_, name)| DeferredToolName(name.clone()))
        .collect()
}

fn search_spec() -> AgentToolSpec {
    AgentToolSpec {
        name: SEARCH_NAME.into(),
        description: "Search available deferred tools by terms in their names and descriptions. Matching tools are loaded for subsequent calls in this run. Use specific terms, separated by spaces.".into(),
        input_schema: json!({
            "type": "object", "properties": {
                "query": { "type": "string", "minLength": 1 },
                "limit": { "type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "default": DEFAULT_LIMIT }
            }, "required": ["query"], "additionalProperties": false
        }),
        output_schema: json!({
            "type": "object", "properties": { "loaded_tools": { "type": "array", "items": { "type": "string" } } },
            "required": ["loaded_tools"], "additionalProperties": false
        }),
        metadata: Value::Null,
        annotations: ToolAnnotations { read_only: false, ..ToolAnnotations::default() },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, description: &str) -> AgentToolSpec {
        AgentToolSpec {
            name: name.into(),
            description: description.into(),
            ..search_spec()
        }
    }

    fn names(tools: &[AgentToolSpec], query: &str, limit: usize) -> Vec<String> {
        matching_names(tools, query, limit)
            .into_iter()
            .map(|name| name.0)
            .collect()
    }

    #[test]
    fn lexical_search_ranks_distinct_terms_and_breaks_ties_by_name() {
        let tools = [
            spec("refund_daily", "Daily refund totals"),
            spec("refund_weekly", "Weekly refund totals"),
            spec("other_weekly", "Weekly summary"),
        ];
        assert_eq!(
            names(&tools, "REFUND weekly", 3),
            ["refund_weekly", "other_weekly", "refund_daily"]
        );
        assert_eq!(names(&tools, "refund_weekly", 1), ["refund_weekly"]);
        assert_eq!(names(&tools, "weekly weekly refund", 1), ["refund_weekly"]);
        assert!(names(&tools, "missing", 5).is_empty());
    }

    #[test]
    fn chinese_description_substrings_are_searchable() {
        // Chinese strings are compatibility fixtures for multilingual registry descriptions.
        let tools = [
            spec("a", "退款总额"),
            spec("b", "每周退款趋势"),
            spec("c", "销售趋势"),
        ];
        assert_eq!(names(&tools, "退款 趋势", 3), ["b", "a", "c"]);
        assert_eq!(names(&tools, "退款趋势", 5), ["b"]);
    }

    #[test]
    fn schemas_and_metadata_are_not_searchable() {
        let mut tool = spec("sales", "Read totals");
        tool.input_schema = json!({ "refund": "weekly" });
        tool.metadata = json!({ "refund": "weekly" });
        assert!(names(&[tool], "refund weekly", 5).is_empty());
    }

    #[test]
    fn search_argument_limits_are_bounded() {
        assert_eq!(search_input(&json!({ "query": "x" })).unwrap().limit, 5);
        assert_eq!(
            search_input(&json!({ "query": "x", "limit": 20 }))
                .unwrap()
                .limit,
            20
        );
        for input in [
            json!({ "query": " " }),
            json!({ "query": "!" }),
            json!({}),
            json!({ "query": "x", "limit": 0 }),
            json!({ "query": "x", "limit": 21 }),
            json!({ "query": "x", "limit": -1 }),
            json!({ "query": "x", "limit": 1.5 }),
            json!({ "query": "x", "extra": true }),
        ] {
            assert!(matches!(
                search_input(&input),
                Err(AgentError::InvalidBuiltInTool(_))
            ));
        }
        let tools = (0..30)
            .map(|index| spec(&format!("tool_{index:02}"), "matching"))
            .collect::<Vec<_>>();
        assert_eq!(
            names(&tools, "matching", DEFAULT_LIMIT).len(),
            DEFAULT_LIMIT
        );
        assert_eq!(names(&tools, "matching", MAX_LIMIT).len(), MAX_LIMIT);
    }

    #[test]
    fn reserved_and_duplicate_catalog_names_are_typed_errors() {
        for (direct, deferred) in [(vec![search_spec()], vec![]), (vec![], vec![search_spec()])] {
            assert!(
                matches!(ToolCatalog::new(direct, deferred, &mut BTreeSet::new()),
                Err(AgentError::ReservedToolName(name)) if name == "tool_search")
            );
        }
        let tool = spec("duplicate", "Description");
        for (direct, deferred) in [
            (vec![tool.clone(), tool.clone()], vec![]),
            (vec![], vec![tool.clone(), tool.clone()]),
            (vec![tool.clone()], vec![tool.clone()]),
        ] {
            assert!(
                matches!(ToolCatalog::new(direct, deferred, &mut BTreeSet::new()),
                Err(AgentError::DuplicateToolName(name)) if name == "duplicate")
            );
        }
    }
}
