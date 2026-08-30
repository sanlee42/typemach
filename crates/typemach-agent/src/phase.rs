use std::collections::HashSet;

use crate::{AgentMessage, ContentBlock};

const FINAL_ANSWER_PROMPT: &str = "[Final answer phase]\nProvide the final answer now. Use only the canonical conversation and successful tool evidence below. Do not call tools or describe internal planning.";

pub(super) fn final_system_suffix(suffix: Option<&str>) -> String {
    match suffix.map(str::trim).filter(|value| !value.is_empty()) {
        Some(suffix) => format!("{suffix}\n\n{FINAL_ANSWER_PROMPT}"),
        None => FINAL_ANSWER_PROMPT.to_string(),
    }
}

pub(super) fn final_messages(messages: &[AgentMessage]) -> Vec<AgentMessage> {
    let pairs = successful_pairs(messages);
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| project_message(index, message, &pairs))
        .collect()
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Position {
    message: usize,
    block: usize,
}

struct SuccessfulPairs {
    calls: HashSet<Position>,
    results: HashSet<Position>,
}

fn successful_pairs(messages: &[AgentMessage]) -> SuccessfulPairs {
    let mut pairs = SuccessfulPairs {
        calls: HashSet::new(),
        results: HashSet::new(),
    };
    for (message_index, message) in messages.iter().enumerate() {
        let AgentMessage::Assistant { content } = message else {
            continue;
        };
        let results = following_results(messages, message_index + 1);
        let mut used = HashSet::new();
        for (block_index, block) in content.iter().enumerate() {
            let ContentBlock::ToolUse(call) = block else {
                continue;
            };
            let Some((position, result)) = results.iter().find(|(position, result)| {
                !used.contains(position) && result.tool_use_id == call.id
            }) else {
                continue;
            };
            used.insert(*position);
            if !result.is_error {
                pairs.calls.insert(Position {
                    message: message_index,
                    block: block_index,
                });
                pairs.results.insert(*position);
            }
        }
    }
    pairs
}

fn following_results(
    messages: &[AgentMessage],
    start: usize,
) -> Vec<(Position, &crate::ToolResult)> {
    let mut results = Vec::new();
    for (message_index, message) in messages.iter().enumerate().skip(start) {
        let AgentMessage::User { content } = message else {
            break;
        };
        if content
            .iter()
            .any(|block| !matches!(block, ContentBlock::ToolResult(_)))
        {
            break;
        }
        for (block_index, block) in content.iter().enumerate() {
            if let ContentBlock::ToolResult(result) = block {
                results.push((
                    Position {
                        message: message_index,
                        block: block_index,
                    },
                    result,
                ));
            }
        }
    }
    results
}

fn project_message(
    message_index: usize,
    message: &AgentMessage,
    pairs: &SuccessfulPairs,
) -> Option<AgentMessage> {
    let (content, assistant) = match message {
        AgentMessage::User { content } => (content, false),
        AgentMessage::Assistant { content } => (content, true),
    };
    let commentary = assistant
        && content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse(_)));
    let content = content
        .iter()
        .enumerate()
        .filter(|(block_index, block)| {
            keep_block(
                Position {
                    message: message_index,
                    block: *block_index,
                },
                block,
                pairs,
                commentary,
            )
        })
        .map(|(_, block)| block)
        .cloned()
        .collect::<Vec<_>>();
    if content.is_empty() {
        return None;
    }
    Some(if assistant {
        AgentMessage::Assistant { content }
    } else {
        AgentMessage::User { content }
    })
}

fn keep_block(
    position: Position,
    block: &ContentBlock,
    pairs: &SuccessfulPairs,
    commentary: bool,
) -> bool {
    match block {
        ContentBlock::Text { .. } => !commentary,
        ContentBlock::ConversationDigest(_) => true,
        ContentBlock::Thinking { .. } => false,
        ContentBlock::ToolUse(_) => pairs.calls.contains(&position),
        ContentBlock::ToolResult(_) => pairs.results.contains(&position),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolResult, ToolUse};
    use serde_json::json;

    #[test]
    fn repeated_provider_id_is_paired_within_each_transcript_segment() {
        let first = ToolUse {
            id: "reused-id".to_string(),
            name: "metric_point".to_string(),
            input: json!({ "metric": "paid_orders" }),
            raw: None,
        };
        let second = ToolUse {
            id: "reused-id".to_string(),
            name: "metric_point".to_string(),
            input: json!({ "metric": "SECRET_LATER_CALL" }),
            raw: None,
        };
        let messages = vec![
            AgentMessage::user_text("Compare paid orders"),
            AgentMessage::Assistant {
                content: vec![
                    ContentBlock::Text {
                        text: "First public update. ".to_string(),
                    },
                    ContentBlock::ToolUse(first.clone()),
                ],
            },
            AgentMessage::tool_result(ToolResult::ok(&first, json!({ "value": 42 }))),
            AgentMessage::Assistant {
                content: vec![
                    ContentBlock::Text {
                        text: "Later public update. ".to_string(),
                    },
                    ContentBlock::ToolUse(second.clone()),
                ],
            },
            AgentMessage::tool_result(ToolResult::error(&second, "SECRET_FAILED_PAYLOAD")),
        ];

        let projected = final_messages(&messages);
        let encoded = serde_json::to_string(&projected).expect("serialize projection");
        assert!(!encoded.contains("First public update."));
        assert!(!encoded.contains("Later public update."));
        assert!(!encoded.contains("SECRET_LATER_CALL"));
        assert!(!encoded.contains("SECRET_FAILED_PAYLOAD"));
        assert_eq!(projected.len(), 3);
        assert!(matches!(
            &projected[1],
            AgentMessage::Assistant { content }
                if matches!(
                    content.as_slice(),
                    [ContentBlock::ToolUse(call)]
                        if call.input["metric"] == "paid_orders"
                )
        ));
        assert!(matches!(
            &projected[2],
            AgentMessage::User { content }
                if matches!(
                    content.as_slice(),
                    [ContentBlock::ToolResult(result)] if result.content["value"] == 42
                )
        ));
    }
}
