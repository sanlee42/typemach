use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    AgentError, AgentMessage, ContentBlock, ContextCompaction, ContextPolicy, ConversationDigest,
    PromptEstimate, ToolResult, ToolResultArchive, TranscriptArchive,
};

#[derive(Debug, Clone, PartialEq)]
pub struct PromptWindow {
    pub messages: Vec<AgentMessage>,
    pub digest: Option<ConversationDigest>,
    pub compaction: Option<ContextCompaction>,
}

pub fn prompt_window(
    messages: &[AgentMessage],
    policy: &ContextPolicy,
) -> Result<PromptWindow, AgentError> {
    let before = estimate_messages(messages)?;
    if before.estimated_tokens < policy.compact_at_tokens
        || messages.len() <= policy.recent_messages
    {
        return Ok(PromptWindow {
            messages: messages.to_vec(),
            digest: None,
            compaction: None,
        });
    }

    let mut keep_count = policy.recent_messages.max(1).min(messages.len());
    let mut window = build_compacted_window(messages, keep_count)?;
    while window.after.estimated_tokens > policy.max_input_tokens && keep_count > 1 {
        keep_count -= 1;
        window = build_compacted_window(messages, keep_count)?;
    }

    Ok(PromptWindow {
        messages: window.messages,
        digest: Some(window.digest),
        compaction: Some(ContextCompaction {
            before,
            after: window.after,
            archive: window.archive,
        }),
    })
}

pub fn maybe_archive_tool_result(
    result: &ToolResult,
    policy: &ContextPolicy,
) -> Result<(ToolResult, Option<ToolResultArchive>), AgentError> {
    let bytes = serde_json::to_vec(&result.content)
        .map_err(|err| AgentError::Tool(format!("failed to encode tool result: {err}")))?;
    if bytes.len() <= policy.max_tool_result_bytes {
        return Ok((result.clone(), None));
    }

    let archive = ToolResultArchive {
        tool_use_id: result.tool_use_id.clone(),
        name: result.name.clone(),
        byte_count: bytes.len(),
        sha256: sha256_hex(&bytes),
    };
    let archived = ToolResult {
        tool_use_id: result.tool_use_id.clone(),
        name: result.name.clone(),
        content: archived_tool_content(&archive),
        is_error: result.is_error,
        raw: None,
    };
    Ok((archived, Some(archive)))
}

pub fn estimate_messages(messages: &[AgentMessage]) -> Result<PromptEstimate, AgentError> {
    let bytes = serde_json::to_vec(messages)
        .map_err(|err| AgentError::Model(format!("failed to estimate prompt size: {err}")))?;
    Ok(PromptEstimate {
        message_count: messages.len(),
        estimated_tokens: estimate_tokens(bytes.len()),
    })
}

struct CompactedWindow {
    messages: Vec<AgentMessage>,
    digest: ConversationDigest,
    after: PromptEstimate,
    archive: TranscriptArchive,
}

fn build_compacted_window(
    messages: &[AgentMessage],
    keep_count: usize,
) -> Result<CompactedWindow, AgentError> {
    let mut omitted_count = messages.len().saturating_sub(keep_count);
    // Chat completion APIs reject a `role:"tool"` message whose tool call
    // lives in the omitted prefix, so widen the window until it no longer
    // starts on a tool result. Pairing correctness wins over the byte budget.
    while omitted_count > 0 && unsafe_window_start(&messages[omitted_count]) {
        omitted_count -= 1;
    }
    let omitted = &messages[..omitted_count];
    let kept = &messages[omitted_count..];
    let digest = ConversationDigest::compacted_window(omitted_count);
    let mut compacted_messages = Vec::with_capacity(kept.len() + 1);
    compacted_messages.push(AgentMessage::User {
        content: vec![ContentBlock::ConversationDigest(digest.clone())],
    });
    compacted_messages.extend_from_slice(kept);
    let omitted_bytes = serde_json::to_vec(omitted)
        .map_err(|err| AgentError::Model(format!("failed to archive transcript: {err}")))?;
    let after = estimate_messages(&compacted_messages)?;
    Ok(CompactedWindow {
        messages: compacted_messages,
        digest,
        after,
        archive: TranscriptArchive {
            message_count: omitted_count,
            byte_count: omitted_bytes.len(),
            estimated_tokens: estimate_tokens(omitted_bytes.len()),
            sha256: sha256_hex(&omitted_bytes),
        },
    })
}

fn unsafe_window_start(message: &AgentMessage) -> bool {
    match message {
        AgentMessage::User { content } => content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolResult(_))),
        AgentMessage::Assistant { .. } => false,
    }
}

fn archived_tool_content(archive: &ToolResultArchive) -> Value {
    json!({
        "archived": true,
        "tool_use_id": archive.tool_use_id,
        "name": archive.name,
        "byte_count": archive.byte_count,
        "sha256": archive.sha256,
        "reason": "tool result exceeded the active prompt byte budget; use product evidence storage for the full result"
    })
}

fn estimate_tokens(byte_count: usize) -> u64 {
    (byte_count as u64 / 4).saturating_add(1)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolUse;

    fn tool_use_message(id: &str) -> AgentMessage {
        AgentMessage::Assistant {
            content: vec![ContentBlock::ToolUse(ToolUse {
                id: id.to_string(),
                name: "metric_point".to_string(),
                input: json!({}),
                raw: None,
            })],
        }
    }

    fn tool_result_message(id: &str) -> AgentMessage {
        AgentMessage::tool_result(ToolResult {
            tool_use_id: id.to_string(),
            name: "metric_point".to_string(),
            content: json!({ "value": 42 }),
            is_error: false,
            raw: None,
        })
    }

    fn policy(recent_messages: usize, max_input_tokens: u64) -> ContextPolicy {
        ContextPolicy {
            max_input_tokens,
            compact_at_tokens: 1,
            recent_messages,
            ..ContextPolicy::default()
        }
    }

    fn assert_tool_pairing(messages: &[AgentMessage]) {
        let mut seen = std::collections::HashSet::new();
        for message in messages {
            let (AgentMessage::User { content } | AgentMessage::Assistant { content }) = message;
            for block in content {
                match block {
                    ContentBlock::ToolUse(tool_use) => {
                        seen.insert(tool_use.id.clone());
                    }
                    ContentBlock::ToolResult(result) => {
                        assert!(
                            seen.contains(&result.tool_use_id),
                            "tool result {} appears before its tool use",
                            result.tool_use_id
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn compaction_never_starts_window_with_tool_result() {
        let messages = vec![
            AgentMessage::user_text("q1"),
            tool_use_message("tool-1"),
            tool_result_message("tool-1"),
            AgentMessage::assistant_text("a1"),
            AgentMessage::user_text("q2"),
            AgentMessage::assistant_text("a2"),
            AgentMessage::user_text("q3"),
        ];
        let window = prompt_window(&messages, &policy(5, u64::MAX)).expect("window");
        let compaction = window.compaction.expect("compaction");
        assert!(matches!(
            window.messages.first(),
            Some(AgentMessage::User { content })
                if matches!(content.first(), Some(ContentBlock::ConversationDigest(_)))
        ));
        assert_eq!(window.messages.get(1), Some(&tool_use_message("tool-1")));
        assert_eq!(compaction.archive.message_count, 1);
        assert_tool_pairing(&window.messages);
    }

    #[test]
    fn compaction_keeps_trailing_tool_group_intact() {
        let messages = vec![
            AgentMessage::user_text("q1"),
            AgentMessage::assistant_text("a1"),
            AgentMessage::user_text("q2"),
            AgentMessage::Assistant {
                content: vec![
                    ContentBlock::ToolUse(ToolUse {
                        id: "tool-1".to_string(),
                        name: "metric_point".to_string(),
                        input: json!({}),
                        raw: None,
                    }),
                    ContentBlock::ToolUse(ToolUse {
                        id: "tool-2".to_string(),
                        name: "metric_point".to_string(),
                        input: json!({}),
                        raw: None,
                    }),
                ],
            },
            tool_result_message("tool-1"),
            tool_result_message("tool-2"),
        ];
        let window = prompt_window(&messages, &policy(1, u64::MAX)).expect("window");
        let compaction = window.compaction.expect("compaction");
        assert_eq!(compaction.archive.message_count, 3);
        assert_eq!(window.messages.len(), 4);
        assert_tool_pairing(&window.messages);
    }

    #[test]
    fn compaction_shrink_loop_stays_safe_under_tiny_budget() {
        let mut messages = vec![AgentMessage::user_text("q1")];
        for index in 0..6 {
            messages.push(tool_use_message(&format!("tool-{index}")));
            messages.push(tool_result_message(&format!("tool-{index}")));
        }
        let window = prompt_window(&messages, &policy(8, 1)).expect("window");
        assert!(window.compaction.is_some());
        assert!(!unsafe_window_start(&window.messages[1]));
        assert_tool_pairing(&window.messages);
    }

    #[test]
    fn mixed_user_message_with_tool_result_is_unsafe_boundary() {
        let mixed = AgentMessage::User {
            content: vec![
                ContentBlock::ToolResult(ToolResult {
                    tool_use_id: "tool-1".to_string(),
                    name: "metric_point".to_string(),
                    content: json!({}),
                    is_error: false,
                    raw: None,
                }),
                ContentBlock::Text {
                    text: "extra".to_string(),
                },
            ],
        };
        assert!(unsafe_window_start(&mixed));
        assert!(!unsafe_window_start(&AgentMessage::user_text("plain")));
        assert!(!unsafe_window_start(&AgentMessage::assistant_text("a")));
    }
}
