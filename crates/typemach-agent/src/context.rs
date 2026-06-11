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
    if before.estimated_tokens < policy.compact_at_tokens || messages.len() <= policy.recent_turns {
        return Ok(PromptWindow {
            messages: messages.to_vec(),
            digest: None,
            compaction: None,
        });
    }

    let mut keep_count = policy.recent_turns.max(1).min(messages.len());
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
    let omitted_count = messages.len().saturating_sub(keep_count);
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
