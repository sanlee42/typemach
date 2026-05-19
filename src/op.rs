use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ops::Deref;

use crate::error::MachineError;
use crate::run::{RunId, SessionId, ThreadId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Vis {
    Public,
    Internal,
}

impl Vis {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Internal => "internal",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "public" => Some(Self::Public),
            "internal" => Some(Self::Internal),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectStatus {
    Reserved,
    Started,
    Done,
    Failed,
    Unknown,
}

impl EffectStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Reserved => "reserved",
            Self::Started => "started",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Unknown => "unknown",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "reserved" => Some(Self::Reserved),
            "started" => Some(Self::Started),
            "done" => Some(Self::Done),
            "failed" => Some(Self::Failed),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }

    pub fn is_blocking(&self) -> bool {
        matches!(self, Self::Started | Self::Unknown)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Effect {
    pub run_id: RunId,
    pub key: String,
    pub kind: String,
    pub status: EffectStatus,
    pub request: Value,
    pub result: Option<Value>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectUpdate {
    pub key: String,
    pub status: EffectStatus,
    pub result: Option<Value>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

impl EffectUpdate {
    pub fn done(key: impl Into<String>, result: Value) -> Self {
        Self {
            key: key.into(),
            status: EffectStatus::Done,
            result: Some(result),
            error_code: None,
            error_message: None,
        }
    }

    pub fn failed(
        key: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            key: key.into(),
            status: EffectStatus::Failed,
            result: None,
            error_code: Some(code.into()),
            error_message: Some(message.into()),
        }
    }

    pub fn unknown(key: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            status: EffectStatus::Unknown,
            result: None,
            error_code: None,
            error_message: Some(message.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub run_id: RunId,
    pub key: String,
    pub kind: String,
    pub body: Value,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItemWrite {
    pub key: String,
    pub kind: String,
    pub body: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub thread_id: ThreadId,
    pub seq: i64,
    pub key: String,
    pub kind: String,
    pub vis: Vis,
    pub body: Value,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntryWrite {
    pub key: String,
    pub kind: String,
    pub vis: Vis,
    pub body: Value,
}

#[derive(Debug, Clone, Copy)]
pub struct EntryQuery<'a, Scope: ?Sized = Value> {
    pub scope: &'a Scope,
    pub session_id: &'a SessionId,
    pub thread_id: Option<&'a ThreadId>,
    pub kind: Option<&'a str>,
    pub vis: Option<Vis>,
    pub after_seq: i64,
    pub limit: usize,
}

impl<'a, Scope: ?Sized> EntryQuery<'a, Scope> {
    pub fn new(scope: &'a Scope, session_id: &'a SessionId, limit: usize) -> Self {
        Self {
            scope,
            session_id,
            thread_id: None,
            kind: None,
            vis: None,
            after_seq: 0,
            limit,
        }
    }

    pub fn thread(mut self, thread_id: &'a ThreadId) -> Self {
        self.thread_id = Some(thread_id);
        self
    }

    pub fn kind(mut self, kind: &'a str) -> Self {
        self.kind = Some(kind);
        self
    }

    pub fn vis(mut self, vis: Vis) -> Self {
        self.vis = Some(vis);
        self
    }

    pub fn after(mut self, seq: i64) -> Self {
        self.after_seq = seq;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page<T> {
    pub items: Vec<T>,
    pub next: Option<i64>,
}

impl<T> Page<T> {
    pub fn new(items: Vec<T>, next: Option<i64>) -> Self {
        Self { items, next }
    }

    pub fn empty() -> Self {
        Self {
            items: Vec::new(),
            next: None,
        }
    }
}

impl<T> Deref for Page<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

#[async_trait]
pub trait RunOps: Send + Sync {
    async fn reserve(
        &self,
        run_id: &RunId,
        key: &str,
        kind: &str,
        request: Value,
    ) -> Result<Effect, MachineError>;

    async fn start(&self, run_id: &RunId, key: &str) -> Result<Effect, MachineError>;

    async fn push_effect(&self, run_id: &RunId, update: EffectUpdate) -> Result<(), MachineError>;

    async fn push_item(&self, run_id: &RunId, item: ItemWrite) -> Result<(), MachineError>;

    async fn push_entry(&self, run_id: &RunId, entry: EntryWrite) -> Result<(), MachineError>;
}

#[derive(Debug)]
pub struct NoopRunOps;

#[async_trait]
impl RunOps for NoopRunOps {
    async fn reserve(
        &self,
        _run_id: &RunId,
        _key: &str,
        _kind: &str,
        _request: Value,
    ) -> Result<Effect, MachineError> {
        Err(MachineError::RuntimeOpUnavailable)
    }

    async fn start(&self, _run_id: &RunId, _key: &str) -> Result<Effect, MachineError> {
        Err(MachineError::RuntimeOpUnavailable)
    }

    async fn push_effect(
        &self,
        _run_id: &RunId,
        _update: EffectUpdate,
    ) -> Result<(), MachineError> {
        Err(MachineError::RuntimeOpUnavailable)
    }

    async fn push_item(&self, _run_id: &RunId, _item: ItemWrite) -> Result<(), MachineError> {
        Err(MachineError::RuntimeOpUnavailable)
    }

    async fn push_entry(&self, _run_id: &RunId, _entry: EntryWrite) -> Result<(), MachineError> {
        Err(MachineError::RuntimeOpUnavailable)
    }
}
