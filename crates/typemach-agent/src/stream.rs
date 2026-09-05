use std::sync::Arc;

use tokio::sync::mpsc;

use crate::{
    AgentError, AssistantMessageId, AssistantMessageItem, ResponseContentIndex, ResponseOutputIndex,
};

#[derive(Clone)]
pub struct ModelStream {
    tx: mpsc::UnboundedSender<ModelStreamEvent>,
    emitted: Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStreamEvent {
    AssistantMessageStarted {
        message_id: AssistantMessageId,
        output_index: ResponseOutputIndex,
    },
    AssistantMessageDelta {
        message_id: AssistantMessageId,
        output_index: ResponseOutputIndex,
        content_index: ResponseContentIndex,
        delta: String,
        index: usize,
    },
    AssistantMessageDone {
        message: AssistantMessageItem,
    },
}

impl ModelStream {
    pub(crate) fn new(tx: mpsc::UnboundedSender<ModelStreamEvent>) -> Self {
        Self {
            tx,
            emitted: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    pub fn channel() -> (Self, mpsc::UnboundedReceiver<ModelStreamEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self::new(tx), rx)
    }

    pub fn emit(&self, event: ModelStreamEvent) -> Result<(), AgentError> {
        self.tx
            .send(event)
            .map_err(|_| AgentError::Model("model event stream closed".to_string()))?;
        self.emitted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn emitted(&self) -> usize {
        self.emitted.load(std::sync::atomic::Ordering::Relaxed)
    }
}
