use std::sync::Arc;

use tokio::sync::mpsc;

use crate::AgentError;

#[derive(Clone)]
pub struct ModelStream {
    tx: mpsc::UnboundedSender<OutputTextDelta>,
    emitted: Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputTextDelta(pub String);

impl ModelStream {
    pub(crate) fn new(tx: mpsc::UnboundedSender<OutputTextDelta>) -> Self {
        Self {
            tx,
            emitted: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    pub fn channel() -> (Self, mpsc::UnboundedReceiver<OutputTextDelta>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self::new(tx), rx)
    }

    pub fn output_text(&self, delta: impl Into<String>) -> Result<(), AgentError> {
        self.tx
            .send(OutputTextDelta(delta.into()))
            .map_err(|_| AgentError::Model("model delta stream closed".to_string()))?;
        self.emitted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn emitted(&self) -> usize {
        self.emitted.load(std::sync::atomic::Ordering::Relaxed)
    }
}
