use std::sync::Arc;

use tokio::sync::mpsc;

use crate::AgentError;

#[derive(Clone)]
pub struct ModelStream {
    tx: mpsc::UnboundedSender<ModelDelta>,
    emitted: Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDelta {
    pub text: String,
    pub final_answer: bool,
}

impl std::ops::Deref for ModelDelta {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.text
    }
}

impl ModelStream {
    pub(crate) fn new(tx: mpsc::UnboundedSender<ModelDelta>) -> Self {
        Self {
            tx,
            emitted: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    pub fn channel() -> (Self, mpsc::UnboundedReceiver<ModelDelta>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self::new(tx), rx)
    }

    pub fn delta(&self, delta: impl Into<String>) -> Result<(), AgentError> {
        self.send(delta.into(), false)
    }

    pub fn final_delta(&self, delta: impl Into<String>) -> Result<(), AgentError> {
        self.send(delta.into(), true)
    }

    fn send(&self, text: String, final_answer: bool) -> Result<(), AgentError> {
        self.tx
            .send(ModelDelta { text, final_answer })
            .map_err(|_| AgentError::Model("model delta stream closed".to_string()))?;
        self.emitted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn emitted(&self) -> usize {
        self.emitted.load(std::sync::atomic::Ordering::Relaxed)
    }
}
