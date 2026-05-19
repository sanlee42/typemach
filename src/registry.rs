use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_rt::sync::mpsc;

use crate::error::MachineError;
use crate::run::RunId;

#[derive(Debug, Clone)]
pub struct RunHandle {
    pub token: String,
    cancelled: Arc<AtomicBool>,
}

impl RunHandle {
    pub fn new(token: String) -> Self {
        Self {
            token,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

#[derive(Debug)]
struct ActiveRun<E> {
    handle: RunHandle,
    current_seq: i64,
    terminal_published: bool,
    stream_senders: Vec<mpsc::UnboundedSender<E>>,
}

#[derive(Debug, Clone)]
pub struct RunRegistry<E> {
    inner: Arc<Mutex<HashMap<RunId, ActiveRun<E>>>>,
}

impl<E> Default for RunRegistry<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> RunRegistry<E> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn try_insert(
        &self,
        run_id: RunId,
        handle: RunHandle,
        stream_sender: Option<mpsc::UnboundedSender<E>>,
        max_in_flight: usize,
    ) -> Result<(), MachineError> {
        let mut inner = self.inner.lock().expect("run registry lock poisoned");
        if inner.contains_key(&run_id) {
            return Err(MachineError::RunAlreadyActive);
        }
        if inner.len() >= max_in_flight {
            return Err(MachineError::CapacityExceeded);
        }
        inner.insert(
            run_id,
            ActiveRun {
                handle,
                current_seq: 0,
                terminal_published: false,
                stream_senders: stream_sender.into_iter().collect(),
            },
        );
        Ok(())
    }

    pub fn remove(&self, run_id: &RunId) {
        self.inner
            .lock()
            .expect("run registry lock poisoned")
            .remove(run_id);
    }

    pub fn resolve(&self, run_id: &RunId, token: &str) -> Option<RunHandle> {
        let inner = self.inner.lock().expect("run registry lock poisoned");
        inner
            .get(run_id)
            .filter(|run| run.handle.token == token)
            .map(|run| run.handle.clone())
    }

    pub fn handle(&self, run_id: &RunId) -> Option<RunHandle> {
        self.inner
            .lock()
            .expect("run registry lock poisoned")
            .get(run_id)
            .map(|run| run.handle.clone())
    }

    pub fn next_seq(&self, run_id: &RunId) -> Option<i64> {
        let mut inner = self.inner.lock().expect("run registry lock poisoned");
        let run = inner.get_mut(run_id)?;
        run.current_seq += 1;
        Some(run.current_seq)
    }

    pub(crate) fn rewind_seq(&self, run_id: &RunId, seq: i64) -> bool {
        let mut inner = self.inner.lock().expect("run registry lock poisoned");
        let Some(run) = inner.get_mut(run_id) else {
            return false;
        };
        if run.current_seq != seq {
            return false;
        }
        run.current_seq -= 1;
        true
    }

    pub fn subscribe(&self, run_id: &RunId) -> Option<mpsc::UnboundedReceiver<E>> {
        let mut inner = self.inner.lock().expect("run registry lock poisoned");
        let run = inner.get_mut(run_id)?;
        if run.terminal_published {
            return None;
        }
        let (sender, receiver) = mpsc::unbounded_channel();
        run.stream_senders.push(sender);
        Some(receiver)
    }

    pub fn request_cancel(&self, run_id: &RunId) -> Option<RunHandle> {
        let handle = self.handle(run_id)?;
        handle.cancel();
        Some(handle)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("run registry lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .expect("run registry lock poisoned")
            .is_empty()
    }
}

impl<E: Clone> RunRegistry<E> {
    pub fn publish(&self, run_id: &RunId, event: E) {
        let mut inner = self.inner.lock().expect("run registry lock poisoned");
        if let Some(run) = inner.get_mut(run_id) {
            if run.terminal_published {
                return;
            }
            run.stream_senders
                .retain(|sender| sender.send(event.clone()).is_ok());
        }
    }

    pub fn publish_terminal(&self, run_id: &RunId, event: E) -> bool {
        let mut inner = self.inner.lock().expect("run registry lock poisoned");
        let Some(run) = inner.get_mut(run_id) else {
            return false;
        };
        if run.terminal_published {
            return false;
        }
        run.terminal_published = true;
        run.stream_senders
            .retain(|sender| sender.send(event.clone()).is_ok());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_run_id(id: &str) -> RunId {
        RunId::from(id)
    }

    fn test_handle() -> RunHandle {
        RunHandle::new("token".to_string())
    }

    #[test]
    fn rejects_when_full() {
        let registry = RunRegistry::<i64>::new();
        registry
            .try_insert(test_run_id("run-a"), test_handle(), None, 1)
            .expect("first run should fit");
        assert!(
            registry
                .try_insert(test_run_id("run-b"), test_handle(), None, 1)
                .is_err()
        );
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn rejects_duplicate_run_without_replacing_existing() {
        let registry = RunRegistry::<i64>::new();
        let (first_sender, mut first_receiver) = mpsc::unbounded_channel();
        registry
            .try_insert(test_run_id("run-a"), test_handle(), Some(first_sender), 1)
            .expect("first run should fit");

        let (second_sender, mut second_receiver) = mpsc::unbounded_channel();
        let err = registry
            .try_insert(
                test_run_id("run-a"),
                RunHandle::new("replacement".to_string()),
                Some(second_sender),
                1,
            )
            .expect_err("duplicate run should be rejected");

        assert!(matches!(err, MachineError::RunAlreadyActive));
        assert!(
            registry
                .resolve(&test_run_id("run-a"), "replacement")
                .is_none()
        );

        registry.publish(&test_run_id("run-a"), 1);
        assert_eq!(first_receiver.try_recv(), Ok(1));
        assert!(second_receiver.try_recv().is_err());
    }

    #[test]
    fn sequences_events_per_run() {
        let registry = RunRegistry::<i64>::new();
        registry
            .try_insert(test_run_id("run-a"), test_handle(), None, 1)
            .expect("run should fit");
        assert_eq!(registry.next_seq(&test_run_id("run-a")), Some(1));
        assert_eq!(registry.next_seq(&test_run_id("run-a")), Some(2));
        assert_eq!(registry.next_seq(&test_run_id("missing")), None);
    }

    #[test]
    fn marks_cancellation() {
        let registry = RunRegistry::<i64>::new();
        registry
            .try_insert(test_run_id("run-a"), test_handle(), None, 1)
            .expect("run should fit");
        let cancelled = registry
            .request_cancel(&test_run_id("run-a"))
            .expect("active run");
        assert!(cancelled.is_cancelled());
        assert!(
            registry
                .resolve(&test_run_id("run-a"), "token")
                .expect("context")
                .is_cancelled()
        );
    }

    #[test]
    fn publish_terminal_returns_false_when_already_terminal() {
        let registry = RunRegistry::<i64>::new();
        let (sender, mut receiver) = mpsc::unbounded_channel();
        registry
            .try_insert(test_run_id("run-a"), test_handle(), Some(sender), 1)
            .expect("run should fit");

        assert!(registry.publish_terminal(&test_run_id("run-a"), 1));
        assert!(!registry.publish_terminal(&test_run_id("run-a"), 2));

        assert_eq!(receiver.try_recv(), Ok(1));
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn terminal_run_rejects_late_subscribers_and_publish() {
        let registry = RunRegistry::<i64>::new();
        let (sender, mut receiver) = mpsc::unbounded_channel();
        registry
            .try_insert(test_run_id("run-a"), test_handle(), Some(sender), 1)
            .expect("run should fit");

        assert!(registry.publish_terminal(&test_run_id("run-a"), 1));
        assert!(registry.subscribe(&test_run_id("run-a")).is_none());
        registry.publish(&test_run_id("run-a"), 2);

        assert_eq!(receiver.try_recv(), Ok(1));
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn registry_accepts_non_clone_event_for_lifecycle_operations() {
        #[derive(Debug)]
        struct NonClone;

        let registry = RunRegistry::<NonClone>::new();
        registry
            .try_insert(test_run_id("run-a"), test_handle(), None, 1)
            .expect("run should fit");
        assert_eq!(registry.next_seq(&test_run_id("run-a")), Some(1));
        assert!(registry.subscribe(&test_run_id("run-a")).is_some());
        assert!(registry.request_cancel(&test_run_id("run-a")).is_some());
    }

    #[test]
    fn publish_always_accepts() {
        let registry = RunRegistry::<i64>::new();
        let (sender, mut receiver) = mpsc::unbounded_channel();
        registry
            .try_insert(test_run_id("run-a"), test_handle(), Some(sender), 1)
            .expect("run should fit");

        registry.publish(&test_run_id("run-a"), 1);
        registry.publish(&test_run_id("run-a"), 2);

        assert_eq!(receiver.try_recv(), Ok(1));
        assert_eq!(receiver.try_recv(), Ok(2));
    }
}
