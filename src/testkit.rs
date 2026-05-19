use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::checkpoint::{CheckpointRecord, CheckpointStore};
use crate::error::MachineError;
use crate::run::{LeaseId, RunId, SessionId, ThreadId, WorkerId};
use crate::store::{
    CheckpointWrite, FinishRunResult, Lease, LeaseClaim, RunCommit, RunCommitResult,
    RunEventEnvelope, RunEventPayload, RunFinish, RunLease, RunStart, RunStatus, RunTx,
    StoreStartResult,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestPayload {
    pub terminal: bool,
    pub name: String,
}

impl RunEventPayload for TestPayload {
    fn is_terminal(&self) -> bool {
        self.terminal
    }
}

pub type TestEvent = RunEventEnvelope<TestPayload>;

pub async fn run_store_contract<S>(store: &S) -> Result<(), MachineError>
where
    S: CheckpointStore + RunLease<TestEvent> + RunTx<TestEvent, Scope = Value, FinishData = ()>,
{
    idempotent_start_and_scope(store).await?;
    event_checkpoint_and_terminal(store).await?;
    rejected_commits_do_not_advance(store).await?;
    running_thread_is_exclusive(store).await?;
    lease_and_thread_fencing(store).await?;
    stale_reap_releases_thread(store).await?;
    Ok(())
}

async fn idempotent_start_and_scope<S>(store: &S) -> Result<(), MachineError>
where
    S: RunTx<TestEvent, Scope = Value, FinishData = ()>,
{
    let alpha = scope("alpha");
    let first = run_start(
        "contract-idem-a",
        "contract-session-idem",
        "contract-thread-idem",
        alpha.clone(),
    );
    assert!(matches!(
        store.start_run(&with_key(first, "idem-key")).await?,
        StoreStartResult::Created
    ));

    let second = run_start(
        "contract-idem-b",
        "contract-session-idem",
        "contract-thread-idem-b",
        alpha,
    );
    let existing = store.start_run(&with_key(second, "idem-key")).await?;
    let StoreStartResult::Existing(existing) = existing else {
        panic!("same scope/session/key must return existing run");
    };
    assert_eq!(existing.run_id, RunId::from("contract-idem-a"));

    let other = run_start(
        "contract-idem-c",
        "contract-session-idem",
        "contract-thread-idem-c",
        scope("beta"),
    );
    assert!(matches!(
        store.start_run(&with_key(other, "idem-key")).await?,
        StoreStartResult::Created
    ));
    Ok(())
}

async fn event_checkpoint_and_terminal<S>(store: &S) -> Result<(), MachineError>
where
    S: CheckpointStore + RunTx<TestEvent, Scope = Value, FinishData = ()>,
{
    let scope = scope("events");
    let run_id = RunId::from("contract-events-run");
    let session_id = SessionId::from("contract-events-session");
    let thread_id = ThreadId::from("contract-events-thread");
    let start = run_start(
        run_id.as_str(),
        session_id.as_str(),
        thread_id.as_str(),
        scope.clone(),
    );
    assert!(matches!(
        store.start_run(&start).await?,
        StoreStartResult::Created
    ));

    let checkpoint =
        CheckpointRecord::running(json!({"step": 1}), Some(json!("next")), run_id.as_str());
    let commit = RunCommit {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        scope: scope.clone(),
        lease: None,
        checkpoint: Some(CheckpointWrite::new(thread_id.clone(), checkpoint.clone())),
        events: vec![event(run_id.as_str(), session_id.as_str(), 1, false)],
        finish: None,
    };
    assert!(matches!(
        store.commit_run(&commit).await?,
        RunCommitResult::Recorded(_)
    ));
    assert!(store.load_checkpoint(thread_id.as_str()).await?.is_some());

    let duplicate = RunCommit {
        events: vec![event(run_id.as_str(), session_id.as_str(), 1, false)],
        ..commit.clone()
    };
    assert!(matches!(
        store.commit_run(&duplicate).await,
        Err(MachineError::InvalidRunEvent { .. })
    ));

    let finish = RunFinish {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        scope: scope.clone(),
        status: RunStatus::Completed,
        finish_reason: "done".to_string(),
        error_code: None,
        data: (),
    };
    let done = RunCommit {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        scope,
        lease: None,
        checkpoint: None,
        events: vec![event(run_id.as_str(), session_id.as_str(), 2, true)],
        finish: Some(finish),
    };
    assert!(matches!(
        store.commit_run(&done).await?,
        RunCommitResult::Finished {
            result: FinishRunResult::Finished(_),
            ..
        }
    ));
    assert!(matches!(
        store.commit_run(&done).await?,
        RunCommitResult::Finished {
            result: FinishRunResult::AlreadyFinished(_),
            ..
        }
    ));
    Ok(())
}

async fn rejected_commits_do_not_advance<S>(store: &S) -> Result<(), MachineError>
where
    S: CheckpointStore + RunTx<TestEvent, Scope = Value, FinishData = ()>,
{
    let scope = scope("reject");
    let run_id = RunId::from("contract-reject-run");
    let session_id = SessionId::from("contract-reject-session");
    let thread_id = ThreadId::from("contract-reject-thread");
    let start = run_start(
        run_id.as_str(),
        session_id.as_str(),
        thread_id.as_str(),
        scope.clone(),
    );
    assert!(matches!(
        store.start_run(&start).await?,
        StoreStartResult::Created
    ));

    assert!(matches!(
        store
            .record_event(
                &run_id,
                &scope,
                &event(run_id.as_str(), session_id.as_str(), 1, true),
            )
            .await,
        Err(MachineError::InvalidRunEvent { .. })
    ));
    assert!(store.terminal_event(&run_id, &scope).await?.is_none());

    let first_checkpoint =
        CheckpointRecord::running(json!({"step": 1}), Some(json!("next")), run_id.as_str());
    let first = RunCommit {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        scope: scope.clone(),
        lease: None,
        checkpoint: Some(CheckpointWrite::new(
            thread_id.clone(),
            first_checkpoint.clone(),
        )),
        events: vec![event(run_id.as_str(), session_id.as_str(), 1, false)],
        finish: None,
    };
    assert!(matches!(
        store.commit_run(&first).await?,
        RunCommitResult::Recorded(_)
    ));

    let bad_checkpoint =
        CheckpointRecord::running(json!({"step": 999}), Some(json!("bad")), run_id.as_str());
    let bad = RunCommit {
        checkpoint: Some(CheckpointWrite::new(thread_id.clone(), bad_checkpoint)),
        events: vec![event(run_id.as_str(), session_id.as_str(), 1, false)],
        ..first
    };
    assert!(matches!(
        store.commit_run(&bad).await,
        Err(MachineError::InvalidRunEvent { .. })
    ));
    let loaded = store
        .load_checkpoint(thread_id.as_str())
        .await?
        .expect("checkpoint");
    assert_eq!(loaded.state, first_checkpoint.state);
    assert_eq!(loaded.next_step, first_checkpoint.next_step);
    Ok(())
}

async fn running_thread_is_exclusive<S>(store: &S) -> Result<(), MachineError>
where
    S: RunTx<TestEvent, Scope = Value, FinishData = ()>,
{
    let scope = scope("thread");
    let thread_id = ThreadId::from("contract-thread-exclusive");
    let run_id = RunId::from("contract-thread-run");
    let session_id = SessionId::from("contract-thread-session");
    let start = run_start(
        run_id.as_str(),
        session_id.as_str(),
        thread_id.as_str(),
        scope.clone(),
    );
    assert!(matches!(
        store.start_run(&start).await?,
        StoreStartResult::Created
    ));

    let mut blocked = run_start(
        "contract-thread-run-blocked",
        session_id.as_str(),
        thread_id.as_str(),
        scope,
    );
    blocked.lease = Some(LeaseClaim::new(
        WorkerId::from("contract-thread-worker"),
        LeaseId::from("contract-thread-lease"),
        Duration::from_secs(30),
    ));
    assert!(matches!(
        store.start_run(&blocked).await,
        Err(MachineError::ThreadBusy { .. })
    ));
    Ok(())
}

async fn lease_and_thread_fencing<S>(store: &S) -> Result<(), MachineError>
where
    S: CheckpointStore + RunLease<TestEvent> + RunTx<TestEvent, Scope = Value, FinishData = ()>,
{
    let scope = scope("lease");
    let owner = WorkerId::from("contract-worker-a");
    let lease_id = LeaseId::from("contract-lease-a");
    let run_id = RunId::from("contract-lease-run");
    let session_id = SessionId::from("contract-lease-session");
    let thread_id = ThreadId::from("contract-lease-thread");
    let mut start = run_start(
        run_id.as_str(),
        session_id.as_str(),
        thread_id.as_str(),
        scope.clone(),
    );
    start.lease = Some(LeaseClaim::new(
        owner.clone(),
        lease_id.clone(),
        Duration::from_secs(30),
    ));
    assert!(matches!(
        store.start_run(&start).await?,
        StoreStartResult::Created
    ));

    let mut blocked = run_start(
        "contract-lease-run-blocked",
        "contract-lease-session",
        thread_id.as_str(),
        scope.clone(),
    );
    blocked.lease = Some(LeaseClaim::new(
        WorkerId::from("contract-worker-b"),
        LeaseId::from("contract-lease-b"),
        Duration::from_secs(30),
    ));
    assert!(matches!(
        store.start_run(&blocked).await,
        Err(MachineError::ThreadBusy { .. })
    ));
    let blocked_unleased = run_start(
        "contract-lease-run-unleased",
        "contract-lease-session",
        thread_id.as_str(),
        scope.clone(),
    );
    assert!(matches!(
        store.start_run(&blocked_unleased).await,
        Err(MachineError::ThreadBusy { .. })
    ));

    let missing_lease = RunCommit {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        scope: scope.clone(),
        lease: None,
        checkpoint: None,
        events: vec![event(run_id.as_str(), session_id.as_str(), 1, false)],
        finish: None,
    };
    assert!(matches!(
        store.commit_run(&missing_lease).await,
        Err(MachineError::LeaseLost)
    ));

    let wrong_thread = RunCommit {
        lease: Some(lease_id.clone()),
        checkpoint: Some(CheckpointWrite::new(
            ThreadId::from("contract-wrong-thread"),
            CheckpointRecord::running(json!({"bad": true}), None, run_id.as_str()),
        )),
        ..missing_lease.clone()
    };
    assert!(matches!(
        store.commit_run(&wrong_thread).await,
        Err(MachineError::InvalidRunEvent { .. })
    ));

    assert!(
        store
            .renew(
                &Lease::new(run_id.clone(), owner.clone(), lease_id.clone()),
                Duration::from_secs(30)
            )
            .await?
    );
    let checkpoint = CheckpointRecord::running(json!({"leased": true}), None, run_id.as_str());
    let ok = RunCommit {
        lease: Some(lease_id),
        checkpoint: Some(CheckpointWrite::new(thread_id.clone(), checkpoint)),
        ..missing_lease
    };
    assert!(matches!(
        store.commit_run(&ok).await?,
        RunCommitResult::Recorded(_)
    ));
    assert!(store.load_checkpoint(thread_id.as_str()).await?.is_some());
    Ok(())
}

async fn stale_reap_releases_thread<S>(store: &S) -> Result<(), MachineError>
where
    S: RunLease<TestEvent> + RunTx<TestEvent, Scope = Value, FinishData = ()>,
{
    let scope = scope("reap");
    let run_id = RunId::from("contract-stale-run");
    let session_id = SessionId::from("contract-stale-session");
    let thread_id = ThreadId::from("contract-stale-thread");
    let mut stale = run_start(
        run_id.as_str(),
        session_id.as_str(),
        thread_id.as_str(),
        scope.clone(),
    );
    stale.lease = Some(LeaseClaim::new(
        WorkerId::from("contract-stale-worker"),
        LeaseId::from("contract-stale-lease"),
        Duration::from_millis(1),
    ));
    assert!(matches!(
        store.start_run(&stale).await?,
        StoreStartResult::Created
    ));
    async_rt::time::sleep(Duration::from_millis(5)).await;
    let reaped = store
        .reap_stale(&WorkerId::from("contract-reaper"), 8, |run, seq| {
            event(run.run_id.as_str(), run.session_id.as_str(), seq, true)
        })
        .await?;
    assert_eq!(reaped.len(), 1);
    assert_eq!(reaped[0].status, RunStatus::Error);
    assert!(reaped[0].owner.is_none());

    let mut next = run_start(
        "contract-stale-next",
        session_id.as_str(),
        thread_id.as_str(),
        scope,
    );
    next.lease = Some(LeaseClaim::new(
        WorkerId::from("contract-next-worker"),
        LeaseId::from("contract-next-lease"),
        Duration::from_secs(30),
    ));
    assert!(matches!(
        store.start_run(&next).await?,
        StoreStartResult::Created
    ));
    Ok(())
}

fn run_start(run_id: &str, session_id: &str, thread_id: &str, scope: Value) -> RunStart {
    RunStart {
        run_id: RunId::from(run_id),
        session_id: SessionId::from(session_id),
        thread_id: ThreadId::from(thread_id),
        agent_kind: "contract".to_string(),
        model: None,
        client_run_key: None,
        parent_run_id: None,
        retry_of_run_id: None,
        scope,
        metadata: json!({}),
        lease: None,
    }
}

fn with_key(mut start: RunStart, key: &str) -> RunStart {
    start.client_run_key = Some(key.to_string());
    start
}

fn scope(name: &str) -> Value {
    json!({ "tenant": name })
}

fn event(run_id: &str, session_id: &str, seq: i64, terminal: bool) -> TestEvent {
    RunEventEnvelope::new(
        RunId::from(run_id),
        SessionId::from(session_id),
        seq,
        TestPayload {
            terminal,
            name: if terminal { "terminal" } else { "event" }.to_string(),
        },
    )
}
