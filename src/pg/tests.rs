use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use deadpool_postgres::tokio_postgres::NoTls;
use deadpool_postgres::{Pool, Runtime};

use super::*;
use crate::checkpoint::CheckpointRecord;
use crate::error::MachineError;
use crate::run::{LeaseId, RunId, SessionId, WorkerId};
use crate::runtime::{Event, Payload};
use crate::store::{
    Lease, LeaseClaim, RunCommit, RunCommitResult, RunEventEnvelope, RunLease, RunStart, RunStatus,
    RunStore, RunTx, StoreStartResult,
};

#[test]
fn pg_store_roundtrip_skips_without_test_database_url() {
    let Some(url) = std::env::var("TEST_DATABASE_URL").ok() else {
        return;
    };
    if !url.to_ascii_lowercase().contains("test") || url.ends_with("/postgres") {
        panic!("refusing to run typemach pg test against non-test database");
    }

    block_on(async {
        let store = PgStore::<Event>::new(pool(url.clone()));
        reset_schema(&store).await;
        store.ensure_schema().await.expect("schema");
        let scope = serde_json::json!({"tenant": "typemach-test"});
        let run_id = RunId::from(format!("run-{}", unique()));
        let session_id = SessionId::from(format!("session-{}", unique()));
        let thread_id = crate::run::ThreadId::from(format!("thread-{}", unique()));
        store
            .start_run(&RunStart {
                run_id: run_id.clone(),
                session_id: session_id.clone(),
                thread_id: thread_id.clone(),
                agent_kind: "test".to_string(),
                model: None,
                client_run_key: Some("key-a".to_string()),
                parent_run_id: None,
                retry_of_run_id: None,
                scope: scope.clone(),
                metadata: serde_json::json!({}),
                lease: None,
            })
            .await
            .expect("start");

        let checkpoint = CheckpointRecord::running(
            serde_json::json!({"value": 1}),
            Some(serde_json::json!("done")),
            run_id.as_str(),
        );
        let event = RunEventEnvelope::new(
            run_id.clone(),
            session_id.clone(),
            1,
            Payload::StepDone {
                step: serde_json::json!("start"),
                result: crate::run::StepResult::Next,
            },
        );
        let result = store
            .commit_run(&RunCommit {
                run_id: run_id.clone(),
                session_id: session_id.clone(),
                scope: scope.clone(),
                lease: None,
                checkpoint: Some(crate::store::CheckpointWrite::new(
                    thread_id.clone(),
                    checkpoint.clone(),
                )),
                events: vec![event],
                effects: Vec::new(),
                items: Vec::new(),
                finish: None,
            })
            .await
            .expect("commit");
        assert!(matches!(result, RunCommitResult::Recorded(_)));
        assert_eq!(
            store
                .list_events(&run_id, &scope, 0, usize::MAX)
                .await
                .expect("events")
                .len(),
            1
        );

        let lease_run = RunId::from(format!("run-lease-{}", unique()));
        let lease_session = SessionId::from(format!("session-lease-{}", unique()));
        let lease_thread = crate::run::ThreadId::from(format!("thread-lease-{}", unique()));
        let owner = WorkerId::from("worker-a");
        let lease_id = LeaseId::from("lease-a");
        store
            .start_run(&RunStart {
                run_id: lease_run.clone(),
                session_id: lease_session.clone(),
                thread_id: lease_thread.clone(),
                agent_kind: "test".to_string(),
                model: None,
                client_run_key: None,
                parent_run_id: None,
                retry_of_run_id: None,
                scope: scope.clone(),
                metadata: serde_json::json!({}),
                lease: Some(LeaseClaim::new(
                    owner.clone(),
                    lease_id.clone(),
                    Duration::from_secs(30),
                )),
            })
            .await
            .expect("start leased");
        let leased_event = RunEventEnvelope::new(
            lease_run.clone(),
            lease_session.clone(),
            1,
            Payload::Beat {
                thread_id: lease_thread.clone(),
            },
        );
        let missing_lease = RunCommit {
            run_id: lease_run.clone(),
            session_id: lease_session.clone(),
            scope: scope.clone(),
            lease: None,
            checkpoint: None,
            events: vec![leased_event.clone()],
            effects: Vec::new(),
            items: Vec::new(),
            finish: None,
        };
        assert!(matches!(
            store.commit_run(&missing_lease).await,
            Err(MachineError::LeaseLost)
        ));
        assert!(matches!(
            store
                .commit_run(&RunCommit {
                    lease: Some(LeaseId::from("wrong-lease")),
                    ..missing_lease.clone()
                })
                .await,
            Err(MachineError::LeaseLost)
        ));
        assert!(
            store
                .renew(
                    &Lease::new(lease_run.clone(), owner.clone(), lease_id.clone()),
                    Duration::from_secs(30)
                )
                .await
                .expect("renew")
        );
        assert!(matches!(
            store
                .commit_run(&RunCommit {
                    lease: Some(lease_id),
                    ..missing_lease
                })
                .await
                .expect("leased commit"),
            RunCommitResult::Recorded(_)
        ));

        let stale_run = RunId::from(format!("run-stale-{}", unique()));
        let stale_session = SessionId::from(format!("session-stale-{}", unique()));
        let stale_thread = crate::run::ThreadId::from(format!("thread-stale-{}", unique()));
        store
            .start_run(&RunStart {
                run_id: stale_run.clone(),
                session_id: stale_session.clone(),
                thread_id: stale_thread,
                agent_kind: "test".to_string(),
                model: None,
                client_run_key: None,
                parent_run_id: None,
                retry_of_run_id: None,
                scope: scope.clone(),
                metadata: serde_json::json!({}),
                lease: Some(LeaseClaim::new(
                    WorkerId::from("worker-stale"),
                    LeaseId::from("lease-stale"),
                    Duration::from_millis(1),
                )),
            })
            .await
            .expect("start stale");
        async_rt::time::sleep(Duration::from_millis(5)).await;
        let reaped = store
            .reap_stale(&WorkerId::from("reaper"), 8, |run, seq| {
                RunEventEnvelope::new(
                    run.run_id.clone(),
                    run.session_id.clone(),
                    seq,
                    Payload::Fail {
                        error: "lease expired".to_string(),
                    },
                )
            })
            .await
            .expect("reap stale");
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].run_id, stale_run);
        assert_eq!(reaped[0].status, RunStatus::Error);

        let shared_session = SessionId::from(format!("session-shared-{}", unique()));
        let scope_a = serde_json::json!({"tenant": "tenant-a"});
        let scope_b = serde_json::json!({"tenant": "tenant-b"});
        let run_a = RunId::from(format!("run-a-{}", unique()));
        let run_b = RunId::from(format!("run-b-{}", unique()));
        assert!(matches!(
            store
                .start_run(&RunStart {
                    run_id: run_a.clone(),
                    session_id: shared_session.clone(),
                    thread_id: crate::run::ThreadId::from(format!("thread-a-{}", unique())),
                    agent_kind: "test".to_string(),
                    model: None,
                    client_run_key: Some("same-key".to_string()),
                    parent_run_id: None,
                    retry_of_run_id: None,
                    scope: scope_a.clone(),
                    metadata: serde_json::json!({}),
                    lease: None,
                })
                .await
                .expect("start scope a"),
            StoreStartResult::Created
        ));
        assert!(matches!(
            store
                .start_run(&RunStart {
                    run_id: run_b.clone(),
                    session_id: shared_session.clone(),
                    thread_id: crate::run::ThreadId::from(format!("thread-b-{}", unique())),
                    agent_kind: "test".to_string(),
                    model: None,
                    client_run_key: Some("same-key".to_string()),
                    parent_run_id: None,
                    retry_of_run_id: None,
                    scope: scope_b.clone(),
                    metadata: serde_json::json!({}),
                    lease: None,
                })
                .await
                .expect("start scope b"),
            StoreStartResult::Created
        ));
        assert_eq!(
            store
                .find_idempotent_run(&scope_a, &shared_session, "same-key")
                .await
                .expect("idem a")
                .expect("run a")
                .run_id,
            run_a
        );
        assert_eq!(
            store
                .find_idempotent_run(&scope_b, &shared_session, "same-key")
                .await
                .expect("idem b")
                .expect("run b")
                .run_id,
            run_b
        );

        reset_schema(&store).await;
        store.ensure_schema().await.expect("schema");
        let contract = PgStore::<crate::testkit::TestEvent>::new(pool(url));
        crate::testkit::run_store_contract(&contract)
            .await
            .expect("contract");
    });
}

#[test]
fn scope_key_is_stable_for_unordered_maps() {
    let mut left = HashMap::new();
    left.insert("tenant", "demo");
    left.insert("shop", "north");
    let mut right = HashMap::new();
    right.insert("shop", "north");
    right.insert("tenant", "demo");

    assert_eq!(
        scope_key(&left).expect("left"),
        scope_key(&right).expect("right")
    );
}

async fn reset_schema(store: &PgStore<Event>) {
    let client = store.pool().get().await.expect("pool client");
    client
        .batch_execute(
            "DROP TABLE IF EXISTS typemach_run_events CASCADE;
                 DROP TABLE IF EXISTS typemach_thread_leases CASCADE;
                 DROP TABLE IF EXISTS typemach_runs CASCADE;
                 DROP TABLE IF EXISTS typemach_sessions CASCADE;
                 DROP TABLE IF EXISTS typemach_checkpoints CASCADE;",
        )
        .await
        .expect("reset schema");
}

fn pool(url: String) -> Pool {
    let mut cfg = deadpool_postgres::Config::new();
    cfg.url = Some(url);
    cfg.create_pool(Some(Runtime::Tokio1), NoTls)
        .expect("create pool")
}

fn unique() -> String {
    format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    )
}

fn block_on<F>(future: F) -> F::Output
where
    F: std::future::Future,
{
    async_rt::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
        .block_on(future)
}
