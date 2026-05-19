use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
struct TestEvent {
    run_id: RunId,
    session_id: SessionId,
    seq: i64,
    terminal: bool,
    name: &'static str,
}

impl RunEvent for TestEvent {
    fn run_id(&self) -> &RunId {
        &self.run_id
    }

    fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    fn seq(&self) -> i64 {
        self.seq
    }

    fn is_terminal(&self) -> bool {
        self.terminal
    }
}

fn run_start(run_id: &str, session_id: &str, key: Option<&str>) -> RunStart {
    RunStart {
        run_id: RunId::from(run_id),
        session_id: SessionId::from(session_id),
        thread_id: ThreadId::from(format!("thread-{run_id}")),
        agent_kind: "test".to_string(),
        model: None,
        client_run_key: key.map(str::to_string),
        parent_run_id: None,
        retry_of_run_id: None,
        scope: serde_json::json!({"tenant": "demo"}),
        metadata: serde_json::json!({}),
        input: None,
        entries: Vec::new(),
        lease: None,
    }
}

fn event(run_id: &str, session_id: &str, seq: i64, terminal: bool) -> TestEvent {
    TestEvent {
        run_id: RunId::from(run_id),
        session_id: SessionId::from(session_id),
        seq,
        terminal,
        name: if terminal { "terminal" } else { "event" },
    }
}

#[test]
fn memory_store_idempotent_start_returns_existing_run() {
    block_on(async {
        let store = MemoryRunStore::<TestEvent>::new();
        let first = run_start("run-a", "session-a", Some("client-key"));
        let mut second = run_start("run-b", "session-a", Some("client-key"));
        second.thread_id = first.thread_id.clone();
        assert!(matches!(
            store.start_run(&first).await.expect("start"),
            StoreStartResult::Created
        ));
        match store.start_run(&second).await.expect("idempotent") {
            StoreStartResult::Existing(existing) => {
                assert_eq!(existing.run_id, RunId::from("run-a"));
                assert_eq!(existing.status, RunStatus::Running);
                assert!(!existing.cancel_requested);
            }
            StoreStartResult::Created => panic!("expected existing run"),
        }
    });
}

#[test]
fn memory_store_records_running_events_and_skips_after_terminal() {
    block_on(async {
        let store = MemoryRunStore::<TestEvent>::new();
        let start = run_start("run-a", "session-a", None);
        store.start_run(&start).await.expect("start");

        assert!(
            store
                .record_event(
                    &RunId::from("run-a"),
                    &start.scope,
                    &event("run-a", "session-a", 1, false)
                )
                .await
                .expect("record")
        );
        let terminal = event("run-a", "session-a", 2, true);
        let finish = RunFinishRecord {
            run_id: RunId::from("run-a"),
            session_id: SessionId::from("session-a"),
            scope: start.scope.clone(),
            status: RunStatus::Completed,
            finish_reason: "stop".to_string(),
            error_code: None,
            terminal_event: terminal.clone(),
            entries: Vec::new(),
            data: (),
        };
        let result = store.finish_run(&finish).await.expect("finish");
        assert!(matches!(result, FinishRunResult::Finished(_)));
        assert_eq!(
            store
                .terminal_event(&RunId::from("run-a"), &start.scope)
                .await
                .expect("terminal event"),
            Some(terminal.clone())
        );
        assert!(
            !store
                .record_event(
                    &RunId::from("run-a"),
                    &start.scope,
                    &event("run-a", "session-a", 3, false)
                )
                .await
                .expect("post-terminal record")
        );

        let events = store
            .list_events(&RunId::from("run-a"), &start.scope, 0, usize::MAX)
            .await
            .expect("events");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].seq, 1);
        assert_eq!(events[1], terminal);
    });
}

#[test]
fn memory_store_terminal_competes_once() {
    block_on(async {
        let store = MemoryRunStore::<TestEvent>::new();
        let start = run_start("run-a", "session-a", None);
        store.start_run(&start).await.expect("start");
        let first_terminal = event("run-a", "session-a", 1, true);
        let second_terminal = event("run-a", "session-a", 2, true);
        let first = RunFinishRecord {
            run_id: RunId::from("run-a"),
            session_id: SessionId::from("session-a"),
            scope: start.scope.clone(),
            status: RunStatus::Completed,
            finish_reason: "stop".to_string(),
            error_code: None,
            terminal_event: first_terminal.clone(),
            entries: Vec::new(),
            data: (),
        };
        let mut second = first.clone();
        second.terminal_event = second_terminal;
        second.status = RunStatus::Error;
        second.finish_reason = "runtime_failed".to_string();

        assert!(matches!(
            store.finish_run(&first).await.expect("first"),
            FinishRunResult::Finished(_)
        ));
        let result = store.finish_run(&second).await.expect("second");
        assert!(matches!(result, FinishRunResult::AlreadyFinished(_)));
        assert_eq!(result.into_terminal_event(), first_terminal);
    });
}

#[test]
fn memory_store_marks_cancel_requested() {
    block_on(async {
        let store = MemoryRunStore::<TestEvent>::new();
        let start = run_start("run-a", "session-a", None);
        store.start_run(&start).await.expect("start");

        store
            .mark_cancelled(&RunId::from("run-a"), &start.scope)
            .await
            .expect("cancel");

        let lookup = store
            .lookup_run(&RunId::from("run-a"), &start.scope)
            .await
            .expect("lookup")
            .expect("run");
        assert!(lookup.cancel_requested);
        assert_eq!(lookup.status, RunStatus::Running);
    });
}

#[test]
fn memory_store_scopes_terminal_event_cancel_and_record_paths() {
    block_on(async {
        let store = MemoryRunStore::<TestEvent>::new();
        let start = run_start("run-a", "session-a", None);
        let wrong_scope = serde_json::json!({"tenant": "other"});
        store.start_run(&start).await.expect("start");

        assert!(matches!(
            store
                .record_event(
                    &RunId::from("run-a"),
                    &wrong_scope,
                    &event("run-a", "session-a", 1, false),
                )
                .await,
            Err(MachineError::RunNotFound)
        ));
        assert!(matches!(
            store
                .mark_cancelled(&RunId::from("run-a"), &wrong_scope)
                .await,
            Err(MachineError::RunNotFound)
        ));

        let terminal = event("run-a", "session-a", 1, true);
        let finish = RunFinishRecord {
            run_id: RunId::from("run-a"),
            session_id: SessionId::from("session-a"),
            scope: start.scope.clone(),
            status: RunStatus::Completed,
            finish_reason: "stop".to_string(),
            error_code: None,
            terminal_event: terminal.clone(),
            entries: Vec::new(),
            data: (),
        };
        store.finish_run(&finish).await.expect("finish");

        assert_eq!(
            store
                .terminal_event(&RunId::from("run-a"), &wrong_scope)
                .await
                .expect("terminal wrong scope"),
            None
        );
        assert_eq!(
            store
                .terminal_event(&RunId::from("run-a"), &start.scope)
                .await
                .expect("terminal right scope"),
            Some(terminal)
        );
    });
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct TestScope {
    tenant: &'static str,
}

#[test]
fn memory_store_uses_typed_scope_for_lookup_and_idempotency() {
    block_on(async {
        let store = MemoryRunStore::<TestEvent, TestScope>::new();
        let start = RunStart {
            run_id: RunId::from("run-a"),
            session_id: SessionId::from("session-a"),
            thread_id: ThreadId::from("thread-a"),
            agent_kind: "test".to_string(),
            model: None,
            client_run_key: Some("client-key".to_string()),
            parent_run_id: None,
            retry_of_run_id: None,
            scope: TestScope { tenant: "alpha" },
            metadata: serde_json::json!({}),
            input: None,
            entries: Vec::new(),
            lease: None,
        };

        assert!(matches!(
            store.start_run(&start).await.expect("start"),
            StoreStartResult::Created
        ));
        assert!(
            store
                .lookup_run(&RunId::from("run-a"), &TestScope { tenant: "beta" })
                .await
                .expect("lookup beta")
                .is_none()
        );
        assert!(
            store
                .lookup_run(&RunId::from("run-a"), &TestScope { tenant: "alpha" })
                .await
                .expect("lookup alpha")
                .is_some()
        );

        let retry = RunStart {
            run_id: RunId::from("run-b"),
            session_id: SessionId::from("session-a"),
            thread_id: ThreadId::from("thread-b"),
            agent_kind: "test".to_string(),
            model: None,
            client_run_key: Some("client-key".to_string()),
            parent_run_id: None,
            retry_of_run_id: None,
            scope: TestScope { tenant: "beta" },
            metadata: serde_json::json!({}),
            input: None,
            entries: Vec::new(),
            lease: None,
        };
        assert!(matches!(
            store.start_run(&retry).await.expect("cross-scope start"),
            StoreStartResult::Created
        ));
    });
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TestFinishData {
    snapshot: &'static str,
}

#[test]
fn memory_store_persists_typed_finish_data() {
    block_on(async {
        let store = MemoryRunStore::<TestEvent, Value, TestFinishData>::new();
        let start = run_start("run-a", "session-a", None);
        let data = TestFinishData {
            snapshot: "final-state",
        };
        store.start_run(&start).await.expect("start");

        let finish = RunFinishRecord {
            run_id: RunId::from("run-a"),
            session_id: SessionId::from("session-a"),
            scope: start.scope.clone(),
            status: RunStatus::Completed,
            finish_reason: "stop".to_string(),
            error_code: None,
            terminal_event: event("run-a", "session-a", 1, true),
            entries: Vec::new(),
            data: data.clone(),
        };

        store.finish_run(&finish).await.expect("finish");
        assert_eq!(store.finish_data(&RunId::from("run-a")).await, Some(data));
    });
}

#[test]
fn memory_store_rejects_non_increasing_event_seq() {
    block_on(async {
        let store = MemoryRunStore::<TestEvent>::new();
        let start = run_start("run-a", "session-a", None);
        store.start_run(&start).await.expect("start");
        store
            .record_event(
                &RunId::from("run-a"),
                &start.scope,
                &event("run-a", "session-a", 1, false),
            )
            .await
            .expect("record first");

        let err = store
            .record_event(
                &RunId::from("run-a"),
                &start.scope,
                &event("run-a", "session-a", 1, false),
            )
            .await
            .expect_err("duplicate seq should fail");
        assert!(matches!(err, MachineError::InvalidRunEvent { .. }));
    });
}

#[test]
fn memory_store_lists_events_after_cursor() {
    block_on(async {
        let store = MemoryRunStore::<TestEvent>::new();
        let start = run_start("run-a", "session-a", None);
        store.start_run(&start).await.expect("start");
        for seq in 1..=3 {
            store
                .record_event(
                    &RunId::from("run-a"),
                    &start.scope,
                    &event("run-a", "session-a", seq, false),
                )
                .await
                .expect("record");
        }
        let events = store
            .list_events(&RunId::from("run-a"), &start.scope, 1, usize::MAX)
            .await
            .expect("events");
        assert_eq!(
            events.iter().map(|event| event.seq).collect::<Vec<_>>(),
            vec![2, 3]
        );
    });
}

#[test]
fn memory_store_fences_leased_commits() {
    block_on(async {
        let store = MemoryRunStore::<TestEvent>::new();
        let mut start = run_start("run-a", "session-a", None);
        start.lease = Some(LeaseClaim::new(
            WorkerId::from("worker-a"),
            LeaseId::from("lease-a"),
            std::time::Duration::from_secs(30),
        ));
        store.start_run(&start).await.expect("start");

        let missing = RunCommit {
            run_id: RunId::from("run-a"),
            session_id: SessionId::from("session-a"),
            scope: start.scope.clone(),
            lease: None,
            checkpoint: None,
            events: vec![event("run-a", "session-a", 1, false)],
            effects: Vec::new(),
            items: Vec::new(),
            entries: Vec::new(),
            finish: None,
        };
        assert!(matches!(
            store.commit_run(&missing).await,
            Err(MachineError::LeaseLost)
        ));

        let wrong = RunCommit {
            lease: Some(LeaseId::from("lease-b")),
            ..missing.clone()
        };
        assert!(matches!(
            store.commit_run(&wrong).await,
            Err(MachineError::LeaseLost)
        ));

        let ok = RunCommit {
            lease: Some(LeaseId::from("lease-a")),
            ..missing
        };
        assert!(matches!(
            store.commit_run(&ok).await.expect("commit"),
            RunCommitResult::Recorded(_)
        ));

        let finish = RunFinish {
            run_id: RunId::from("run-a"),
            session_id: SessionId::from("session-a"),
            scope: start.scope.clone(),
            status: RunStatus::Completed,
            finish_reason: "done".to_string(),
            error_code: None,
            entries: Vec::new(),
            data: (),
        };
        let done = RunCommit {
            run_id: RunId::from("run-a"),
            session_id: SessionId::from("session-a"),
            scope: start.scope.clone(),
            lease: Some(LeaseId::from("lease-a")),
            checkpoint: None,
            events: vec![event("run-a", "session-a", 2, true)],
            effects: Vec::new(),
            items: Vec::new(),
            entries: Vec::new(),
            finish: Some(finish),
        };
        assert!(matches!(
            store.commit_run(&done).await.expect("finish"),
            RunCommitResult::Finished { .. }
        ));
    });
}

#[test]
fn memory_store_reaps_stale_leases_once() {
    block_on(async {
        let store = MemoryRunStore::<TestEvent>::new();
        let mut start = run_start("run-a", "session-a", None);
        start.lease = Some(LeaseClaim::new(
            WorkerId::from("worker-a"),
            LeaseId::from("lease-a"),
            std::time::Duration::from_millis(1),
        ));
        store.start_run(&start).await.expect("start");
        async_rt::time::sleep(std::time::Duration::from_millis(5)).await;

        let reaped = store
            .reap_stale(&WorkerId::from("reaper"), 8, |run, seq| {
                event(run.run_id.as_str(), run.session_id.as_str(), seq, true)
            })
            .await
            .expect("reap");
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].status, RunStatus::Error);
        assert_eq!(reaped[0].finish_reason.as_deref(), Some("lease_expired"));

        let lookup = store
            .lookup_run(&RunId::from("run-a"), &start.scope)
            .await
            .expect("lookup")
            .expect("run");
        assert_eq!(lookup.status, RunStatus::Error);
        assert_eq!(
            store
                .reap_stale(&WorkerId::from("reaper"), 8, |run, seq| {
                    event(run.run_id.as_str(), run.session_id.as_str(), seq, true)
                })
                .await
                .expect("second reap")
                .len(),
            0
        );
    });
}

#[test]
fn memory_store_matches_contract() {
    block_on(async {
        let store = MemoryRunStore::<crate::testkit::TestEvent>::new();
        crate::testkit::run_store_contract(&store)
            .await
            .expect("contract");
    });
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
