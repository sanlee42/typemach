use super::*;

#[test]
fn lifecycle_subscribe_returns_terminal_replay_when_inactive() {
    block_on(async {
        let lifecycle = lifecycle();
        lifecycle
            .start_run(
                run_start("run-a", None),
                RunHandle::new("token".to_string()),
                None,
            )
            .await
            .expect("start");
        lifecycle
            .finish_run(finish_request("run-a", RunStatus::Completed), payload(true))
            .await
            .expect("finish");

        let RunSubscription::Inactive { status, replay } = lifecycle
            .subscribe(&RunId::from("run-a"), &scope(), 0)
            .await
            .expect("subscribe")
        else {
            panic!("expected inactive subscription");
        };
        assert_eq!(status, RunStatus::Completed);
        assert_eq!(replay.len(), 1);
        assert!(replay[0].is_terminal());
    });
}

#[test]
fn lifecycle_subscribe_reports_missing_run() {
    block_on(async {
        let lifecycle = lifecycle();
        assert!(matches!(
            lifecycle
                .subscribe(&RunId::from("missing"), &scope(), 0)
                .await
                .expect("subscribe"),
            RunSubscription::Missing
        ));
    });
}

#[test]
fn lifecycle_subscribe_quick_fails_remote_running_run() {
    block_on(async {
        let store = Arc::new(MemoryRunStore::<TestEvent>::new());
        let owner = RunLifecycle::new(RunRegistry::new(), Arc::clone(&store), 8);
        owner
            .start_run(
                leased_run_start("run-a", "worker-a"),
                RunHandle::new("token".to_string()),
                None,
            )
            .await
            .expect("start");
        let peer = RunLifecycle::new(RunRegistry::new(), store, 8);

        let err = peer
            .subscribe(&RunId::from("run-a"), &scope(), 0)
            .await
            .expect_err("remote live subscribe must fail");
        assert!(matches!(
            err,
            MachineError::NotOwner {
                owner: Some(owner),
            } if owner == WorkerId::from("worker-a")
        ));
    });
}

#[test]
fn lifecycle_request_cancel_marks_registry_and_store() {
    block_on(async {
        let lifecycle = lifecycle();
        lifecycle
            .start_run(
                run_start("run-a", None),
                RunHandle::new("token".to_string()),
                None,
            )
            .await
            .expect("start");
        let handle = lifecycle
            .request_cancel(&RunId::from("run-a"), &scope())
            .await
            .expect("cancel")
            .expect("active handle");
        assert!(handle.is_cancelled());
        assert!(
            lifecycle
                .store()
                .lookup_run(&RunId::from("run-a"), &scope())
                .await
                .expect("lookup")
                .expect("run")
                .cancel_requested
        );
    });
}

#[test]
fn lifecycle_request_cancel_rejects_wrong_scope_without_cancelling_handle() {
    block_on(async {
        let lifecycle = lifecycle();
        let handle = RunHandle::new("token".to_string());
        lifecycle
            .start_run(run_start("run-a", None), handle.clone(), None)
            .await
            .expect("start");

        let wrong_scope = serde_json::json!({"tenant": "other"});
        let err = lifecycle
            .request_cancel(&RunId::from("run-a"), &wrong_scope)
            .await
            .expect_err("wrong scope must not cancel");
        assert!(matches!(err, MachineError::RunNotFound));
        assert!(!handle.is_cancelled());
        assert!(
            !lifecycle
                .store()
                .lookup_run(&RunId::from("run-a"), &scope())
                .await
                .expect("lookup")
                .expect("run")
                .cancel_requested
        );
    });
}

#[test]
fn lifecycle_request_cancel_quick_fails_remote_running_run() {
    block_on(async {
        let store = Arc::new(MemoryRunStore::<TestEvent>::new());
        let owner = RunLifecycle::new(RunRegistry::new(), Arc::clone(&store), 8);
        let handle = RunHandle::new("token".to_string());
        owner
            .start_run(leased_run_start("run-a", "worker-a"), handle.clone(), None)
            .await
            .expect("start");
        let peer = RunLifecycle::new(RunRegistry::new(), Arc::clone(&store), 8);

        let err = peer
            .request_cancel(&RunId::from("run-a"), &scope())
            .await
            .expect_err("remote cancel must fail");
        assert!(matches!(
            err,
            MachineError::NotOwner {
                owner: Some(owner),
            } if owner == WorkerId::from("worker-a")
        ));
        assert!(!handle.is_cancelled());
        assert!(
            !store
                .lookup_run(&RunId::from("run-a"), &scope())
                .await
                .expect("lookup")
                .expect("run")
                .cancel_requested
        );
    });
}

#[test]
fn lifecycle_start_run_returns_idempotent_existing_without_registry_insert() {
    block_on(async {
        let lifecycle = RunLifecycle::new(
            RunRegistry::new(),
            Arc::new(MemoryRunStore::<TestEvent>::new()),
            1,
        );
        assert!(matches!(
            lifecycle
                .start_run(
                    run_start("run-a", Some("client-key")),
                    RunHandle::new("token-a".to_string()),
                    None,
                )
                .await
                .expect("first"),
            StartRunResult::Started
        ));
        let mut retry = run_start("run-b", Some("client-key"));
        retry.thread_id = ThreadId::from("thread-run-a");
        match lifecycle
            .start_run(retry, RunHandle::new("token-b".to_string()), None)
            .await
            .expect("second")
        {
            StartRunResult::Existing(existing) => {
                assert_eq!(existing.run_id, RunId::from("run-a"));
            }
            StartRunResult::Started | StartRunResult::NotRegistered(_) => {
                panic!("expected idempotent existing run")
            }
        }
        assert_eq!(lifecycle.registry().len(), 1);
    });
}

#[test]
fn lifecycle_start_run_does_not_persist_when_registry_is_full() {
    block_on(async {
        let lifecycle = RunLifecycle::new(
            RunRegistry::new(),
            Arc::new(MemoryRunStore::<TestEvent>::new()),
            0,
        );

        assert!(matches!(
            lifecycle
                .start_run(
                    run_start("run-a", None),
                    RunHandle::new("token".to_string()),
                    None,
                )
                .await
                .expect("start"),
            StartRunResult::NotRegistered(StartRunRejection::CapacityExceeded)
        ));

        let lookup = lifecycle
            .store()
            .lookup_run(&RunId::from("run-a"), &scope())
            .await
            .expect("lookup");
        assert!(lookup.is_none());
        assert_eq!(lifecycle.registry().len(), 0);
    });
}

#[test]
fn lifecycle_finish_detached_completes_stored_unregistered_run() {
    block_on(async {
        let lifecycle = RunLifecycle::new(
            RunRegistry::new(),
            Arc::new(MemoryRunStore::<TestEvent>::new()),
            0,
        );
        lifecycle
            .store()
            .start_run(&run_start("run-a", None))
            .await
            .expect("start");

        let result = lifecycle
            .finish_detached(finish_request("run-a", RunStatus::Error), payload(true))
            .await
            .expect("detached finish");
        assert!(matches!(result, FinishRunResult::Finished(_)));
        assert_eq!(result.terminal_event().seq, 1);

        let lookup = lifecycle
            .store()
            .lookup_run(&RunId::from("run-a"), &scope())
            .await
            .expect("lookup")
            .expect("stored run");
        assert_eq!(lookup.status, RunStatus::Error);
    });
}

#[test]
fn lifecycle_finish_detached_uses_last_stored_event_seq() {
    block_on(async {
        let lifecycle = RunLifecycle::new(
            RunRegistry::new(),
            Arc::new(MemoryRunStore::<TestEvent>::new()),
            0,
        );
        lifecycle
            .store()
            .start_run(&run_start("run-a", None))
            .await
            .expect("start");
        lifecycle
            .store()
            .record_event(&RunId::from("run-a"), &scope(), &event("run-a", 1, false))
            .await
            .expect("record");

        let result = lifecycle
            .finish_detached(finish_request("run-a", RunStatus::Completed), payload(true))
            .await
            .expect("detached finish");
        assert_eq!(result.terminal_event().seq, 2);
    });
}

#[test]
fn lifecycle_finish_detached_rejects_wrong_scope_before_active_check() {
    block_on(async {
        let lifecycle = lifecycle();
        lifecycle
            .start_run(
                run_start("run-a", None),
                RunHandle::new("token".to_string()),
                None,
            )
            .await
            .expect("start");

        let mut finish = finish_request("run-a", RunStatus::Completed);
        finish.scope = serde_json::json!({"tenant": "other"});
        let err = lifecycle
            .finish_detached(finish, payload(true))
            .await
            .expect_err("wrong scope must be hidden");
        assert!(matches!(err, MachineError::RunNotFound));
    });
}

#[test]
fn lifecycle_finish_detached_rejects_active_run() {
    block_on(async {
        let lifecycle = lifecycle();
        lifecycle
            .start_run(
                run_start("run-a", None),
                RunHandle::new("token".to_string()),
                None,
            )
            .await
            .expect("start");

        let err = lifecycle
            .finish_detached(finish_request("run-a", RunStatus::Completed), payload(true))
            .await
            .expect_err("active run must use registry finish");
        assert!(matches!(err, MachineError::RunAlreadyActive));
    });
}
