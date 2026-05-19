use super::*;

#[test]
fn lifecycle_appends_persists_and_publishes_event() {
    block_on(async {
        let lifecycle = lifecycle();
        let (sender, mut receiver) = mpsc::unbounded_channel();
        lifecycle
            .start_run(
                run_start("run-a", None),
                RunHandle::new("token".to_string()),
                Some(sender),
            )
            .await
            .expect("start");

        let result = lifecycle
            .append_event(
                &RunId::from("run-a"),
                &SessionId::from("session-a"),
                &scope(),
                payload(false),
            )
            .await
            .expect("append");
        assert!(matches!(result, AppendEventResult::Recorded(_)));
        assert_eq!(receiver.try_recv().expect("published").seq, 1);

        let events = lifecycle
            .store()
            .list_events(&RunId::from("run-a"), &scope(), 0, usize::MAX)
            .await
            .expect("events");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].seq, 1);
    });
}

#[test]
fn lifecycle_finishes_once_and_blocks_later_appends() {
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
        let terminal = lifecycle
            .finish_run(finish_request("run-a", RunStatus::Completed), payload(true))
            .await
            .expect("finish");
        assert!(terminal.is_finished());
        assert_eq!(terminal.terminal_event().seq, 1);
        assert_eq!(lifecycle.registry().len(), 0);
        let append = lifecycle
            .append_event(
                &RunId::from("run-a"),
                &SessionId::from("session-a"),
                &scope(),
                payload(false),
            )
            .await;
        assert!(matches!(append, Err(MachineError::RunNotFound)));
    });
}

#[test]
fn lifecycle_terminal_seq_follows_last_appended_event() {
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
            .append_event(
                &RunId::from("run-a"),
                &SessionId::from("session-a"),
                &scope(),
                payload(false),
            )
            .await
            .expect("append");

        let terminal = lifecycle
            .finish_run(finish_request("run-a", RunStatus::Completed), payload(true))
            .await
            .expect("finish");
        assert_eq!(terminal.terminal_event().seq, 2);
    });
}

#[test]
fn lifecycle_terminal_releases_capacity() {
    block_on(async {
        let lifecycle = RunLifecycle::new(
            RunRegistry::new(),
            Arc::new(MemoryRunStore::<TestEvent>::new()),
            1,
        );
        lifecycle
            .start_run(
                run_start("run-a", None),
                RunHandle::new("token-a".to_string()),
                None,
            )
            .await
            .expect("start first");
        lifecycle
            .finish_run(finish_request("run-a", RunStatus::Completed), payload(true))
            .await
            .expect("finish");

        assert!(matches!(
            lifecycle
                .start_run(
                    run_start("run-b", None),
                    RunHandle::new("token-b".to_string()),
                    None,
                )
                .await
                .expect("start second"),
            StartRunResult::Started
        ));
    });
}

#[test]
fn lifecycle_repeated_finish_after_registry_cleanup_returns_existing_terminal() {
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
        let first = lifecycle
            .finish_run(finish_request("run-a", RunStatus::Completed), payload(true))
            .await
            .expect("first finish");
        let called = Arc::new(AtomicBool::new(false));

        let second = lifecycle
            .finish_with(finish_request("run-a", RunStatus::Error), {
                let called = Arc::clone(&called);
                move |seq| {
                    called.store(true, Ordering::SeqCst);
                    event("run-a", seq, true)
                }
            })
            .await
            .expect("second finish");

        assert!(matches!(second, FinishRunResult::AlreadyFinished(_)));
        assert_eq!(second.terminal_event(), first.terminal_event());
        assert!(!called.load(Ordering::SeqCst));
    });
}

#[test]
fn lifecycle_invalid_finish_keeps_active_lock_and_rewinds_seq() {
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
            .finish_with(finish_request("run-a", RunStatus::Completed), |seq| {
                RunEventEnvelope::new(
                    RunId::from("run-a"),
                    SessionId::from("wrong-session"),
                    seq,
                    payload(true),
                )
            })
            .await
            .expect_err("invalid terminal event should fail");
        assert!(matches!(err, MachineError::InvalidRunEvent { .. }));
        assert!(
            lifecycle
                .event_locks
                .lock()
                .await
                .contains_key(&RunId::from("run-a")),
            "active run lock should remain registered after validation failure"
        );

        let terminal = lifecycle
            .finish_run(finish_request("run-a", RunStatus::Completed), payload(true))
            .await
            .expect("finish");
        assert_eq!(terminal.terminal_event().seq, 1);
    });
}

#[test]
fn lifecycle_serializes_appends_for_one_run() {
    block_on(async {
        let store = Arc::new(BlockingRecordStore::new(1));
        let lifecycle = RunLifecycle::new(RunRegistry::new(), Arc::clone(&store), 8);
        lifecycle
            .start_run(
                run_start("run-a", None),
                RunHandle::new("token".to_string()),
                None,
            )
            .await
            .expect("start");

        let first_lifecycle = lifecycle.clone();
        let first = async_rt::spawn(async move {
            first_lifecycle
                .append_event(
                    &RunId::from("run-a"),
                    &SessionId::from("session-a"),
                    &scope(),
                    payload(false),
                )
                .await
        });
        async_rt::time::timeout(Duration::from_secs(1), store.blocked.notified())
            .await
            .expect("first append should block in store");

        let second_lifecycle = lifecycle.clone();
        let second = async_rt::spawn(async move {
            second_lifecycle
                .append_event(
                    &RunId::from("run-a"),
                    &SessionId::from("session-a"),
                    &scope(),
                    payload(false),
                )
                .await
        });
        async_rt::time::sleep(Duration::from_millis(20)).await;
        assert!(
            store
                .inner
                .list_events(&RunId::from("run-a"), &scope(), 0, usize::MAX)
                .await
                .expect("events")
                .is_empty()
        );

        store.release.notify_one();
        first
            .await
            .expect("first task")
            .expect("first append should finish");
        second
            .await
            .expect("second task")
            .expect("second append should finish");

        let events = store
            .inner
            .list_events(&RunId::from("run-a"), &scope(), 0, usize::MAX)
            .await
            .expect("events");
        assert_eq!(
            events.iter().map(|event| event.seq).collect::<Vec<_>>(),
            vec![1, 2]
        );
    });
}

#[test]
fn lifecycle_subscribe_replays_then_tails_active_run() {
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
            .append_event(
                &RunId::from("run-a"),
                &SessionId::from("session-a"),
                &scope(),
                payload(false),
            )
            .await
            .expect("append");

        let RunSubscription::Active { replay, mut tail } = lifecycle
            .subscribe(&RunId::from("run-a"), &scope(), 0)
            .await
            .expect("subscribe")
        else {
            panic!("expected active subscription");
        };
        assert_eq!(replay.len(), 1);
        assert_eq!(tail.cursor(), RunCursor::new(1));

        lifecycle
            .append_event(
                &RunId::from("run-a"),
                &SessionId::from("session-a"),
                &scope(),
                payload(false),
            )
            .await
            .expect("append live");
        assert_eq!(tail.next_event().await.expect("live").seq, 2);
    });
}

#[test]
fn lifecycle_subscribe_returns_replay_page_before_live_tail() {
    block_on(async {
        let lifecycle = lifecycle();
        lifecycle
            .start_run(
                run_start("run-replay", None),
                RunHandle::new("token".to_string()),
                None,
            )
            .await
            .expect("start");
        for _ in 0..=crate::lifecycle::REPLAY_LIMIT {
            lifecycle
                .append_event(
                    &RunId::from("run-replay"),
                    &SessionId::from("session-a"),
                    &scope(),
                    payload(false),
                )
                .await
                .expect("append");
        }

        let RunSubscription::Replay { page } = lifecycle
            .subscribe(&RunId::from("run-replay"), &scope(), 0)
            .await
            .expect("subscribe")
        else {
            panic!("expected replay page");
        };
        assert_eq!(page.len(), crate::lifecycle::REPLAY_LIMIT);
        assert_eq!(
            page.cursor(),
            RunCursor::new(crate::lifecycle::REPLAY_LIMIT as i64)
        );

        let RunSubscription::Active { replay, tail } = lifecycle
            .subscribe(&RunId::from("run-replay"), &scope(), page.cursor())
            .await
            .expect("subscribe")
        else {
            panic!("expected active");
        };
        assert_eq!(replay.len(), 1);
        assert_eq!(
            tail.cursor(),
            RunCursor::new(crate::lifecycle::REPLAY_LIMIT as i64 + 1)
        );
    });
}

#[test]
fn run_tail_filters_events_at_or_before_cursor() {
    block_on(async {
        let (sender, receiver) = mpsc::unbounded_channel();
        let mut tail = RunTail::new(receiver, RunCursor::new(1));
        sender.send(event("run-a", 1, false)).expect("send dup");
        sender.send(event("run-a", 2, false)).expect("send fresh");

        assert_eq!(tail.next_event().await.expect("fresh").seq, 2);
        assert_eq!(tail.cursor(), RunCursor::new(2));
    });
}
