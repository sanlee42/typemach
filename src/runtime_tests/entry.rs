use super::*;

#[test]
fn tx_runtime_records_entry_before_step_finishes() {
    block_on(async {
        let rt = tx_runtime(8);
        let run_id = RunId::from("run-tx-progress");
        let result = rt
            .stream(
                request(run_id.as_str(), Mode::Progress),
                start(None),
                StreamConfig {
                    heartbeat_interval: Duration::from_secs(30),
                    channel_capacity: 32,
                },
            )
            .await
            .expect("stream");
        let StartResult::Started(mut rx) = result else {
            panic!("expected started");
        };

        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let event = rx
                .next_event_timeout(deadline)
                .await
                .expect("progress step should start");
            if matches!(
                event,
                RunStreamEvent::StepStarted {
                    step: Step::Start,
                    ..
                }
            ) {
                break;
            }
        }

        let entry = loop {
            let page = rt
                .store()
                .list_entries(
                    EntryQuery::new(&scope(), &SessionId::from("session-1"), 8)
                        .thread(&ThreadId::from("thread-run-tx-progress"))
                        .kind("progress")
                        .vis(Vis::Public),
                )
                .await
                .expect("entries");
            if let Some(entry) = page.items.into_iter().next() {
                break entry;
            }
            assert!(Instant::now() < deadline);
            async_rt::time::sleep(Duration::from_millis(10)).await;
        };
        assert_eq!(entry.key, "progress-1");
        assert_eq!(entry.body, json!({"stage": "started"}));

        assert!(rt.cancel(&run_id, &scope()).await.expect("cancel"));
        while let Some(event) = rx.next_event_timeout(deadline).await {
            if matches!(event, RunStreamEvent::Cancelled) {
                break;
            }
        }
    });
}

#[test]
fn record_entry_checks_lease_and_conflicts() {
    block_on(async {
        let store = MemoryRunStore::<Event>::new();
        let run_id = RunId::from("run-record-entry");
        let session_id = SessionId::from("session-record-entry");
        let lease_id = LeaseId::from("lease-record-entry");
        store
            .start_run(&RunStart {
                run_id: run_id.clone(),
                session_id: session_id.clone(),
                thread_id: ThreadId::from("thread-record-entry"),
                agent_kind: "test".to_string(),
                model: None,
                client_run_key: None,
                parent_run_id: None,
                retry_of_run_id: None,
                scope: scope(),
                metadata: json!({}),
                input: None,
                entries: Vec::new(),
                lease: Some(LeaseClaim::new(
                    WorkerId::from("worker-record-entry"),
                    lease_id.clone(),
                    Duration::from_secs(30),
                )),
            })
            .await
            .expect("start");

        let wrong_lease = store
            .record_entry(
                &run_id,
                &scope(),
                Some(&LeaseId::from("wrong-lease")),
                entry_write("progress-store", json!({"n": 1})),
            )
            .await
            .expect_err("wrong lease");
        assert!(matches!(wrong_lease, MachineError::LeaseLost));

        let first = store
            .record_entry(
                &run_id,
                &scope(),
                Some(&lease_id),
                entry_write("progress-store", json!({"n": 1})),
            )
            .await
            .expect("record");
        assert_eq!(first.seq, 1);

        let duplicate = store
            .record_entry(
                &run_id,
                &scope(),
                Some(&lease_id),
                entry_write("progress-store", json!({"n": 1})),
            )
            .await
            .expect("duplicate");
        assert_eq!(duplicate.seq, first.seq);

        let conflict = store
            .record_entry(
                &run_id,
                &scope(),
                Some(&lease_id),
                entry_write("progress-store", json!({"n": 2})),
            )
            .await
            .expect_err("conflict");
        assert!(matches!(conflict, MachineError::EntryConflict));
    });
}

#[test]
fn record_entry_rejects_terminal_run() {
    block_on(async {
        let rt = tx_runtime(8);
        let run_id = RunId::from("run-record-terminal");
        let result = rt
            .invoke(request(run_id.as_str(), Mode::Complete), start(None))
            .await
            .expect("invoke");
        assert!(matches!(
            result,
            StartResult::Started(RunOutput::Completed { .. })
        ));

        let error = rt
            .store()
            .record_entry(
                &run_id,
                &scope(),
                None,
                entry_write("progress-late", json!({"n": 1})),
            )
            .await
            .expect_err("terminal run");
        assert!(matches!(error, MachineError::RunNotFound));
    });
}
