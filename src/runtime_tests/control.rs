use super::*;

#[test]
fn cancel_stops_active_run_and_records_cancelled() {
    block_on(async {
        let rt = runtime(8);
        let run_id = RunId::from("run-cancel");
        let result = rt
            .stream(
                request(run_id.as_str(), Mode::Slow),
                start(None),
                StreamConfig::default(),
            )
            .await
            .expect("stream");
        let StartResult::Started(mut rx) = result else {
            panic!("expected started");
        };

        assert!(rt.cancel(&run_id, &scope()).await.expect("cancel"));
        let mut cancelled = false;
        let deadline = Instant::now() + Duration::from_secs(1);
        while let Some(event) = rx.next_event_timeout(deadline).await {
            if matches!(event, RunStreamEvent::Cancelled) {
                cancelled = true;
                break;
            }
        }
        assert!(cancelled);
        let lookup = rt
            .store()
            .lookup_run(&run_id, &scope())
            .await
            .expect("lookup")
            .expect("run");
        assert_eq!(lookup.status, RunStatus::Cancelled);
    });
}

#[test]
fn dropped_receiver_cancels_and_finishes_run() {
    block_on(async {
        let rt = runtime(8);
        let run_id = RunId::from("run-drop");
        let result = rt
            .stream(
                request(run_id.as_str(), Mode::Slow),
                start(None),
                StreamConfig::default(),
            )
            .await
            .expect("stream");
        let StartResult::Started(rx) = result else {
            panic!("expected started");
        };
        drop(rx);

        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Some(lookup) = rt
                .store()
                .lookup_run(&run_id, &scope())
                .await
                .expect("lookup")
                && lookup.status == RunStatus::Cancelled
            {
                break;
            }
            assert!(Instant::now() < deadline);
            async_rt::time::sleep(Duration::from_millis(10)).await;
        }
    });
}

#[test]
fn dropped_receiver_cancels_quiet_transition_without_waiting_for_heartbeat() {
    block_on(async {
        let rt = runtime(8);
        let run_id = RunId::from("run-drop-quiet");
        let result = rt
            .stream(
                request(run_id.as_str(), Mode::Slow),
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
                .expect("quiet step should start");
            if matches!(
                event,
                RunStreamEvent::StepStarted {
                    step: Step::Slow,
                    ..
                }
            ) {
                break;
            }
        }
        drop(rx);

        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Some(lookup) = rt
                .store()
                .lookup_run(&run_id, &scope())
                .await
                .expect("lookup")
                && lookup.status == RunStatus::Cancelled
            {
                break;
            }
            assert!(Instant::now() < deadline);
            async_rt::time::sleep(Duration::from_millis(10)).await;
        }
    });
}

#[test]
fn idempotent_key_returns_existing_run() {
    block_on(async {
        let rt = runtime(8);
        let first_id = RunId::from("run-idem-1");
        let result = rt
            .invoke(
                request(first_id.as_str(), Mode::Complete),
                start(Some("key-1")),
            )
            .await
            .expect("first invoke");
        assert!(matches!(result, StartResult::Started(_)));

        let mut req = request("run-idem-2", Mode::Complete);
        req.thread_id = crate::run::ThreadId::from("thread-run-idem-1");
        let result = rt
            .invoke(req, start(Some("key-1")))
            .await
            .expect("second invoke");
        let StartResult::Existing(existing) = result else {
            panic!("expected existing");
        };
        assert_eq!(existing.run_id, first_id);
    });
}

#[test]
fn idempotent_key_rejects_different_request_input() {
    block_on(async {
        let rt = runtime(8);
        let result = rt
            .invoke(
                request("run-idem-input-1", Mode::Complete),
                start(Some("key-input")),
            )
            .await
            .expect("first invoke");
        assert!(matches!(result, StartResult::Started(_)));

        let mut req = request("run-idem-input-2", Mode::Slow);
        req.thread_id = crate::run::ThreadId::from("thread-run-idem-input-1");
        let result = rt.invoke(req, start(Some("key-input"))).await;
        assert!(matches!(result, Err(MachineError::StartConflict)));
    });
}

#[test]
fn tx_runtime_idempotent_key_rejects_different_request_input() {
    block_on(async {
        let rt = tx_runtime(8);
        let result = rt
            .invoke(
                request("run-tx-idem-input-1", Mode::Complete),
                start(Some("tx-key-input")),
            )
            .await
            .expect("first invoke");
        assert!(matches!(result, StartResult::Started(_)));

        let mut req = request("run-tx-idem-input-2", Mode::Slow);
        req.thread_id = crate::run::ThreadId::from("thread-run-tx-idem-input-1");
        let result = rt.invoke(req, start(Some("tx-key-input"))).await;
        assert!(matches!(result, Err(MachineError::StartConflict)));
    });
}

#[test]
fn capacity_rejection_does_not_persist_or_poison_key() {
    block_on(async {
        let rt = runtime(1);
        let blocker_id = RunId::from("run-capacity-blocker");
        let result = rt
            .stream(
                request(blocker_id.as_str(), Mode::Slow),
                start(None),
                StreamConfig::default(),
            )
            .await
            .expect("stream");
        let StartResult::Started(mut blocker_rx) = result else {
            panic!("expected blocker to start");
        };

        let rejected_id = RunId::from("run-capacity-rejected");
        let result = rt
            .stream(
                request(rejected_id.as_str(), Mode::Complete),
                start(Some("capacity-key")),
                StreamConfig::default(),
            )
            .await
            .expect("stream");
        assert!(matches!(
            result,
            StartResult::Rejected(StartRunRejection::CapacityExceeded)
        ));

        let lookup = rt
            .store()
            .lookup_run(&rejected_id, &scope())
            .await
            .expect("lookup");
        assert!(lookup.is_none());

        assert!(rt.cancel(&blocker_id, &scope()).await.expect("cancel"));
        let deadline = Instant::now() + Duration::from_secs(1);
        while let Some(event) = blocker_rx.next_event_timeout(deadline).await {
            if matches!(event, RunStreamEvent::Cancelled) {
                break;
            }
        }

        let retry_id = RunId::from("run-capacity-retry");
        let result = rt
            .invoke(
                request(retry_id.as_str(), Mode::Complete),
                start(Some("capacity-key")),
            )
            .await
            .expect("retry");
        assert!(matches!(result, StartResult::Started(_)));
    });
}

#[test]
fn subscribe_respects_scope() {
    block_on(async {
        let rt = runtime(8);
        let run_id = RunId::from("run-scope");
        let result = rt
            .invoke(request(run_id.as_str(), Mode::Complete), start(None))
            .await
            .expect("invoke");
        assert!(matches!(result, StartResult::Started(_)));

        let wrong_scope = json!({"tenant": "other"});
        let sub = rt
            .subscribe(&run_id, &wrong_scope, 0)
            .await
            .expect("subscribe");
        assert!(matches!(sub, RunSubscription::Missing));
    });
}
