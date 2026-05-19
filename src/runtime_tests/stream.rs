use super::*;

#[test]
fn stream_persists_and_forwards_completed_events() {
    block_on(async {
        let rt = runtime(8);
        let run_id = RunId::from("run-stream");
        let result = rt
            .stream(
                request(run_id.as_str(), Mode::Complete),
                start(None),
                StreamConfig::default(),
            )
            .await
            .expect("stream");
        let StartResult::Started(mut rx) = result else {
            panic!("expected started");
        };

        let mut completed = None;
        while let Some(event) = rx.next_event().await {
            if let RunStreamEvent::Completed { output, .. } = event {
                completed = Some(output);
                break;
            }
        }

        assert_eq!(completed.as_deref(), Some("value=1"));
        let events = rt
            .store()
            .list_events(&run_id, &scope(), 0)
            .await
            .expect("events");
        assert!(matches!(
            events.first().map(|event| &event.payload),
            Some(Payload::Start { .. })
        ));
        assert!(matches!(
            events.last().map(|event| &event.payload),
            Some(Payload::Done { .. })
        ));
        let lookup = rt
            .store()
            .lookup_run(&run_id, &scope())
            .await
            .expect("lookup")
            .expect("run");
        assert_eq!(lookup.status, RunStatus::Completed);
    });
}

#[test]
fn invoke_returns_output_and_records_events() {
    block_on(async {
        let rt = runtime(8);
        let run_id = RunId::from("run-invoke");
        let result = rt
            .invoke(request(run_id.as_str(), Mode::Complete), start(None))
            .await
            .expect("invoke");
        let StartResult::Started(output) = result else {
            panic!("expected started");
        };
        match output {
            RunOutput::Completed { output, .. } => assert_eq!(output, "value=1"),
            RunOutput::Interrupted { .. } => panic!("expected completed"),
        }
        let events = rt
            .store()
            .list_events(&run_id, &scope(), 0)
            .await
            .expect("events");
        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload, Payload::Signal { .. }))
        );
    });
}

#[test]
fn tx_runtime_commits_final_checkpoint_with_terminal_events() {
    block_on(async {
        let rt = tx_runtime(8);
        let run_id = RunId::from("run-tx-complete");
        let result = rt
            .invoke(request(run_id.as_str(), Mode::Complete), start(None))
            .await
            .expect("invoke");
        assert!(matches!(
            result,
            StartResult::Started(RunOutput::Completed { .. })
        ));

        let events = rt
            .store()
            .list_events(&run_id, &scope(), 0)
            .await
            .expect("events");
        let step_done = events
            .iter()
            .position(|event| {
                matches!(
                    event.payload,
                    Payload::StepDone {
                        result: StepResult::Complete,
                        ..
                    }
                )
            })
            .expect("final step event");
        assert!(matches!(
            events.get(step_done + 1).map(|event| &event.payload),
            Some(Payload::Done { .. })
        ));

        let thread_id = format!("thread-{run_id}");
        let checkpoint = rt
            .store()
            .load(thread_id.as_str())
            .await
            .expect("load checkpoint")
            .expect("checkpoint");
        assert_eq!(checkpoint.run_id.as_deref(), Some(run_id.as_str()));
        assert!(checkpoint.next_step.is_none());
        let lookup = rt
            .store()
            .lookup_run(&run_id, &scope())
            .await
            .expect("lookup")
            .expect("run");
        assert_eq!(lookup.status, RunStatus::Completed);
    });
}

#[test]
fn tx_runtime_commits_interrupt_checkpoint_with_terminal_events() {
    block_on(async {
        let rt = tx_runtime(8);
        let run_id = RunId::from("run-tx-interrupt");
        let result = rt
            .invoke(request(run_id.as_str(), Mode::Interrupt), start(None))
            .await
            .expect("invoke");
        assert!(matches!(
            result,
            StartResult::Started(RunOutput::Interrupted { .. })
        ));

        let events = rt
            .store()
            .list_events(&run_id, &scope(), 0)
            .await
            .expect("events");
        let step_done = events
            .iter()
            .position(|event| {
                matches!(
                    event.payload,
                    Payload::StepDone {
                        result: StepResult::Interrupt,
                        ..
                    }
                )
            })
            .expect("interrupt step event");
        assert!(matches!(
            events.get(step_done + 1).map(|event| &event.payload),
            Some(Payload::Interrupt { .. })
        ));

        let thread_id = format!("thread-{run_id}");
        let checkpoint = rt
            .store()
            .load(thread_id.as_str())
            .await
            .expect("load checkpoint")
            .expect("checkpoint");
        assert_eq!(checkpoint.run_id.as_deref(), Some(run_id.as_str()));
        assert!(checkpoint.interrupt.is_some());
        assert!(checkpoint.interrupted_step.is_some());
        let lookup = rt
            .store()
            .lookup_run(&run_id, &scope())
            .await
            .expect("lookup")
            .expect("run");
        assert_eq!(lookup.status, RunStatus::Interrupted);
    });
}

#[test]
fn tx_runtime_reaps_stale_leased_runs() {
    block_on(async {
        let rt = tx_runtime(8);
        let run_id = RunId::from("run-stale");
        let session_id = SessionId::from("session-stale");
        rt.store()
            .start_run(&RunStart {
                run_id: run_id.clone(),
                session_id: session_id.clone(),
                thread_id: ThreadId::from("thread-stale"),
                agent_kind: "test".to_string(),
                model: None,
                client_run_key: None,
                parent_run_id: None,
                retry_of_run_id: None,
                scope: scope(),
                metadata: json!({}),
                lease: Some(LeaseClaim::new(
                    WorkerId::from("worker-stale"),
                    LeaseId::from("lease-stale"),
                    Duration::from_millis(1),
                )),
            })
            .await
            .expect("start");
        async_rt::time::sleep(Duration::from_millis(5)).await;

        let reaped = rt.reap(8).await.expect("reap");
        assert_eq!(reaped.len(), 1);
        assert_eq!(reaped[0].run_id, run_id);
        assert_eq!(reaped[0].status, RunStatus::Error);

        let events = rt
            .store()
            .list_events(&RunId::from("run-stale"), &scope(), 0)
            .await
            .expect("events");
        assert!(matches!(
            events.last().map(|event| &event.payload),
            Some(Payload::Fail { .. })
        ));
    });
}

#[test]
fn interrupted_run_gets_interrupted_status() {
    block_on(async {
        let rt = runtime(8);
        let run_id = RunId::from("run-interrupt");
        let result = rt
            .invoke(request(run_id.as_str(), Mode::Interrupt), start(None))
            .await
            .expect("invoke");
        let StartResult::Started(output) = result else {
            panic!("expected started");
        };
        match output {
            RunOutput::Interrupted { interrupt, .. } => assert_eq!(interrupt, "answer?"),
            RunOutput::Completed { .. } => panic!("expected interrupt"),
        }
        let lookup = rt
            .store()
            .lookup_run(&run_id, &scope())
            .await
            .expect("lookup")
            .expect("run");
        assert_eq!(lookup.status, RunStatus::Interrupted);
    });
}

#[test]
fn failed_run_records_error_terminal() {
    block_on(async {
        let rt = runtime(8);
        let run_id = RunId::from("run-fail");
        let mut req = request(run_id.as_str(), Mode::Loop);
        req.runtime_limits.max_steps = 1;
        let err = rt
            .invoke(req, start(None))
            .await
            .expect_err("invoke should fail");
        assert!(matches!(err, MachineError::MaxStepsExceeded { max: 1 }));

        let lookup = rt
            .store()
            .lookup_run(&run_id, &scope())
            .await
            .expect("lookup")
            .expect("run");
        assert_eq!(lookup.status, RunStatus::Error);
        let terminal = rt
            .store()
            .terminal_event(&run_id, &scope())
            .await
            .expect("terminal")
            .expect("terminal event");
        assert!(matches!(terminal.payload, Payload::Fail { .. }));
    });
}
