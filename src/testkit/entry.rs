use super::*;
use crate::op::{EntryQuery, EntryWrite, Vis};

pub(super) async fn idempotent_start_checks_input<S>(store: &S) -> Result<(), MachineError>
where
    S: RunTx<TestEvent, Scope = Value, FinishData = ()>,
{
    let scope = scope("input");
    let mut first = run_start(
        "contract-input-a",
        "contract-input-session",
        "contract-input-thread-a",
        scope.clone(),
    );
    first.client_run_key = Some("input-key".to_string());
    first.input = Some(json!({"message": "same"}));
    first.entries = vec![entry(
        "msg:user:1",
        "msg",
        Vis::Public,
        json!({"text": "same"}),
    )];
    assert!(matches!(
        store.start_run(&first).await?,
        StoreStartResult::Created
    ));

    let mut same = run_start(
        "contract-input-b",
        "contract-input-session",
        "contract-input-thread-b",
        scope.clone(),
    );
    same.client_run_key = Some("input-key".to_string());
    same.input = Some(json!({"message": "same"}));
    let existing = store.start_run(&same).await?;
    let StoreStartResult::Existing(existing) = existing else {
        panic!("same idempotent input must return existing run");
    };
    assert_eq!(existing.run_id, first.run_id);

    let mut conflicting_entry = same.clone();
    conflicting_entry.run_id = RunId::from("contract-input-entry-conflict");
    conflicting_entry.entries = vec![entry(
        "msg:user:1",
        "msg",
        Vis::Public,
        json!({"text": "changed"}),
    )];
    assert!(matches!(
        store.start_run(&conflicting_entry).await,
        Err(MachineError::EntryConflict)
    ));

    let mut different = same;
    different.run_id = RunId::from("contract-input-c");
    different.input = Some(json!({"message": "different"}));
    assert!(matches!(
        store.start_run(&different).await,
        Err(MachineError::InputConflict)
    ));
    Ok(())
}

pub(super) async fn entries_are_session_scoped_and_private<S>(store: &S) -> Result<(), MachineError>
where
    S: RunTx<TestEvent, Scope = Value, FinishData = ()>,
{
    let scope_a = scope("entries");
    let other = scope("entries-other");
    let run_id = RunId::from("contract-entry-run");
    let session_id = SessionId::from("contract-entry-session");
    let thread_id = ThreadId::from("contract-entry-thread");
    let mut start = run_start(
        run_id.as_str(),
        session_id.as_str(),
        thread_id.as_str(),
        scope_a.clone(),
    );
    start.entries = vec![entry(
        "msg:user:1",
        "msg",
        Vis::Public,
        json!({"role": "user", "text": "hello"}),
    )];
    assert!(matches!(
        store.start_run(&start).await?,
        StoreStartResult::Created
    ));

    let public = store
        .list_entries(
            EntryQuery::new(&scope_a, &session_id, 8)
                .thread(&thread_id)
                .kind("msg")
                .vis(Vis::Public),
        )
        .await?;
    assert_eq!(public.len(), 1);
    assert_eq!(public[0].seq, 1);
    assert_eq!(public[0].body["role"], "user");
    assert!(
        store
            .latest_entry(
                &other,
                &session_id,
                Some(&thread_id),
                "msg",
                Some(Vis::Public),
            )
            .await?
            .is_none()
    );

    let private = RunCommit {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        scope: scope_a.clone(),
        lease: None,
        checkpoint: None,
        events: vec![event(run_id.as_str(), session_id.as_str(), 1, false)],
        effects: Vec::new(),
        items: Vec::new(),
        entries: vec![entry(
            "trace:1",
            "trace",
            Vis::Internal,
            json!({"private": true}),
        )],
        finish: None,
    };
    assert!(matches!(
        store.commit_run(&private).await?,
        RunCommitResult::Recorded(_)
    ));
    assert_eq!(
        store
            .list_entries(EntryQuery::new(&scope_a, &session_id, 8).vis(Vis::Public))
            .await?
            .len(),
        1
    );
    let latest_private = store
        .latest_entry(
            &scope_a,
            &session_id,
            Some(&thread_id),
            "trace",
            Some(Vis::Internal),
        )
        .await?
        .expect("internal trace");
    assert_eq!(latest_private.seq, 2);

    let bad = RunCommit {
        events: vec![event(run_id.as_str(), session_id.as_str(), 2, false)],
        entries: vec![entry(
            "trace:1",
            "trace",
            Vis::Internal,
            json!({"private": false}),
        )],
        ..private.clone()
    };
    assert!(matches!(
        store.commit_run(&bad).await,
        Err(MachineError::EntryConflict)
    ));
    assert_eq!(store.list_events(&run_id, &scope_a, 0, 8).await?.len(), 1);

    let finish = RunFinish {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        scope: scope_a.clone(),
        status: RunStatus::Completed,
        finish_reason: "done".to_string(),
        error_code: None,
        entries: Vec::new(),
        data: (),
    };
    let done = RunCommit {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        scope: scope_a.clone(),
        lease: None,
        checkpoint: None,
        events: vec![event(run_id.as_str(), session_id.as_str(), 2, true)],
        effects: Vec::new(),
        items: Vec::new(),
        entries: vec![
            entry(
                "msg:assistant:1",
                "msg",
                Vis::Public,
                json!({"role": "assistant", "text": "done"}),
            ),
            entry(
                "snapshot:1",
                "semantic_snapshot",
                Vis::Internal,
                json!({"durable_context": {"last_turn": "done"}}),
            ),
        ],
        finish: Some(finish),
    };
    assert!(matches!(
        store.commit_run(&done).await?,
        RunCommitResult::Finished { .. }
    ));
    assert!(matches!(
        store.commit_run(&done).await?,
        RunCommitResult::Finished {
            result: FinishRunResult::AlreadyFinished(_),
            ..
        }
    ));
    assert_eq!(
        store
            .list_entries(
                EntryQuery::new(&scope_a, &session_id, 8)
                    .kind("msg")
                    .vis(Vis::Public),
            )
            .await?
            .len(),
        2
    );
    let snapshot = store
        .latest_entry(
            &scope_a,
            &session_id,
            Some(&thread_id),
            "semantic_snapshot",
            Some(Vis::Internal),
        )
        .await?
        .expect("snapshot");
    assert_eq!(snapshot.seq, 4);
    assert_eq!(store.list_events(&run_id, &scope_a, 0, 8).await?.len(), 2);
    Ok(())
}

fn entry(key: &str, kind: &str, vis: Vis, body: Value) -> EntryWrite {
    EntryWrite {
        key: key.to_string(),
        kind: kind.to_string(),
        vis,
        body,
    }
}
