use super::*;
use crate::RunEventEnvelope;
use crate::run::{LeaseId, ThreadId};
use crate::store::{CheckpointWrite, LeaseClaim};
use crate::testkit::{TestEvent, TestPayload};
use tokio_rusqlite::rusqlite::StatementStatus;

#[test]
fn sqlite_store_matches_contract() {
    block_on(async {
        let store = SqliteStore::<crate::testkit::TestEvent>::memory()
            .await
            .expect("store");
        store.ensure_schema().await.expect("schema");
        crate::testkit::run_store_contract(&store)
            .await
            .expect("contract");
    });
}

#[test]
fn sqlite_ensure_schema_adds_run_start_columns() {
    block_on(async {
        let store = SqliteStore::<crate::testkit::TestEvent>::memory()
            .await
            .expect("store");
        store
            .call(|conn| {
                conn.execute_batch(
                    "CREATE TABLE typemach_runs (
                        run_id TEXT PRIMARY KEY,
                        scope_key TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        agent_kind TEXT NOT NULL,
                        model TEXT NULL,
                        client_run_key TEXT NULL,
                        parent_run_id TEXT NULL,
                        retry_of_run_id TEXT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        status TEXT NOT NULL,
                        cancel_requested INTEGER NOT NULL DEFAULT 0,
                        started_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                        finished_at INTEGER NULL,
                        finish_reason TEXT NULL,
                        error_code TEXT NULL,
                        finish_data TEXT NULL,
                        owner_id TEXT NULL,
                        lease_id TEXT NULL,
                        lease_expires_at INTEGER NULL,
                        attempt INTEGER NOT NULL DEFAULT 0,
                        created_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000),
                        updated_at INTEGER NOT NULL DEFAULT (unixepoch() * 1000)
                    );",
                )
                .map_err(store_db)?;
                Ok(())
            })
            .await
            .expect("old schema");
        store.ensure_schema().await.expect("schema");
        let (has_input, has_start_sig) = store
            .call(|conn| {
                let mut stmt = conn
                    .prepare("PRAGMA table_info(typemach_runs)")
                    .map_err(store_db)?;
                let rows = stmt
                    .query_map([], |row| row.get::<_, String>(1))
                    .map_err(store_db)?;
                let mut has_input = false;
                let mut has_start_sig = false;
                for row in rows {
                    match row.map_err(store_db)?.as_str() {
                        "input" => has_input = true,
                        "start_sig" => has_start_sig = true,
                        _ => {}
                    }
                }
                Ok((has_input, has_start_sig))
            })
            .await
            .expect("columns");
        assert!(has_input);
        assert!(has_start_sig);

        let run_id = RunId::from("sqlite-upgrade-run");
        let session_id = SessionId::from("sqlite-upgrade-session");
        store
            .start_run(&RunStart {
                run_id: run_id.clone(),
                session_id: session_id.clone(),
                thread_id: ThreadId::from("sqlite-upgrade-thread"),
                agent_kind: "test".to_string(),
                model: None,
                client_run_key: Some("upgrade-key".to_string()),
                parent_run_id: None,
                retry_of_run_id: None,
                scope: serde_json::json!({"tenant": "upgrade"}),
                metadata: serde_json::json!({}),
                input: Some(serde_json::json!({"message": "hello"})),
                entries: Vec::new(),
                lease: None,
            })
            .await
            .expect("start");
        assert!(
            store
                .lookup_run(&run_id, &serde_json::json!({"tenant": "upgrade"}))
                .await
                .expect("lookup")
                .is_some()
        );
    });
}

#[test]
fn ensure_schema_rejects_noncompact_terminal_without_mutation() {
    block_on(async {
        let store = SqliteStore::<crate::runtime::Event>::memory()
            .await
            .expect("store");
        store.ensure_schema().await.expect("schema");
        let snapshot = "x".repeat(1_800 * 1024);
        store
            .call(move |conn| {
                conn.execute(
                    "INSERT INTO typemach_sessions (scope_key, session_id, scope)
                     VALUES ('{}', 'cutover-session', '{}')",
                    [],
                )
                .map_err(store_db)?;
                conn.execute(
                    "INSERT INTO typemach_runs (
                        run_id, scope_key, session_id, thread_id, scope, agent_kind,
                        metadata, start_sig, status
                     ) VALUES (
                        'cutover-run', '{}', 'cutover-session', 'cutover-thread', '{}',
                        'test', '{}', '', 'completed'
                     )",
                    [],
                )
                .map_err(store_db)?;
                let event = serde_json::json!({
                    "run_id": "cutover-run",
                    "session_id": "cutover-session",
                    "seq": 1,
                    "payload": {
                        "type": "done",
                        "trace": [{"old": true}],
                        "output": {"ok": true},
                        "snapshot": {"memory_marker": snapshot}
                    }
                });
                conn.execute(
                    "INSERT INTO typemach_run_events
                     (run_id, session_id, seq, terminal, event)
                     VALUES ('cutover-run', 'cutover-session', 1, 1, ?1)",
                    [event.to_string()],
                )
                .map_err(store_db)?;
                Ok(())
            })
            .await
            .expect("old terminal");

        let error = store
            .ensure_schema()
            .await
            .expect_err("noncompact terminal must fail schema validation");
        assert!(matches!(
            error,
            MachineError::InvalidRunEvent { ref reason }
                if reason == "terminal cutover-run does not match the compact contract"
        ));
        let payload = store
            .call(|conn| {
                conn.query_row(
                    "SELECT event -> '$.payload'
                       FROM typemach_run_events
                      WHERE run_id = 'cutover-run'",
                    [],
                    |row| row.get::<_, String>(0),
                )
                .map_err(store_db)
            })
            .await
            .expect("compact payload");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&payload).expect("payload JSON"),
            serde_json::json!({
                "type": "done",
                "trace": [{"old": true}],
                "output": {"ok": true},
                "snapshot": {"memory_marker": "x".repeat(1_800 * 1024)}
            })
        );

        for (name, invalid) in [
            ("malformed", "{"),
            ("scalar", "7"),
            ("array", "[]"),
            ("empty object", "{}"),
            ("missing envelope", r#"{"payload":{"type":"cancel"}}"#),
            (
                "invalid envelope",
                r#"{"run_id":"cutover-run","session_id":"cutover-session","seq":"1","payload":{"type":"cancel"}}"#,
            ),
        ] {
            let fixture = invalid.to_owned();
            store
                .call(move |conn| {
                    conn.execute(
                        "UPDATE typemach_run_events SET event = ?1 WHERE run_id = 'cutover-run'",
                        [fixture],
                    )
                    .map_err(store_db)?;
                    Ok(())
                })
                .await
                .unwrap_or_else(|error| panic!("{name} terminal fixture: {error}"));
            let error = match store.ensure_schema().await {
                Ok(()) => panic!("{name} terminal must fail schema validation"),
                Err(error) => error,
            };
            assert!(matches!(
                error,
                MachineError::InvalidRunEvent { ref reason }
                    if reason == "terminal cutover-run does not match the compact contract"
            ));
            let raw = store
                .call(|conn| {
                    conn.query_row(
                        "SELECT event FROM typemach_run_events WHERE run_id = 'cutover-run'",
                        [],
                        |row| row.get::<_, String>(0),
                    )
                    .map_err(store_db)
                })
                .await
                .unwrap_or_else(|error| panic!("{name} terminal after rejection: {error}"));
            assert_eq!(raw, invalid, "{name} terminal mutated");
        }

        for valid in [
            r#"{"run_id":"cutover-run","session_id":"cutover-session","seq":1,"payload":{"type":"fail","error":"failed"}}"#,
            r#"{"run_id":"cutover-run","session_id":"cutover-session","seq":1,"payload":{"type":"cancel"}}"#,
        ] {
            let fixture = valid.to_owned();
            store
                .call(move |conn| {
                    conn.execute(
                        "UPDATE typemach_run_events SET event = ?1 WHERE run_id = 'cutover-run'",
                        [fixture],
                    )
                    .map_err(store_db)?;
                    Ok(())
                })
                .await
                .expect("valid terminal fixture");
            store
                .ensure_schema()
                .await
                .expect("valid terminal contract");
        }
    });
}

#[test]
fn ensure_schema_accepts_ten_thousand_exact_compact_terminals() {
    block_on(async {
        let store = SqliteStore::<crate::runtime::Event>::memory()
            .await
            .expect("store");
        store.ensure_schema().await.expect("schema");
        store
            .call(|conn| {
                conn.execute(
                    "INSERT INTO typemach_sessions (scope_key, session_id, scope)
                     VALUES ('{}', 'compact-session', '{}')",
                    [],
                )
                .map_err(store_db)?;
                conn.execute(
                    "WITH RECURSIVE n(x) AS (
                        VALUES(0) UNION ALL SELECT x + 1 FROM n WHERE x < 9999
                     )
                     INSERT INTO typemach_runs (
                        run_id, scope_key, session_id, thread_id, scope, agent_kind,
                        metadata, start_sig, status
                     )
                     SELECT printf('compact-%05d', x), '{}', 'compact-session',
                            printf('compact-thread-%05d', x), '{}', 'test', '{}', '', 'completed'
                       FROM n",
                    [],
                )
                .map_err(store_db)?;
                conn.execute(
                    "INSERT INTO typemach_run_events (run_id, session_id, seq, terminal, event)
                     SELECT run_id, 'compact-session', 1, 1,
                            json_object(
                                'run_id', run_id,
                                'session_id', 'compact-session',
                                'seq', 1,
                                'payload', json_object(
                                    'type', 'done',
                                    'output', json_object('ok', json('true')),
                                    'checkpoint', run_id
                                )
                            )
                       FROM typemach_runs
                      WHERE run_id LIKE 'compact-%'",
                    [],
                )
                .map_err(store_db)?;
                Ok(())
            })
            .await
            .expect("compact terminal");

        store.ensure_schema().await.expect("compact contract");
    });
}

#[test]
fn terminal_lookup_work_is_independent_of_history() {
    block_on(async {
        let store = SqliteStore::<TestEvent>::memory().await.expect("store");
        store.ensure_schema().await.expect("schema");
        let scope = serde_json::json!({"tenant": "tail"});
        let key = scope_key(&scope).expect("scope key");
        let session = SessionId::from("tail-session");
        store
            .ensure_session(Some(session.clone()), &scope)
            .await
            .expect("session");
        seed_run(&store, "tail-run", &session, "tail-thread", &scope).await;
        seed_terminal(&store, "tail-run", &session).await;

        let before = terminal_steps(&store, "tail-run", &key).await;
        let scope_json = serde_json::to_string(&scope).expect("scope JSON");
        let history_key = key.clone();
        store
            .call(move |conn| {
                conn.execute(
                    "WITH RECURSIVE n(x) AS (
                         VALUES(0) UNION ALL SELECT x + 1 FROM n WHERE x < 9999
                     )
                     INSERT INTO typemach_runs (
                         run_id, scope_key, session_id, thread_id, scope, agent_kind,
                         metadata, start_sig, status
                     )
                     SELECT printf('history-%05d', x), ?1, ?2, printf('thread-%05d', x),
                            ?3, 'test', '{}', '', 'completed'
                       FROM n",
                    (&history_key, session.as_str(), &scope_json),
                )
                .map_err(store_db)?;
                conn.execute(
                    "INSERT INTO typemach_run_events (run_id, session_id, seq, terminal, event)
                     SELECT run_id, ?1, 1, 1,
                            json_object(
                                'run_id', run_id,
                                'session_id', ?1,
                                'seq', 1,
                                'payload', json_object('terminal', json('true'), 'name', 'done')
                            )
                       FROM typemach_runs
                      WHERE run_id LIKE 'history-%'",
                    [session.as_str()],
                )
                .map_err(store_db)?;
                Ok(())
            })
            .await
            .expect("history");

        let after = terminal_steps(&store, "tail-run", &key).await;
        assert_eq!(after, before);
        assert!(matches!(
            store
                .terminal_event(&RunId::from("tail-run"), &scope)
                .await
                .expect("terminal")
                .expect("terminal event")
                .payload,
            TestPayload { terminal: true, .. }
        ));
    });
}

#[test]
fn final_commit_rolls_back_or_persists_as_one_fact() {
    block_on(async {
        let store = SqliteStore::<TestEvent>::memory().await.expect("store");
        store.ensure_schema().await.expect("schema");
        let scope = serde_json::json!({"tenant": "atomic"});
        let run = RunId::from("atomic-run");
        let session = SessionId::from("atomic-session");
        let thread = ThreadId::from("atomic-thread");
        let lease = LeaseId::from("atomic-lease");
        let mut start = run_start(&run, &session, &thread, &scope);
        start.lease = Some(LeaseClaim::new(
            WorkerId::from("atomic-worker"),
            lease.clone(),
            Duration::from_secs(30),
        ));
        store.start_run(&start).await.expect("start");
        let commit = terminal_commit(&run, &session, &thread, &scope, &lease);

        store
            .call(|conn| {
                conn.execute_batch(
                    "CREATE TRIGGER fail_final_update
                     BEFORE UPDATE OF status ON typemach_runs
                     WHEN NEW.status = 'completed'
                     BEGIN SELECT RAISE(ABORT, 'crash before commit'); END;",
                )
                .map_err(store_db)?;
                Ok(())
            })
            .await
            .expect("trigger");
        assert!(matches!(
            store.commit_run(&commit).await,
            Err(MachineError::StoreDb(_))
        ));
        assert_eq!(
            atomic_counts(&store).await,
            (0, 0, 1, "running".to_string())
        );

        store
            .call(|conn| {
                conn.execute_batch("DROP TRIGGER fail_final_update")
                    .map_err(store_db)?;
                Ok(())
            })
            .await
            .expect("remove trigger");
        assert!(matches!(
            store.commit_run(&commit).await.expect("commit"),
            RunCommitResult::Finished { .. }
        ));
        assert_eq!(
            atomic_counts(&store).await,
            (1, 1, 0, "completed".to_string())
        );
        let checkpoint = store
            .load_checkpoint(thread.as_str())
            .await
            .expect("load checkpoint")
            .expect("checkpoint");
        assert_eq!(checkpoint.run_id.as_deref(), Some(run.as_str()));
    });
}

async fn seed_run(
    store: &SqliteStore<TestEvent>,
    run: &str,
    session: &SessionId,
    thread: &str,
    scope: &serde_json::Value,
) {
    store
        .start_run(&run_start(
            &RunId::from(run),
            session,
            &ThreadId::from(thread),
            scope,
        ))
        .await
        .expect("start run");
}

async fn seed_terminal(store: &SqliteStore<TestEvent>, run: &str, session: &SessionId) {
    let run = RunId::from(run);
    store
        .finish_run(&RunFinishRecord {
            run_id: run.clone(),
            session_id: session.clone(),
            scope: serde_json::json!({"tenant": "tail"}),
            status: RunStatus::Completed,
            finish_reason: "done".to_string(),
            error_code: None,
            terminal_event: test_event(&run, session, 1),
            entries: Vec::new(),
            data: (),
        })
        .await
        .expect("finish run");
}

fn run_start(
    run: &RunId,
    session: &SessionId,
    thread: &ThreadId,
    scope: &serde_json::Value,
) -> RunStart<serde_json::Value> {
    RunStart {
        run_id: run.clone(),
        session_id: session.clone(),
        thread_id: thread.clone(),
        agent_kind: "test".to_string(),
        model: None,
        client_run_key: None,
        parent_run_id: None,
        retry_of_run_id: None,
        scope: scope.clone(),
        metadata: serde_json::json!({}),
        input: None,
        entries: Vec::new(),
        lease: None,
    }
}

fn test_event(run: &RunId, session: &SessionId, seq: i64) -> TestEvent {
    RunEventEnvelope::new(
        run.clone(),
        session.clone(),
        seq,
        TestPayload {
            terminal: true,
            name: "done".to_string(),
        },
    )
}

fn terminal_commit(
    run: &RunId,
    session: &SessionId,
    thread: &ThreadId,
    scope: &serde_json::Value,
    lease: &LeaseId,
) -> RunCommit<TestEvent> {
    RunCommit {
        run_id: run.clone(),
        session_id: session.clone(),
        scope: scope.clone(),
        lease: Some(lease.clone()),
        checkpoint: Some(CheckpointWrite::new(
            thread.clone(),
            CheckpointRecord::running(serde_json::json!({"state": "final"}), None, run.as_str()),
        )),
        events: vec![test_event(run, session, 1)],
        effects: Vec::new(),
        items: Vec::new(),
        entries: Vec::new(),
        finish: Some(RunFinish {
            run_id: run.clone(),
            session_id: session.clone(),
            scope: scope.clone(),
            status: RunStatus::Completed,
            finish_reason: "done".to_string(),
            error_code: None,
            entries: Vec::new(),
            data: (),
        }),
    }
}

async fn terminal_steps(store: &SqliteStore<TestEvent>, run: &str, key: &str) -> i32 {
    let run = run.to_string();
    let key = key.to_string();
    store
        .call(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT event.event
                       FROM typemach_run_events event
                       JOIN typemach_runs run ON run.run_id = event.run_id
                      WHERE event.run_id = ?1 AND run.scope_key = ?2 AND event.terminal = 1
                      ORDER BY event.seq DESC
                      LIMIT 1",
                )
                .map_err(store_db)?;
            let _: String = stmt
                .query_row((run, key), |row| row.get(0))
                .map_err(store_db)?;
            Ok(stmt.get_status(StatementStatus::VmStep))
        })
        .await
        .expect("terminal steps")
}

async fn atomic_counts(store: &SqliteStore<TestEvent>) -> (i64, i64, i64, String) {
    store
        .call(|conn| {
            conn.query_row(
                "SELECT
                    (SELECT count(*) FROM typemach_checkpoints WHERE thread_id = 'atomic-thread'),
                    (SELECT count(*) FROM typemach_run_events WHERE run_id = 'atomic-run'),
                    (SELECT count(*) FROM typemach_thread_leases WHERE run_id = 'atomic-run'),
                    (SELECT status FROM typemach_runs WHERE run_id = 'atomic-run')",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .map_err(store_db)
        })
        .await
        .expect("atomic counts")
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
