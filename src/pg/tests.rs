use std::collections::HashMap;

use deadpool_postgres::tokio_postgres::NoTls;
use deadpool_postgres::{Pool, Runtime};

use super::*;
use crate::run::{RunId, SessionId, ThreadId};
use crate::store::{RunStart, RunStore};

#[test]
fn pg_store_matches_contract() {
    let Some(url) = std::env::var("TEST_DATABASE_URL").ok() else {
        eprintln!("skip pg_store_matches_contract: set TEST_DATABASE_URL to run postgres tests");
        return;
    };
    check_test_url(&url);

    block_on(async {
        let store = PgStore::<crate::testkit::TestEvent>::new(pool(url));
        reset_schema(&store).await.expect("reset schema");
        store.ensure_schema().await.expect("schema");
        crate::testkit::run_store_contract(&store)
            .await
            .expect("contract");
    });
}

#[test]
fn pg_ensure_schema_adds_run_start_columns() {
    let Some(url) = std::env::var("TEST_DATABASE_URL").ok() else {
        eprintln!(
            "skip pg_ensure_schema_adds_run_start_columns: set TEST_DATABASE_URL to run postgres tests"
        );
        return;
    };
    check_test_url(&url);

    block_on(async {
        let store = PgStore::<crate::testkit::TestEvent>::new(pool(url));
        reset_schema(&store).await.expect("reset schema");
        let client = store.pool().get().await.expect("pool client");
        client
            .batch_execute(
                "CREATE TABLE typemach_runs (
                    run_id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    scope JSONB NOT NULL,
                    agent_kind TEXT NOT NULL,
                    model TEXT NULL,
                    client_run_key TEXT NULL,
                    parent_run_id TEXT NULL,
                    retry_of_run_id TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    status TEXT NOT NULL,
                    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    finished_at TIMESTAMPTZ NULL,
                    finish_reason TEXT NULL,
                    error_code TEXT NULL,
                    finish_data JSONB NULL,
                    owner_id TEXT NULL,
                    lease_id TEXT NULL,
                    lease_expires_at TIMESTAMPTZ NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );",
            )
            .await
            .expect("old schema");
        drop(client);

        store.ensure_schema().await.expect("schema");
        let client = store.pool().get().await.expect("pool client");
        let columns = client
            .query(
                "SELECT column_name
                 FROM information_schema.columns
                 WHERE table_name = 'typemach_runs'",
                &[],
            )
            .await
            .expect("columns")
            .into_iter()
            .map(|row| row.get::<_, String>(0))
            .collect::<Vec<_>>();
        assert!(columns.iter().any(|column| column == "input"));
        assert!(columns.iter().any(|column| column == "start_sig"));
        drop(client);

        let run_id = RunId::from("pg-upgrade-run");
        store
            .start_run(&RunStart {
                run_id: run_id.clone(),
                session_id: SessionId::from("pg-upgrade-session"),
                thread_id: ThreadId::from("pg-upgrade-thread"),
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
fn pg_ensure_schema_rejects_noncompact_terminal_without_mutation() {
    let Some(url) = std::env::var("TEST_DATABASE_URL").ok() else {
        eprintln!(
            "skip pg_ensure_schema_rejects_noncompact_terminal_without_mutation: set TEST_DATABASE_URL to run postgres tests"
        );
        return;
    };
    check_test_url(&url);

    block_on(async {
        let store = PgStore::<crate::runtime::Event>::new(pool(url));
        reset_schema(&store).await.expect("reset schema");
        store.ensure_schema().await.expect("schema");
        let client = store.pool().get().await.expect("pool client");
        client
            .batch_execute(
                "INSERT INTO typemach_sessions (scope_key, session_id, scope)
                 VALUES ('{}', 'cutover-session', '{}');
                 INSERT INTO typemach_runs (
                    run_id, scope_key, session_id, thread_id, scope, agent_kind,
                    metadata, start_sig, status
                 ) VALUES (
                    'cutover-run', '{}', 'cutover-session', 'cutover-thread', '{}',
                    'test', '{}', '', 'completed'
                 );
                 INSERT INTO typemach_run_events (run_id, session_id, seq, terminal, event)
                 VALUES (
                    'cutover-run', 'cutover-session', 1, TRUE,
                    '{
                        \"run_id\": \"cutover-run\",
                        \"session_id\": \"cutover-session\",
                        \"seq\": 1,
                        \"payload\": {
                            \"type\": \"done\",
                            \"output\": {\"ok\": true},
                            \"checkpoint\": \"cutover-run\"
                        }
                    }'::jsonb
                 );",
            )
            .await
            .expect("compact terminal");
        drop(client);
        store.ensure_schema().await.expect("compact contract");

        let client = store.pool().get().await.expect("pool client");
        client
            .execute(
                "UPDATE typemach_run_events
                    SET event = jsonb_set(event, '{payload,snapshot}', '{\"old\": true}'::jsonb)
                  WHERE run_id = 'cutover-run'",
                &[],
            )
            .await
            .expect("noncompact fixture");
        drop(client);

        let error = store
            .ensure_schema()
            .await
            .expect_err("noncompact terminal must fail schema validation");
        assert!(matches!(
            error,
            MachineError::InvalidRunEvent { ref reason }
                if reason == "terminal cutover-run does not match the compact contract"
        ));
        let client = store.pool().get().await.expect("pool client");
        let snapshot = client
            .query_one(
                "SELECT (event #> '{payload,snapshot}')::text
                   FROM typemach_run_events
                  WHERE run_id = 'cutover-run'",
                &[],
            )
            .await
            .expect("stored terminal")
            .get::<_, String>(0);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&snapshot).expect("snapshot JSON"),
            serde_json::json!({"old": true})
        );

        for (name, invalid) in [
            ("scalar", "7"),
            ("array", "[]"),
            ("empty object", "{}"),
            ("missing envelope", r#"{"payload":{"type":"cancel"}}"#),
            (
                "invalid envelope",
                r#"{"run_id":"cutover-run","session_id":"cutover-session","seq":"1","payload":{"type":"cancel"}}"#,
            ),
        ] {
            let client = store.pool().get().await.expect("pool client");
            client
                .execute(
                    "UPDATE typemach_run_events SET event = $1::jsonb WHERE run_id = 'cutover-run'",
                    &[&invalid],
                )
                .await
                .unwrap_or_else(|error| panic!("{name} terminal fixture: {error}"));
            drop(client);
            let error = match store.ensure_schema().await {
                Ok(()) => panic!("{name} terminal must fail schema validation"),
                Err(error) => error,
            };
            assert!(matches!(
                error,
                MachineError::InvalidRunEvent { ref reason }
                    if reason == "terminal cutover-run does not match the compact contract"
            ));
            let client = store.pool().get().await.expect("pool client");
            let raw = client
                .query_one(
                    "SELECT event::text FROM typemach_run_events WHERE run_id = 'cutover-run'",
                    &[],
                )
                .await
                .unwrap_or_else(|error| panic!("{name} terminal after rejection: {error}"))
                .get::<_, String>(0);
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&raw).expect("stored JSON"),
                serde_json::from_str::<serde_json::Value>(invalid).expect("fixture JSON"),
                "{name} terminal mutated"
            );
        }

        for valid in [
            r#"{"run_id":"cutover-run","session_id":"cutover-session","seq":1,"payload":{"type":"fail","error":"failed"}}"#,
            r#"{"run_id":"cutover-run","session_id":"cutover-session","seq":1,"payload":{"type":"cancel"}}"#,
        ] {
            let client = store.pool().get().await.expect("pool client");
            client
                .execute(
                    "UPDATE typemach_run_events SET event = $1::jsonb WHERE run_id = 'cutover-run'",
                    &[&valid],
                )
                .await
                .expect("valid terminal fixture");
            drop(client);
            store
                .ensure_schema()
                .await
                .expect("valid terminal contract");
        }
    });
}

fn check_test_url(url: &str) {
    let db = database_name(url);
    if db != "scratch" && !db.contains("test") {
        panic!("refusing to run typemach pg test against non-test database `{db}`");
    }
}

fn database_name(url: &str) -> &str {
    let without_query = url.split(['?', '#']).next().unwrap_or(url);
    without_query
        .rsplit('/')
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or("")
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

async fn reset_schema<E, Scope, Data>(store: &PgStore<E, Scope, Data>) -> Result<(), String> {
    let client = store.pool().get().await.map_err(|err| err.to_string())?;
    client
        .batch_execute(
            "DROP TABLE IF EXISTS typemach_run_events CASCADE;
                 DROP TABLE IF EXISTS typemach_thread_leases CASCADE;
                 DROP TABLE IF EXISTS typemach_runs CASCADE;
                 DROP TABLE IF EXISTS typemach_sessions CASCADE;
                 DROP TABLE IF EXISTS typemach_checkpoints CASCADE;",
        )
        .await
        .map_err(|err| err.to_string())
}

fn pool(url: String) -> Pool {
    let mut cfg = deadpool_postgres::Config::new();
    cfg.url = Some(url);
    cfg.create_pool(Some(Runtime::Tokio1), NoTls)
        .expect("create pool")
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
