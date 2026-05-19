use super::*;
use crate::run::ThreadId;

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
