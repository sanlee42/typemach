use super::*;

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
