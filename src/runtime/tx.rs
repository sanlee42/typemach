use std::collections::{HashMap, VecDeque};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;

use super::{
    Event, EventRecord, LeaseCfg, MachineOutput, MachineRx, Payload, Start, StartResult,
    StreamEvent, error_code, event_record, new_lease_id,
};
use crate::checkpoint::{CheckpointRecord, CheckpointSaver, CheckpointStore};
use crate::error::MachineError;
use crate::lifecycle::{RunCursor, RunLifecycle, RunSubscription, StartRunResult};
use crate::machine::Machine;
use crate::op::RunOps;
use crate::registry::{RunHandle, RunRegistry};
use crate::run::{
    LeaseId, RunEventReceiver, RunId, RunOutput, RunRequest, RunStreamEvent, SessionId, StepResult,
    StreamConfig, ThreadId,
};
use crate::runner::Runner;
use crate::store::{
    CheckpointWrite, CommitPlan, Lease, LeaseClaim, RunCommitResult, RunEventEnvelope, RunFinish,
    RunLease, RunLookup, RunStart, RunStatus, RunTx,
};

mod op;
use op::*;

struct TxSaver<S> {
    inner: Arc<S>,
    pending: async_rt::sync::Mutex<HashMap<ThreadId, VecDeque<CheckpointRecord>>>,
}

impl<S> TxSaver<S> {
    fn new(inner: Arc<S>) -> Self {
        Self {
            inner,
            pending: async_rt::sync::Mutex::new(HashMap::new()),
        }
    }

    async fn take(&self, thread_id: &ThreadId) -> Option<CheckpointWrite> {
        let mut pending = self.pending.lock().await;
        let queue = pending.get_mut(thread_id)?;
        let record = queue.pop_front()?;
        if queue.is_empty() {
            pending.remove(thread_id);
        }
        Some(CheckpointWrite::new(thread_id.clone(), record))
    }

    async fn clear(&self, thread_id: &ThreadId) {
        self.pending.lock().await.remove(thread_id);
    }
}

#[async_trait]
impl<S> CheckpointSaver for TxSaver<S>
where
    S: CheckpointStore,
{
    async fn save(
        &self,
        thread_id: &str,
        checkpoint: &CheckpointRecord,
    ) -> Result<(), MachineError> {
        let mut pending = self.pending.lock().await;
        pending
            .entry(ThreadId::from(thread_id))
            .or_insert_with(VecDeque::new)
            .push_back(checkpoint.clone());
        Ok(())
    }

    async fn load(&self, thread_id: &str) -> Result<Option<CheckpointRecord>, MachineError> {
        self.inner.load_checkpoint(thread_id).await
    }
}

pub struct TxRuntime<M, S>
where
    M: Machine,
    S: CheckpointStore + RunLease<Event> + RunTx<Event> + 'static,
{
    runner: Runner<M, TxSaver<S>>,
    life: RunLifecycle<Event, S>,
    saver: Arc<TxSaver<S>>,
    store: Arc<S>,
    lease: LeaseCfg,
}

impl<M, S> Clone for TxRuntime<M, S>
where
    M: Machine,
    S: CheckpointStore + RunLease<Event> + RunTx<Event> + 'static,
{
    fn clone(&self) -> Self {
        Self {
            runner: self.runner.clone(),
            life: self.life.clone(),
            saver: Arc::clone(&self.saver),
            store: Arc::clone(&self.store),
            lease: self.lease.clone(),
        }
    }
}

impl<M, S> TxRuntime<M, S>
where
    M: Machine,
    S: CheckpointStore + RunLease<Event> + RunTx<Event> + 'static,
    S::FinishData: Default,
{
    pub fn new(machine: M, store: Arc<S>, max: usize) -> Self {
        Self::with_lease(machine, store, max, LeaseCfg::default())
    }

    pub fn with_lease(machine: M, store: Arc<S>, max: usize, lease: LeaseCfg) -> Self {
        let saver = Arc::new(TxSaver::new(Arc::clone(&store)));
        Self {
            runner: Runner::new(machine, Arc::clone(&saver)),
            life: RunLifecycle::new(RunRegistry::new(), Arc::clone(&store), max),
            saver,
            store,
            lease,
        }
    }

    pub fn life(&self) -> &RunLifecycle<Event, S> {
        &self.life
    }

    pub fn store(&self) -> &S {
        self.life.store()
    }

    pub fn lease(&self) -> &LeaseCfg {
        &self.lease
    }

    pub async fn reap(&self, limit: usize) -> Result<Vec<RunLookup>, MachineError> {
        self.store
            .reap_stale(&self.lease.owner, limit, |run, seq| {
                RunEventEnvelope::new(
                    run.run_id.clone(),
                    run.session_id.clone(),
                    seq,
                    Payload::Fail {
                        error: "lease expired".to_string(),
                    },
                )
            })
            .await
    }

    pub async fn stream(
        &self,
        req: RunRequest<M::Input>,
        start: Start<S::Scope>,
        cfg: StreamConfig,
    ) -> Result<StartResult<MachineRx<M>>, MachineError>
    where
        M::Input: Serialize,
    {
        self.life
            .ensure_session(Some(req.session_id.clone()), &start.scope)
            .await?;

        let scope = start.scope.clone();
        let input = Some(serde_json::to_value(&req.input).map_err(MachineError::Serialization)?);
        let lease_id = new_lease_id();
        let claim = LeaseClaim::new(self.lease.owner.clone(), lease_id.clone(), self.lease.ttl);
        let lease = Lease::new(req.run_id.clone(), self.lease.owner.clone(), lease_id);
        let run = RunStart {
            run_id: req.run_id.clone(),
            session_id: req.session_id.clone(),
            thread_id: req.thread_id.clone(),
            agent_kind: start.kind,
            model: start.model,
            client_run_key: start.key,
            parent_run_id: start.parent,
            retry_of_run_id: start.retry_of,
            scope: start.scope,
            metadata: start.meta,
            input,
            entries: start.entries,
            lease: Some(claim),
        };
        let token = start.token.unwrap_or_else(|| req.run_id.to_string());
        match self
            .life
            .start_run(run, RunHandle::new(token), None)
            .await?
        {
            StartRunResult::Started => {}
            StartRunResult::Existing(existing) => return Ok(StartResult::Existing(existing)),
            StartRunResult::NotRegistered(rejection) => {
                return Ok(StartResult::Rejected(rejection));
            }
        }

        let run_id = req.run_id.clone();
        let session_id = req.session_id.clone();
        let thread_id = req.thread_id.clone();
        let ops = Arc::new(TxRunOps::new(
            Arc::clone(&self.store),
            req.run_id.clone(),
            scope.clone(),
            lease.id.clone(),
        ));
        let run_ops: Arc<dyn RunOps> = ops.clone();
        let raw = self.runner.stream_with_ops(req, cfg, run_ops);
        let (tx, receiver) = async_rt::sync::mpsc::channel(cfg.channel_capacity());
        let runner = self.runner.clone();
        let life = self.life.clone();
        let saver = Arc::clone(&self.saver);
        let lease_stop = Arc::new(AtomicBool::new(false));
        spawn_renew(
            Arc::clone(&self.store),
            self.runner.clone(),
            lease.clone(),
            self.lease.clone(),
            Arc::clone(&lease_stop),
        );
        async_rt::spawn(async move {
            let ctx = TxCtx {
                runner,
                life,
                saver,
                tx,
                run_id,
                session_id,
                thread_id,
                scope,
                lease,
                ops,
                lease_stop,
            };
            tx_bridge::<M, S>(ctx, raw).await;
        });

        Ok(StartResult::Started(RunEventReceiver { receiver }))
    }

    pub async fn invoke(
        &self,
        req: RunRequest<M::Input>,
        start: Start<S::Scope>,
    ) -> Result<StartResult<MachineOutput<M>>, MachineError>
    where
        M::Input: Serialize,
    {
        let stream = self.stream(req, start, StreamConfig::default()).await?;
        let StartResult::Started(mut rx) = stream else {
            return Ok(stream.map_started(|_| unreachable!()));
        };

        loop {
            match rx.next_event().await {
                Some(RunStreamEvent::Completed {
                    trace,
                    output,
                    snapshot,
                }) => {
                    return Ok(StartResult::Started(RunOutput::Completed {
                        trace,
                        output,
                        snapshot,
                    }));
                }
                Some(RunStreamEvent::Interrupted {
                    interrupt,
                    snapshot,
                }) => {
                    return Ok(StartResult::Started(RunOutput::Interrupted {
                        interrupt,
                        snapshot,
                    }));
                }
                Some(RunStreamEvent::Failed { error }) => return Err(error),
                Some(RunStreamEvent::Cancelled) => return Err(MachineError::Cancelled),
                Some(_) => {}
                None => return Err(MachineError::StreamClosed),
            }
        }
    }

    pub async fn cancel(&self, run_id: &RunId, scope: &S::Scope) -> Result<bool, MachineError> {
        let active = self.life.request_cancel(run_id, scope).await?.is_some();
        self.runner.cancel_run(run_id).await;
        Ok(active)
    }

    pub async fn subscribe(
        &self,
        run_id: &RunId,
        scope: &S::Scope,
        cursor: impl Into<RunCursor>,
    ) -> Result<RunSubscription<Event>, MachineError> {
        self.life.subscribe(run_id, scope, cursor).await
    }
}

fn spawn_renew<M, S>(
    store: Arc<S>,
    runner: Runner<M, TxSaver<S>>,
    lease: Lease,
    cfg: LeaseCfg,
    stop: Arc<AtomicBool>,
) where
    M: Machine,
    S: CheckpointStore + RunLease<Event> + RunTx<Event> + 'static,
{
    async_rt::spawn(async move {
        let every = cfg.renew.max(Duration::from_millis(1));
        loop {
            async_rt::time::sleep(every).await;
            if stop.load(Ordering::SeqCst) {
                return;
            }
            match store.renew(&lease, cfg.ttl).await {
                Ok(true) => {}
                Ok(false) | Err(_) => {
                    async_rt::task::yield_now().await;
                    if !stop.load(Ordering::SeqCst) {
                        runner.cancel_run(&lease.run).await;
                    }
                    return;
                }
            }
        }
    });
}

struct PendingStep<M>
where
    M: Machine,
{
    event: StreamEvent<M>,
    checkpoint: CheckpointWrite,
    payload: Payload,
}

struct TxCtx<M, S>
where
    M: Machine,
    S: CheckpointStore + RunLease<Event> + RunTx<Event>,
{
    runner: Runner<M, TxSaver<S>>,
    life: RunLifecycle<Event, S>,
    saver: Arc<TxSaver<S>>,
    tx: async_rt::sync::mpsc::Sender<StreamEvent<M>>,
    run_id: RunId,
    session_id: SessionId,
    thread_id: ThreadId,
    scope: S::Scope,
    lease: Lease,
    ops: Arc<TxRunOps<S>>,
    lease_stop: Arc<AtomicBool>,
}

async fn tx_bridge<M, S>(ctx: TxCtx<M, S>, raw: MachineRx<M>)
where
    M: Machine,
    S: CheckpointStore + RunLease<Event> + RunTx<Event> + 'static,
    S::FinishData: Default,
{
    tx_bridge_inner::<M, S>(&ctx, raw).await;
    ctx.lease_stop.store(true, Ordering::SeqCst);
}

async fn tx_bridge_inner<M, S>(ctx: &TxCtx<M, S>, mut raw: MachineRx<M>)
where
    M: Machine,
    S: CheckpointStore + RunLease<Event> + RunTx<Event> + 'static,
    S::FinishData: Default,
{
    let mut pending_step = None;
    loop {
        match raw.receiver.try_recv() {
            Ok(event) => {
                if tx_forward_event::<M, S>(ctx, event, &mut pending_step).await {
                    return;
                }
                continue;
            }
            Err(async_rt::sync::mpsc::error::TryRecvError::Empty) => {}
            Err(async_rt::sync::mpsc::error::TryRecvError::Disconnected) => {
                tx_cancel_and_finish::<M, S>(ctx, "stream_closed", pending_step).await;
                return;
            }
        }

        async_rt::select! {
            event = raw.next_event() => {
                let Some(event) = event else {
                    tx_cancel_and_finish::<M, S>(ctx, "stream_closed", pending_step).await;
                    return;
                };
                if tx_forward_event::<M, S>(ctx, event, &mut pending_step).await
                {
                    return;
                }
            }
            _ = ctx.tx.closed() => {
                tx_cancel_and_finish::<M, S>(ctx, "receiver_closed", pending_step).await;
                return;
            }
        }
    }
}

async fn tx_forward_event<M, S>(
    ctx: &TxCtx<M, S>,
    event: StreamEvent<M>,
    pending_step: &mut Option<PendingStep<M>>,
) -> bool
where
    M: Machine,
    S: CheckpointStore + RunLease<Event> + RunTx<Event> + 'static,
    S::FinishData: Default,
{
    let record = event_record::<M>(&event);
    let record = match record {
        Ok(record) => record,
        Err(error) => {
            ctx.runner.cancel_run(&ctx.run_id).await;
            tx_finish_error::<M, S>(ctx, &error, pending_step.take()).await;
            let _ = ctx.tx.send(RunStreamEvent::Failed { error }).await;
            return true;
        }
    };

    match record {
        EventRecord::Append(payload) => {
            if let Payload::StepDone { result, .. } = &payload {
                let checkpoint = match ctx.saver.take(&ctx.thread_id).await {
                    Some(checkpoint) => checkpoint,
                    None => {
                        let error = MachineError::InvalidRunEvent {
                            reason: "missing checkpoint for step event".to_string(),
                        };
                        ctx.runner.cancel_run(&ctx.run_id).await;
                        tx_finish_error::<M, S>(ctx, &error, pending_step.take()).await;
                        let _ = ctx.tx.send(RunStreamEvent::Failed { error }).await;
                        return true;
                    }
                };
                if matches!(result, StepResult::Interrupt | StepResult::Complete) {
                    *pending_step = Some(PendingStep {
                        event,
                        checkpoint,
                        payload,
                    });
                    return false;
                }

                let ops = ctx.ops.take().await;
                let record = ctx
                    .life
                    .commit_events(
                        &ctx.run_id,
                        &ctx.session_id,
                        &ctx.scope,
                        CommitPlan {
                            lease: Some(ctx.lease.id.clone()),
                            checkpoint: Some(checkpoint),
                            event_count: 0,
                            effects: ops.effects,
                            items: ops.items,
                            entries: ops.entries,
                            finish: None,
                        },
                        vec![payload],
                    )
                    .await;
                if let Err(error) =
                    tx_handle_record_error::<M, S>(ctx, record, pending_step.take()).await
                {
                    let _ = ctx.tx.send(RunStreamEvent::Failed { error }).await;
                    return true;
                }
            } else {
                if pending_step.is_some() {
                    let error = MachineError::InvalidRunEvent {
                        reason: "terminal step must be followed by terminal event".to_string(),
                    };
                    ctx.runner.cancel_run(&ctx.run_id).await;
                    tx_finish_error::<M, S>(ctx, &error, pending_step.take()).await;
                    let _ = ctx.tx.send(RunStreamEvent::Failed { error }).await;
                    return true;
                }
                let record = ctx
                    .life
                    .commit_events(
                        &ctx.run_id,
                        &ctx.session_id,
                        &ctx.scope,
                        CommitPlan {
                            lease: Some(ctx.lease.id.clone()),
                            checkpoint: None,
                            event_count: 0,
                            effects: Vec::new(),
                            items: Vec::new(),
                            entries: Vec::new(),
                            finish: None,
                        },
                        vec![payload],
                    )
                    .await;
                if let Err(error) = tx_handle_record_error::<M, S>(ctx, record, None).await {
                    let _ = ctx.tx.send(RunStreamEvent::Failed { error }).await;
                    return true;
                }
            }

            if ctx.tx.send(event).await.is_err() {
                tx_cancel_and_finish::<M, S>(ctx, "receiver_closed", pending_step.take()).await;
                return true;
            }
            false
        }
        EventRecord::Finish {
            payload,
            status,
            reason,
            code,
        } => {
            let pending = pending_step.take();
            let mut payloads = Vec::new();
            let checkpoint = pending.as_ref().map(|pending| pending.checkpoint.clone());
            if let Some(pending) = &pending {
                payloads.push(pending.payload.clone());
            }
            payloads.push(payload);
            let finish = RunFinish {
                run_id: ctx.run_id.clone(),
                session_id: ctx.session_id.clone(),
                scope: ctx.scope.clone(),
                status,
                finish_reason: reason.to_string(),
                error_code: code,
                entries: Vec::new(),
                data: S::FinishData::default(),
            };
            let ops = ctx.ops.take().await;
            let record = ctx
                .life
                .commit_events(
                    &ctx.run_id,
                    &ctx.session_id,
                    &ctx.scope,
                    CommitPlan {
                        lease: Some(ctx.lease.id.clone()),
                        checkpoint,
                        event_count: 0,
                        effects: ops.effects,
                        items: ops.items,
                        entries: ops.entries,
                        finish: Some(finish),
                    },
                    payloads,
                )
                .await;
            match record {
                Ok(result) if !result.is_skipped() => {
                    ctx.lease_stop.store(true, Ordering::SeqCst);
                }
                Ok(_) => {
                    let error = MachineError::RunNotFound;
                    ctx.runner.cancel_run(&ctx.run_id).await;
                    tx_finish_error::<M, S>(ctx, &error, pending).await;
                    let _ = ctx.tx.send(RunStreamEvent::Failed { error }).await;
                    return true;
                }
                Err(error) => {
                    ctx.runner.cancel_run(&ctx.run_id).await;
                    tx_finish_error::<M, S>(ctx, &error, pending).await;
                    let _ = ctx.tx.send(RunStreamEvent::Failed { error }).await;
                    return true;
                }
            }
            if let Some(pending) = pending
                && ctx.tx.send(pending.event).await.is_err()
            {
                return true;
            }
            let _ = ctx.tx.send(event).await;
            true
        }
    }
}

async fn tx_handle_record_error<M, S>(
    ctx: &TxCtx<M, S>,
    record: Result<RunCommitResult<Event>, MachineError>,
    pending: Option<PendingStep<M>>,
) -> Result<(), MachineError>
where
    M: Machine,
    S: CheckpointStore + RunLease<Event> + RunTx<Event> + 'static,
    S::FinishData: Default,
{
    match record {
        Ok(result) if !result.is_skipped() => Ok(()),
        Ok(_) => {
            let error = MachineError::RunNotFound;
            ctx.runner.cancel_run(&ctx.run_id).await;
            tx_finish_error::<M, S>(ctx, &error, pending).await;
            Err(error)
        }
        Err(error) => {
            ctx.runner.cancel_run(&ctx.run_id).await;
            tx_finish_error::<M, S>(ctx, &error, pending).await;
            Err(error)
        }
    }
}

async fn tx_cancel_and_finish<M, S>(
    ctx: &TxCtx<M, S>,
    reason: &str,
    pending: Option<PendingStep<M>>,
) where
    M: Machine,
    S: CheckpointStore + RunLease<Event> + RunTx<Event> + 'static,
    S::FinishData: Default,
{
    let _ = ctx.life.request_cancel(&ctx.run_id, &ctx.scope).await;
    ctx.runner.cancel_run(&ctx.run_id).await;
    let checkpoint = pending.as_ref().map(|pending| pending.checkpoint.clone());
    let mut payloads = Vec::new();
    if let Some(pending) = pending {
        payloads.push(pending.payload);
    }
    payloads.push(Payload::Cancel);
    let finish = RunFinish {
        run_id: ctx.run_id.clone(),
        session_id: ctx.session_id.clone(),
        scope: ctx.scope.clone(),
        status: RunStatus::Cancelled,
        finish_reason: reason.to_string(),
        error_code: None,
        entries: Vec::new(),
        data: S::FinishData::default(),
    };
    let ops = ctx.ops.take().await;
    let _ = ctx
        .life
        .commit_events(
            &ctx.run_id,
            &ctx.session_id,
            &ctx.scope,
            CommitPlan {
                lease: Some(ctx.lease.id.clone()),
                checkpoint,
                event_count: 0,
                effects: ops.effects,
                items: ops.items,
                entries: ops.entries,
                finish: Some(finish),
            },
            payloads,
        )
        .await;
    ctx.lease_stop.store(true, Ordering::SeqCst);
    ctx.saver.clear(&ctx.thread_id).await;
}

async fn tx_finish_error<M, S>(
    ctx: &TxCtx<M, S>,
    error: &MachineError,
    pending: Option<PendingStep<M>>,
) where
    M: Machine,
    S: CheckpointStore + RunLease<Event> + RunTx<Event> + 'static,
    S::FinishData: Default,
{
    let checkpoint = pending.as_ref().map(|pending| pending.checkpoint.clone());
    let mut payloads = Vec::new();
    if let Some(pending) = pending {
        payloads.push(pending.payload);
    }
    payloads.push(Payload::Fail {
        error: error.to_string(),
    });
    let finish = RunFinish {
        run_id: ctx.run_id.clone(),
        session_id: ctx.session_id.clone(),
        scope: ctx.scope.clone(),
        status: RunStatus::Error,
        finish_reason: "failed".to_string(),
        error_code: Some(error_code(error).to_string()),
        entries: Vec::new(),
        data: S::FinishData::default(),
    };
    let ops = ctx.ops.take().await;
    let _ = ctx
        .life
        .commit_events(
            &ctx.run_id,
            &ctx.session_id,
            &ctx.scope,
            CommitPlan {
                lease: Some(ctx.lease.id.clone()),
                checkpoint,
                event_count: 0,
                effects: ops.effects,
                items: ops.items,
                entries: ops.entries,
                finish: Some(finish),
            },
            payloads,
        )
        .await;
    ctx.lease_stop.store(true, Ordering::SeqCst);
}
