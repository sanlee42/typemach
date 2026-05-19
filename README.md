# typemach

Typed state machine runtime for LLM agents. Checkpoint, interrupt, resume, streaming, cancellation.

LLM agents that call tools, wait for humans, stream output, or survive cancellation are state machines. typemach provides the runtime mechanics. It does not call LLMs — you bring the model, the prompts, and the tools. It gives you typed steps, typed interrupts, live streaming, bounded backpressure, and transparent checkpoint persistence.

LangChain and LangGraph run in Python. Steps are strings in `add_node("answer")`. State is an untyped dict. Interrupt payloads are JSON-shaped. Streaming often means replay after completion. That works until a mistyped step name ships, or a malformed interrupt hits production, or you need to cancel an in-flight model call and the framework does not let you.

typemach is the same idea — a checkpointed state machine for multi-turn agent execution — implemented for a compiler rather than an interpreter.

```rust
#[async_trait]
impl Machine for MyAgent {
    type Step      = Step;
    type State     = State;
    type Input     = Input;
    type Signal    = Signal;
    type Output    = Output;
    type Interrupt = Interrupt;

    async fn transition(&self, step: Self::Step, state: &mut Self::State, ctx: &RunContext<...>)
        -> Result<Transition<Self::Step, Self::Interrupt, Self::Output>, MachineError>
    {
        match step {
            Step::Prepare => {
                ctx.emit(Signal::ToolStarted).await?;
                Ok(Transition::Next(Step::Execute))
            }
            Step::Execute => Ok(Transition::Interrupt(AskUser { question: "..." })),
            Step::Resume  => Ok(Transition::Complete(Output { answer: "..." })),
        }
    }
}
```

## Features

- Steps are enum variants. Missing a branch → compiler error.
- Interrupt and resume are typed. No `Command(resume=...)` with an opaque payload.
- `Runner::stream` emits events while the transition future runs. Not a replay.
- Cooperative cancellation drops in-flight futures.
- `CheckpointSaver` trait. In-memory, SQLite, and Postgres backends. Records are transparent JSON.
- Bounded backpressure on stream channels.
- Heartbeat during long steps and while waiting for the session lock.
- `Runtime` wraps `Runner` + run lifecycle storage for persisted run events.
- `TxRuntime` + `PgStore`/`SqliteStore` commit checkpoint, stream events, and terminal run state together.
- `TxRuntime` leases runs, renews active ownership, fences checkpoints by thread, and reaps stale `running` runs as errors.
- `MachineState` has a blanket impl for any `Serialize + DeserializeOwned` type.

## Install

```toml
[dependencies]
typemach = { git = "https://github.com/sanlee42/typemach.git", features = ["sqlite"] }
```

## Example

```rust
use std::sync::Arc;
use typemach::{Machine, MemorySaver, Runner, RunRequest, Transition};

// Define your step, state, input, signal, output, interrupt types.
// impl Machine for your agent struct.
// Pass it to Runner.

let runner = Runner::new(agent, Arc::new(MemorySaver::new()));

// Blocking turn
let output = runner.invoke(request).await?;

// Streaming turn
let mut stream = runner.stream(request, StreamConfig {
    heartbeat_interval: Duration::from_secs(2),
    channel_capacity: 32,
});
while let Some(event) = stream.next_event().await {
    match event {
        RunStreamEvent::Signal { signal } => handle(signal),
        RunStreamEvent::Completed { output, .. } => break,
        RunStreamEvent::Failed { error } => return Err(error),
        _ => {}
    }
}

// Cancel
runner.cancel_run(&run_id).await;
```

Checkpoints are written after every step transition. Interrupts persist state + step + typed payload. Resume loads the record and calls `apply_resume_input`.

For persisted runs, use `TxRuntime` with `PgStore` or `SqliteStore`. Both stores own generic `typemach_*` tables for sessions, runs, run events, thread leases, and checkpoints. They store typed event envelopes as JSON and keep `scope`, metadata, idempotency keys, cancellation, terminal status, and checkpoint writes in one transactional path.

`TxRuntime` also owns run and thread leases. Each active run carries a `LeaseId`; every commit checks that token before writing checkpoint/events/terminal state. Stores enforce at most one `running` run per `thread_id`, and leased runs also claim the target thread lease, so two workers cannot advance different runs against the same checkpoint thread. If a process dies, another instance can call `TxRuntime::reap(limit)` to finalize expired `running` runs as `error` with `lease_expired`, releasing the thread for a later run.

SQLite is available behind the `sqlite` feature and is the default local durable backend. Use `sqlite-bundled` if the target environment should build with bundled SQLite instead of a system library. Postgres remains the production network store behind the `postgres` feature.

The optional `testkit` feature exposes reusable store conformance tests for backend authors. It covers idempotent start, scope isolation, event sequence checks, terminal-once behavior, running-thread exclusivity, rejected-commit rollback, lease fencing, checkpoint commits, and stale-run reaping.

## License

MIT OR Apache-2.0
