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
- `CheckpointSaver` trait. In-memory and Postgres backends. Records are transparent JSON.
- Bounded backpressure on stream channels.
- Heartbeat during long steps and while waiting for the session lock.
- `MachineState` has a blanket impl for any `Serialize + DeserializeOwned` type.

## Install

```toml
[dependencies]
typemach = { git = "https://github.com/sanlee42/typemach.git", features = ["checkpoint-postgres"] }
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

## License

MIT OR Apache-2.0
