# typemach

> Typed machines for agents that have to survive real production runtime.

LLM agents are not just prompt calls. The moment an agent can call tools, wait for a human, resume later, stream partial output, or be cancelled mid-flight, it is a state machine.

Most agent frameworks let that machine live in runtime strings, dictionaries, callbacks, and replayed event logs. That is fast to prototype and painful to operate. A misspelled step name becomes a late crash. A malformed interrupt becomes a production incident. A "stream" often means "run everything, then replay what happened from memory."

typemach exists for the opposite model:

- Steps are Rust enum variants, not graph node strings.
- State is a serializable Rust type, not an unowned dictionary.
- Interrupts and resume input are typed application contracts.
- Streaming is live runtime output, not result replay.
- Cancellation can drop an in-flight transition future.
- Checkpoints are explicit records owned by a small storage trait.

typemach is deliberately not an LLM framework. It does not know about prompts, tools, vector databases, models, agents, or business protocols. It gives you the runtime boundary for a typed, checkpointed, cancellable state machine. Your code owns the intelligence.

## What makes it different

| Runtime concern | String/dict agent runtimes | typemach |
|---|---|---|
| Step routing | `add_node("answer", ...)` | `Transition::Next(Step::Answer)` |
| Exhaustiveness | Runtime branch coverage | Rust `match` over your step enum |
| Interrupts | JSON-shaped payloads | `Transition::Interrupt(MyInterrupt)` |
| Resume | Opaque command payload | `apply_resume_input` plus `ResumeAction` |
| Streaming | Often replay after completion | `Runner::stream` while the transition is running |
| Agent events | Framework-owned event schema | Machine-owned typed `Signal` |
| Backpressure | Usually hidden | Bounded channel capacity |
| Cancellation | Best-effort flags | Cooperative cancel that can drop the transition future |
| Persistence | Framework black box | `CheckpointSaver` trait |

The special part is not that typemach can run a graph. The special part is that the runtime behavior you need in production is part of one typed contract: state, step, input, signal, output, interrupt.

## Install

typemach is currently consumed from Git:

```toml
[dependencies]
typemach = { git = "https://github.com/sanlee42/typemach.git" }
```

For Postgres checkpoints:

```toml
[dependencies]
typemach = { git = "https://github.com/sanlee42/typemach.git", features = ["checkpoint-postgres"] }
```

For applications, pin a commit or tag instead of floating on the default branch.

## The machine contract

```rust
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use typemach::{Machine, MachineError, RunContext, Transition};

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Step {
    Prepare,
    Answer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct State {
    answer: Option<String>,
}

#[derive(Debug, Clone)]
struct Input {
    question: String,
}

#[derive(Debug)]
enum Signal {
    ToolStarted,
    AnswerDelta(String),
}

#[derive(Debug)]
struct Output {
    answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Interrupt {
    question: String,
}

struct Agent;

#[async_trait]
impl Machine for Agent {
    type Step = Step;
    type State = State;
    type Input = Input;
    type Signal = Signal;
    type Output = Output;
    type Interrupt = Interrupt;

    fn start_step(&self) -> Self::Step {
        Step::Prepare
    }

    fn new_state(
        &self,
        _input: &Self::Input,
        _previous: Option<&Self::State>,
        _snapshot: Option<&serde_json::Value>,
    ) -> Result<Self::State, MachineError> {
        Ok(State { answer: None })
    }

    fn apply_resume_input(
        &self,
        _state: &mut Self::State,
        _input: &Self::Input,
    ) -> Result<(), MachineError> {
        Ok(())
    }

    async fn transition(
        &self,
        step: Self::Step,
        state: &mut Self::State,
        ctx: &RunContext<Self::Input, Self::Step, Self::Signal, Self::Output, Self::Interrupt>,
    ) -> Result<Transition<Self::Step, Self::Interrupt, Self::Output>, MachineError> {
        match step {
            Step::Prepare => {
                ctx.emit(Signal::ToolStarted).await?;
                Ok(Transition::Next(Step::Answer))
            }
            Step::Answer => {
                let answer = format!("answering: {}", ctx.input.question);
                ctx.emit(Signal::AnswerDelta(answer.clone())).await?;
                state.answer = Some(answer.clone());
                Ok(Transition::Complete(Output { answer }))
            }
        }
    }
}
```

The trait has six associated types:

- `Step`: every runnable state in your machine.
- `State`: durable machine state, serialized into checkpoints.
- `Input`: one turn of caller input.
- `Signal`: typed live events emitted by your transition code.
- `Output`: successful terminal output.
- `Interrupt`: typed pending work needed before resume.

There is no graph builder, no magic command object, and no framework event schema to conform to.

## Running a machine

Use `invoke` when the caller wants a single terminal result:

```rust
let output = runner.invoke(request).await?;
```

Use `stream` when the caller needs live runtime events:

```rust
let mut events = runner.stream(
    request,
    StreamConfig {
        heartbeat_interval: std::time::Duration::from_secs(2),
        channel_capacity: 32,
    },
);

while let Some(event) = events.next_event().await {
    match event {
        RunStreamEvent::Signal { signal } => {
            // Your typed machine signal.
        }
        RunStreamEvent::Completed { output, .. } => {
            break;
        }
        RunStreamEvent::Failed { error } => {
            return Err(error);
        }
        _ => {}
    }
}
```

`RunStreamEvent` is the runtime event surface:

- `Started`
- `Heartbeat`
- `StepStarted`
- `StepFinished`
- `Signal`
- `Interrupted`
- `Completed`
- `Failed`
- `Cancelled`

Signals emitted inside a transition are ordered before the terminal event for that transition. Heartbeats are emitted while a transition is still running and while a run is waiting for the thread lock. If the receiver is dropped, the stream driver stops and the active run is cleaned up.

## Checkpoint, interrupt, resume

Every completed step stores a checkpoint through `CheckpointSaver`. On interrupt, typemach stores the durable state, interrupted step, typed interrupt payload, and run id. On resume, it loads the checkpoint, calls `apply_resume_input`, and resumes using `resume_action`.

```rust
use std::sync::Arc;
use typemach::{MemorySaver, Runner};

let runner = Runner::new(agent, Arc::new(MemorySaver::new()));
```

The optional Postgres backend stores the same checkpoint record in `typemach_checkpoints`:

```rust
#[cfg(feature = "checkpoint-postgres")]
{
    use typemach::checkpoint_pg::PostgresSaver;

    let saver = PostgresSaver::new(pool);
    saver.ensure_schema().await?;
    let runner = Runner::new(agent, Arc::new(saver));
}
```

Checkpoint records are versioned. The state payload is transparent JSON, so operators can inspect what was persisted instead of reverse-engineering a framework database.

## Cancellation and backpressure

`Runner::cancel_run(&run_id)` marks a run as cancelled and notifies the active cancel token. The runner selects transition execution against cancellation, so a slow model call or tool future can be dropped when the caller gives up.

Streaming uses a bounded channel. That is intentional. If a downstream adapter stops reading, typemach does not keep buffering an unbounded event log in memory.

## LLM integration

LLM calls belong in your machine transitions. typemach does not wrap providers, invent tool schemas, or parse model-specific protocols for you.

See [`examples/minimal_llm.rs`](examples/minimal_llm.rs) for a minimal OpenAI-compatible streaming example:

```bash
OPENAI_API_KEY=... \
OPENAI_BASE_URL=https://api.openai.com/v1 \
OPENAI_MODEL=... \
cargo run --example minimal_llm -- "Explain typemach in one sentence"
```

The example:

- Calls `OPENAI_BASE_URL/chat/completions` with `stream: true`.
- Parses `data:` server-sent events.
- Emits typed `AssistantDelta` signals from `RunContext::emit`.
- Prints typemach runtime events as they arrive.
- Fails explicitly on missing config, HTTP errors, malformed chunks, missing `[DONE]`, or an empty final answer.

No fake model, no generated fallback answer, no hidden provider default.

## Design boundary

typemach is small because it only owns runtime mechanics:

- machine execution
- checkpoint persistence boundary
- interrupt and resume lifecycle
- live stream event ordering
- heartbeat
- cooperative cancellation
- bounded backpressure
- max step enforcement

Everything else stays outside the crate. That boundary is the point. LLM agents need a runtime that is strict enough to operate and small enough to trust.

## License

MIT OR Apache-2.0
