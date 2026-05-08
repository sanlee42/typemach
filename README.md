# typemach

> *If your agent doesn't compile, it can't crash at 3 AM.*

A typed state machine runtime for LLM agents. One trait. Compile-time step validation. Checkpoint, interrupt, resume, cancel — all built in.

```rust
#[async_trait]
impl Machine for MyAgent {
    type Step   = MyStep;     // enum — compiler checks every route
    type State  = MyState;    // Serialize + DeserializeOwned → blanket impl
    type Input  = MyInput;    // whatever your turn carries
    type Signal = MySignal;   // typed streaming signals
    type Output = MyOutput;   // whatever your caller consumes
    type Interrupt = MyInterrupt; // typed, not a dict

    fn start_step(&self) -> Self::Step { MyStep::Prepare }

    async fn transition(
        &self,
        step: Self::Step,
        state: &mut Self::State,
        ctx: &RunContext<Self::Input, Self::Step, Self::Signal, Self::Output, Self::Interrupt>,
    )
        -> Result<Transition<Self::Step, Self::Interrupt, Self::Output>, MachineError>
    {
        match step {
            MyStep::Prepare => {
                ctx.emit(MySignal::ToolStarted).await?;
                state.tools = fetch_tools(ctx).await?;
                Ok(Transition::Next(MyStep::Plan))
            }
            MyStep::Plan => {
                if state.need_clarification {
                    Ok(Transition::Interrupt(MyInterrupt::Ask { question: "which metric?" }))
                } else {
                    Ok(Transition::Complete(MyOutput { answer: "done" }))
                }
            }
        }
    }
}
```

---

## The problem

LangGraph, CrewAI, AutoGen — all Python. All runtime. A mistyped field in your `TypedDict`, a missing `isinstance` check, and your agent fails in production with a `KeyError` nobody saw coming. Python gives you flexibility. It does not give you the answer to "did I handle every state?"

## What typemach does differently

| | Everyone else | typemach |
|---|---|---|
| State machine | `add_node("think", ...)` — strings | `Transition::Next(MyStep::Think)` — enum |
| Interrupt | `interrupt({"field": "date"})` — dict | `Transition::Interrupt(AskDate)` — type |
| Resume | `Command(resume="tomorrow")` — opaque string | `apply_resume_input(state, input)` — typed method |
| Checkpoint | framework black box | `CheckpointSaver` trait, bring your own DB |
| API surface | `StateGraph`, `Command`, `Send`, `Pregel`, `Channel`, ... | `Machine` trait, 5 methods |
| Code overhead | Large runtime package | Small typed runtime |

## The contract

```rust
pub trait Machine {
    type Step:     Clone + Debug + Serialize + DeserializeOwned + Send + Sync;
    type State:    MachineState;
    type Input:    Clone + Send + Sync;
    type Signal:   Send + Sync;
    type Output:   Send + Sync;
    type Interrupt: Clone + Serialize + DeserializeOwned + Send + Sync;

    fn start_step(&self) -> Self::Step;
    fn resume_action(&self, interrupt: &Self::Interrupt) -> ResumeAction<Self::Step>;
    fn new_state(&self, input: &Self::Input, previous: Option<&Self::State>, snapshot: Option<&Value>) -> Result<Self::State, MachineError>;
    fn apply_resume_input(&self, state: &mut Self::State, input: &Self::Input) -> Result<(), MachineError>;

    async fn transition(
        &self,
        step: Self::Step,
        state: &mut Self::State,
        ctx: &RunContext<Self::Input, Self::Step, Self::Signal, Self::Output, Self::Interrupt>,
    ) -> Result<Transition<Self::Step, Self::Interrupt, Self::Output>, MachineError>;
}
```

6 associated types. 5 methods. That's the whole API.

## What you get for free

Once you `impl Machine`, the `Runner<M, CK>` gives you:

- **Checkpoint persistence** — every step transition saves state. On resume, `Runner` loads the last checkpoint and calls `apply_resume_input`.
- **Session isolation** — per-session async lock, no concurrent turn execution on the same session.
- **Step enforcement** — `max_steps` is checked every iteration. Exceeded → `MachineError::MaxStepsExceeded`.
- **Streaming** — `runner.stream(request, config)` emits typed lifecycle events, heartbeats, and machine-owned signals through a bounded channel.
- **Cancellation** — `runner.cancel_run(run_id)` notifies the active run and can drop an in-flight transition.
- **Typed trace** — every step transition is recorded as `{"step": "Plan", "result": "next"}`.
- **No allocation noise** — `MachineState` has a blanket impl for any `Serialize + DeserializeOwned` type. Your state doesn't need a custom `to_json`/`from_json` method.

## Checkpoint backends

```rust
// In-memory (tests, dev)
let runner = Runner::new(agent, Arc::new(MemorySaver::new()));

// Postgres (production)
let pool = deadpool_postgres::Pool::new(/* ... */);
let saver = PostgresSaver::new(pool);
saver.ensure_schema().await?;
let runner = Runner::new(agent, Arc::new(saver));
```

`CheckpointRecord` is versioned (`version: 1`) — future format migrations are explicit, not silent.

## Error handling

```rust
pub enum MachineError {
    CheckpointDb(Box<dyn Error>),     // transport — DB is down
    CheckpointPool(String),           // transport — pool exhausted
    Serialization(serde_json::Error), // domain — your state type changed
    Deserialization(serde_json::Error),
    MaxStepsExceeded { max: u32 },    // lifecycle — agent looped
    Cancelled,                        // lifecycle — caller gave up
    Transition(Box<dyn Error>),       // domain — your step logic failed
    // ...
}
```

Transport errors carry source context. Domain errors are actionable. No `anyhow` in library code. No opaque strings.

## Philosophy

typemach is not a framework. It doesn't know about LLMs, tools, prompts, or agents. It knows about **typed state machines with checkpoint**. You bring the agent logic; it handles the runtime.

- **No builder pattern** — you write `match step { ... }`, not `add_node().add_edge().compile()`.
- **No magic strings** — every step is a variant in your enum. Miss a branch → compiler error.
- **No hidden state** — checkpoint records are transparent JSON. You can inspect them in `psql`.
- **One trait** — if you can `impl Machine`, you're done.

## License

MIT OR Apache-2.0
