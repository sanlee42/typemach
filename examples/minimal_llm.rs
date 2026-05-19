use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, SystemTime, SystemTimeError, UNIX_EPOCH};
use thiserror::Error;
use typemach::{
    Machine, MachineError, MemorySaver, RunCommand, RunContext, RunId, RunRequest, RunStreamEvent,
    Runner, RuntimeLimits, SessionId, StepResult, StreamConfig, ThreadId, Transition,
};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
const STREAM_CHANNEL_CAPACITY: usize = 16;
const MAX_STEPS: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Step {
    Answer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct State {
    prompt: String,
    answer: Option<String>,
}

#[derive(Debug, Clone)]
struct PromptInput {
    prompt: String,
}

#[derive(Debug, Serialize)]
enum Signal {
    ModelStarted { model: String },
    AssistantDelta { text: String },
}

#[derive(Debug, Serialize)]
struct Answer {
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Interrupt {
    question: String,
}

struct MinimalLlmMachine {
    client: OpenAiChat,
}

#[async_trait]
impl Machine for MinimalLlmMachine {
    type Step = Step;
    type State = State;
    type Input = PromptInput;
    type Signal = Signal;
    type Output = Answer;
    type Interrupt = Interrupt;

    fn start_step(&self) -> Self::Step {
        Step::Answer
    }

    fn new_state(
        &self,
        input: &Self::Input,
        _previous: Option<&Self::State>,
        _snapshot: Option<&serde_json::Value>,
    ) -> Result<Self::State, MachineError> {
        Ok(State {
            prompt: input.prompt.clone(),
            answer: None,
        })
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
            Step::Answer => {
                ctx.emit(Signal::ModelStarted {
                    model: self.client.model().to_string(),
                })
                .await?;

                let answer = self
                    .client
                    .stream_answer(&state.prompt, |text| async move {
                        ctx.emit(Signal::AssistantDelta { text }).await
                    })
                    .await?;

                state.answer = Some(answer.clone());
                Ok(Transition::Complete(Answer { text: answer }))
            }
        }
    }
}

#[derive(Clone)]
struct OpenAiChat {
    http: reqwest::Client,
    config: LlmConfig,
}

impl OpenAiChat {
    fn new(config: LlmConfig) -> Result<Self, ExampleError> {
        let http = reqwest::Client::builder()
            .build()
            .map_err(ExampleError::BuildHttpClient)?;
        Ok(Self { http, config })
    }

    fn model(&self) -> &str {
        &self.config.model
    }

    async fn stream_answer<F, Fut>(
        &self,
        prompt: &str,
        mut on_delta: F,
    ) -> Result<String, MachineError>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = Result<(), MachineError>>,
    {
        let response = self
            .open_stream(prompt)
            .await
            .map_err(MachineError::transition)?;
        let mut chunks = response.bytes_stream();
        let mut line_buffer = Vec::new();
        let mut answer = String::new();

        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.map_err(|err| MachineError::transition(ExampleError::Stream(err)))?;
            for line in split_sse_lines(&mut line_buffer, chunk.as_ref())
                .map_err(MachineError::transition)?
            {
                if handle_sse_line(&line, &mut answer, &mut on_delta).await? {
                    return require_answer(answer);
                }
            }
        }

        if !line_buffer.is_empty() {
            let line = String::from_utf8(line_buffer)
                .map_err(|err| MachineError::transition(ExampleError::InvalidStreamUtf8(err)))?;
            if handle_sse_line(
                line.trim_end_matches(['\r', '\n']),
                &mut answer,
                &mut on_delta,
            )
            .await?
            {
                return require_answer(answer);
            }
        }

        Err(MachineError::transition(ExampleError::MissingDone))
    }

    async fn open_stream(&self, prompt: &str) -> Result<reqwest::Response, ExampleError> {
        let request = ChatRequest {
            model: &self.config.model,
            stream: true,
            messages: [ChatMessage {
                role: "user",
                content: prompt,
            }],
        };

        let response = self
            .http
            .post(self.config.chat_completions_url())
            .bearer_auth(&self.config.api_key)
            .json(&request)
            .send()
            .await
            .map_err(ExampleError::Request)?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        let body = response.text().await.map_err(ExampleError::Request)?;
        Err(ExampleError::HttpStatus { status, body })
    }
}

#[derive(Clone)]
struct LlmConfig {
    api_key: String,
    api_base_url: String,
    model: String,
}

impl LlmConfig {
    fn from_env() -> Result<Self, ExampleError> {
        let api_key = required_env("OPENAI_API_KEY")?;
        let api_base_url = required_env("OPENAI_BASE_URL")?
            .trim_end_matches('/')
            .to_string();
        if api_base_url.is_empty() {
            return Err(ExampleError::MissingEnv("OPENAI_BASE_URL"));
        }
        let model = required_env("OPENAI_MODEL")?;
        Ok(Self {
            api_key,
            api_base_url,
            model,
        })
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.api_base_url)
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    stream: bool,
    messages: [ChatMessage<'a>; 1],
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatChunk {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    delta: ChatDelta,
}

#[derive(Deserialize)]
struct ChatDelta {
    content: Option<String>,
}

#[derive(Debug, Error)]
enum ExampleError {
    #[error("missing required environment variable {0}")]
    MissingEnv(&'static str),

    #[error("prompt is required as a command-line argument")]
    MissingPrompt,

    #[error("failed to read system clock: {0}")]
    SystemClock(#[source] SystemTimeError),

    #[error("failed to build HTTP client: {0}")]
    BuildHttpClient(#[source] reqwest::Error),

    #[error("OpenAI-compatible request failed: {0}")]
    Request(#[source] reqwest::Error),

    #[error("OpenAI-compatible stream failed: {0}")]
    Stream(#[source] reqwest::Error),

    #[error("OpenAI-compatible endpoint returned {status}: {body}")]
    HttpStatus {
        status: reqwest::StatusCode,
        body: String,
    },

    #[error("stream line was not valid UTF-8: {0}")]
    InvalidStreamUtf8(#[source] std::string::FromUtf8Error),

    #[error("stream chunk was not valid JSON: {0}")]
    InvalidChunkJson(#[source] serde_json::Error),

    #[error("stream ended before data: [DONE]")]
    MissingDone,

    #[error("model returned an empty answer")]
    EmptyAnswer,

    #[error("machine interrupted unexpectedly: {0}")]
    UnexpectedInterrupt(String),

    #[error("event stream ended before a terminal event")]
    MissingTerminalEvent,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    async_rt::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = prompt_from_args()?;
    let config = LlmConfig::from_env()?;
    let machine = MinimalLlmMachine {
        client: OpenAiChat::new(config)?,
    };
    let runner = Runner::new(machine, Arc::new(MemorySaver::new()));
    let mut events = runner.stream(build_request(prompt)?, stream_config());

    while let Some(event) = events.next_event().await {
        if handle_event(event)? {
            return Ok(());
        }
    }

    Err(Box::new(ExampleError::MissingTerminalEvent) as Box<dyn std::error::Error>)
}

fn stream_config() -> StreamConfig {
    StreamConfig {
        heartbeat_interval: HEARTBEAT_INTERVAL,
        channel_capacity: STREAM_CHANNEL_CAPACITY,
    }
}

fn build_request(prompt: String) -> Result<RunRequest<PromptInput>, ExampleError> {
    let suffix = run_suffix()?;
    Ok(RunRequest {
        run_id: RunId::from(format!("minimal-llm-{suffix}")),
        session_id: SessionId::from(format!("minimal-llm-session-{suffix}")),
        thread_id: ThreadId::from(format!("minimal-llm-thread-{suffix}")),
        command: RunCommand::Start,
        input: PromptInput { prompt },
        snapshot: None,
        runtime_limits: RuntimeLimits {
            max_steps: MAX_STEPS,
            allow_clarification: false,
        },
    })
}

fn prompt_from_args() -> Result<String, ExampleError> {
    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if prompt.trim().is_empty() {
        return Err(ExampleError::MissingPrompt);
    }
    Ok(prompt)
}

fn required_env(name: &'static str) -> Result<String, ExampleError> {
    let value = std::env::var(name).map_err(|_| ExampleError::MissingEnv(name))?;
    if value.trim().is_empty() {
        return Err(ExampleError::MissingEnv(name));
    }
    Ok(value)
}

fn run_suffix() -> Result<String, ExampleError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(ExampleError::SystemClock)?;
    Ok(elapsed.as_millis().to_string())
}

fn split_sse_lines(buffer: &mut Vec<u8>, chunk: &[u8]) -> Result<Vec<String>, ExampleError> {
    buffer.extend_from_slice(chunk);
    let mut lines = Vec::new();

    while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
        let mut raw_line = buffer.drain(..=newline).collect::<Vec<_>>();
        while matches!(raw_line.last(), Some(b'\n' | b'\r')) {
            raw_line.pop();
        }
        lines.push(String::from_utf8(raw_line).map_err(ExampleError::InvalidStreamUtf8)?);
    }

    Ok(lines)
}

async fn handle_sse_line<F, Fut>(
    line: &str,
    answer: &mut String,
    on_delta: &mut F,
) -> Result<bool, MachineError>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<(), MachineError>>,
{
    let Some(payload) = line.strip_prefix("data:").map(str::trim) else {
        return Ok(false);
    };
    if payload.is_empty() {
        return Ok(false);
    }
    if payload == "[DONE]" {
        return Ok(true);
    }

    let chunk: ChatChunk = serde_json::from_str(payload)
        .map_err(|err| MachineError::transition(ExampleError::InvalidChunkJson(err)))?;
    for choice in chunk.choices {
        if let Some(text) = choice.delta.content {
            if text.is_empty() {
                continue;
            }
            answer.push_str(&text);
            on_delta(text).await?;
        }
    }
    Ok(false)
}

fn require_answer(answer: String) -> Result<String, MachineError> {
    if answer.trim().is_empty() {
        return Err(MachineError::transition(ExampleError::EmptyAnswer));
    }
    Ok(answer)
}

fn handle_event(
    event: RunStreamEvent<Step, Signal, Answer, Interrupt>,
) -> Result<bool, Box<dyn std::error::Error>> {
    match event {
        RunStreamEvent::Started { run_id, .. } => {
            println!("run started: {run_id}");
        }
        RunStreamEvent::Heartbeat { .. } => {
            println!("\nheartbeat");
        }
        RunStreamEvent::StepStarted { step, step_count } => {
            println!("step started: {step:?} #{step_count}");
        }
        RunStreamEvent::StepFinished { step, result } => {
            print_step_finished(step, result);
        }
        RunStreamEvent::Signal { signal } => {
            print_signal(signal)?;
        }
        RunStreamEvent::Interrupted { interrupt, .. } => {
            return Err(Box::new(ExampleError::UnexpectedInterrupt(
                interrupt.question,
            )));
        }
        RunStreamEvent::Completed { output, .. } => {
            println!("\n\nfinal:\n{}", output.text);
            return Ok(true);
        }
        RunStreamEvent::Failed { error } => {
            return Err(Box::new(error));
        }
        RunStreamEvent::Cancelled => {
            return Err(Box::new(MachineError::Cancelled));
        }
    }
    Ok(false)
}

fn print_step_finished(step: Step, result: StepResult) {
    println!("\nstep finished: {step:?} {result:?}");
}

fn print_signal(signal: Signal) -> Result<(), std::io::Error> {
    match signal {
        Signal::ModelStarted { model } => {
            println!("model started: {model}");
        }
        Signal::AssistantDelta { text } => {
            print!("{text}");
            std::io::stdout().flush()?;
        }
    }
    Ok(())
}
