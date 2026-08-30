use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::Value;

use crate::responses::{DecodeFailure, decode_response as decode_responses, responses_request};
use crate::responses_stream::decode_stream as decode_responses_stream;
use crate::{
    AgentConfig, AgentError, AgentModel, ModelRequest, ModelResponse, ModelStream, ReasoningEffort,
    ToolChoice,
};

#[derive(Clone)]
pub struct ConfiguredModel {
    client: reqwest::Client,
    config: AgentConfig,
    endpoint: String,
}

impl ConfiguredModel {
    pub fn new(config: AgentConfig) -> Result<Self, AgentError> {
        validate_config(&config)?;
        // A total-request timeout would kill healthy long streams, so the
        // client only bounds connect latency and the idle gap between
        // chunks; non-streaming calls add a per-request total in attempt().
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .map_err(|err| AgentError::Config(format!("failed to build HTTP client: {err}")))?;
        let endpoint = endpoint(&config.base_url)?;
        Ok(Self {
            client,
            config,
            endpoint,
        })
    }

    pub fn with_client(client: reqwest::Client, config: AgentConfig) -> Result<Self, AgentError> {
        validate_config(&config)?;
        let endpoint = endpoint(&config.base_url)?;
        Ok(Self {
            client,
            config,
            endpoint,
        })
    }
}

#[async_trait]
impl AgentModel for ConfiguredModel {
    async fn next_step(
        &self,
        request: ModelRequest,
        stream: ModelStream,
    ) -> Result<ModelResponse, AgentError> {
        let body = self.request_body(request)?;
        let max_attempts = self.config.max_retries.saturating_add(1);
        let mut attempt = 0_u32;
        loop {
            attempt += 1;
            let emitted_before = stream.emitted();
            let failure = match self.attempt(&body, &stream).await {
                Ok(response) => return Ok(response),
                Err(failure) => failure,
            };
            // Never replay a response the user has partially seen: once a
            // delta went out, the failure is final.
            let streamed = stream.emitted() > emitted_before;
            if !failure.retryable || streamed || attempt >= max_attempts {
                return Err(AgentError::Model(format!(
                    "model request failed after {attempt} attempts: {}",
                    failure.message
                )));
            }
            tokio::time::sleep(retry_delay(attempt, failure.retry_after)).await;
        }
    }
}

struct AttemptFailure {
    message: String,
    retryable: bool,
    retry_after: Option<Duration>,
}

impl ConfiguredModel {
    fn request_body(&self, request: ModelRequest) -> Result<Value, AgentError> {
        serde_json::to_value(responses_request(&self.config, request)?)
            .map_err(|err| AgentError::Model(format!("failed to encode responses request: {err}")))
    }

    async fn attempt(
        &self,
        body: &Value,
        stream: &ModelStream,
    ) -> Result<ModelResponse, AttemptFailure> {
        let headers = headers(&self.config).map_err(|err| AttemptFailure {
            message: err.to_string(),
            retryable: false,
            retry_after: None,
        })?;
        let mut request = self.client.post(&self.endpoint).headers(headers).json(body);
        if !self.config.stream {
            request = request.timeout(Duration::from_secs(self.config.request_timeout_secs));
        }
        let response = request.send().await.map_err(|err| {
            let retryable = retryable_request_error(&err);
            AttemptFailure {
                message: format!("model request failed: {err}"),
                retryable,
                retry_after: None,
            }
        })?;
        let status = response.status();
        if !status.is_success() {
            let retry_after = parse_retry_after(response.headers());
            let body = response
                .text()
                .await
                .unwrap_or_else(|err| format!("failed to read error body: {err}"));
            return Err(AttemptFailure {
                message: format!("model request failed ({status}): {body}"),
                retryable: matches!(status.as_u16(), 408 | 429 | 500 | 502 | 503 | 504),
                retry_after,
            });
        }
        let decoded = match self.config.stream {
            true => decode_responses_stream(response, stream.clone()).await,
            false => decode_responses(response).await,
        };
        decoded.map_err(AttemptFailure::from)
    }
}

impl From<DecodeFailure> for AttemptFailure {
    fn from(failure: DecodeFailure) -> Self {
        Self {
            message: failure.message().to_string(),
            retryable: failure.retryable(),
            retry_after: None,
        }
    }
}

fn retryable_request_error(err: &reqwest::Error) -> bool {
    err.is_connect() || err.is_timeout()
}

/// Server-provided Retry-After (integer seconds form) wins, clamped to 30s;
/// otherwise exponential backoff with full jitter in [base/2, base].
fn retry_delay(attempt: u32, retry_after: Option<Duration>) -> Duration {
    if let Some(retry_after) = retry_after {
        return retry_after.min(Duration::from_secs(30));
    }
    let base = Duration::from_millis(500)
        .saturating_mul(2_u32.saturating_pow(attempt.saturating_sub(1)))
        .min(Duration::from_secs(10));
    let half = base / 2;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.subsec_nanos() as u64)
        .unwrap_or_default();
    half + Duration::from_nanos(nanos % half.as_nanos().max(1) as u64)
}

fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

fn validate_config(config: &AgentConfig) -> Result<(), AgentError> {
    if config.api_key.trim().is_empty() {
        return Err(AgentError::Config("api_key must not be empty".to_string()));
    }
    if config.model.trim().is_empty() {
        return Err(AgentError::Config("model must not be empty".to_string()));
    }
    if config.base_url.trim().is_empty() {
        return Err(AgentError::Config("base_url must not be empty".to_string()));
    }
    if config.request_timeout_secs == 0 {
        return Err(AgentError::Config(
            "request_timeout_secs must be greater than zero".to_string(),
        ));
    }
    endpoint(&config.base_url)?;
    Ok(())
}

fn endpoint(base_url: &str) -> Result<String, AgentError> {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.ends_with("/responses") {
        return Ok(trimmed.to_string());
    }
    let parsed = reqwest::Url::parse(trimmed)
        .map_err(|err| AgentError::Config(format!("base_url is not a valid URL: {err}")))?;
    if parsed.path().trim_end_matches('/').is_empty() {
        return Ok(format!("{trimmed}/responses"));
    }
    Err(AgentError::Config(
        "base_url must be an origin or an explicit /responses endpoint".to_string(),
    ))
}

pub(crate) fn headers(config: &AgentConfig) -> Result<HeaderMap, AgentError> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", config.api_key.trim())).map_err(|err| {
            AgentError::Config(format!("invalid authorization header value: {err}"))
        })?,
    );
    Ok(headers)
}

pub(crate) fn combined_system(config: &AgentConfig, request: &ModelRequest) -> Option<String> {
    let base = config
        .system
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let suffix = request
        .system_suffix
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    match (base, suffix) {
        (Some(base), Some(suffix)) => Some(format!("{base}\n\n{suffix}")),
        (Some(base), None) => Some(base.to_string()),
        (None, Some(suffix)) => Some(suffix.to_string()),
        (None, None) => None,
    }
}

pub(crate) fn tool_choice_value(choice: ToolChoice) -> &'static str {
    match choice {
        ToolChoice::Auto => "auto",
        ToolChoice::Required => "required",
        ToolChoice::None => "none",
    }
}

pub(crate) fn effort_value(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
    }
}

pub(crate) fn encode_arguments(input: &Value) -> Result<String, AgentError> {
    match input {
        Value::Null => Ok("{}".to_string()),
        other => serde_json::to_string(other)
            .map_err(|err| AgentError::Model(format!("failed to encode tool arguments: {err}"))),
    }
}

pub(crate) fn decode_arguments(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()))
}

pub(crate) fn tool_result_content(content: &Value) -> Result<String, AgentError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(value) => Ok(value.clone()),
        other => serde_json::to_string(other)
            .map_err(|err| AgentError::Model(format!("failed to encode tool result: {err}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentMessage, AgentToolSpec, ToolAnnotations};
    use serde_json::json;

    #[test]
    fn retry_delay_backs_off_with_jitter_and_honors_retry_after() {
        for _ in 0..16 {
            let first = retry_delay(1, None);
            assert!(first >= Duration::from_millis(250), "first: {first:?}");
            assert!(first <= Duration::from_millis(500), "first: {first:?}");
            let second = retry_delay(2, None);
            assert!(second >= Duration::from_millis(500), "second: {second:?}");
            assert!(second <= Duration::from_millis(1000), "second: {second:?}");
            let deep = retry_delay(16, None);
            assert!(deep <= Duration::from_secs(10), "deep: {deep:?}");
        }
        assert_eq!(retry_delay(1, Some(Duration::ZERO)), Duration::ZERO);
        assert_eq!(
            retry_delay(1, Some(Duration::from_secs(2))),
            Duration::from_secs(2)
        );
        assert_eq!(
            retry_delay(1, Some(Duration::from_secs(9999))),
            Duration::from_secs(30)
        );
    }

    fn spec() -> AgentToolSpec {
        AgentToolSpec {
            name: "report".to_string(),
            description: "finish".to_string(),
            input_schema: json!({ "type": "object" }),
            output_schema: Value::Null,
            metadata: Value::Null,
            annotations: ToolAnnotations::default(),
        }
    }

    fn request(tools: Vec<AgentToolSpec>) -> ModelRequest {
        ModelRequest {
            messages: vec![AgentMessage::user_text("hi")],
            tools,
            context: Value::Null,
            turn: 1,
            system_suffix: None,
            tool_choice: None,
        }
    }

    #[test]
    fn tool_choice_required_is_sent_when_tools_are_offered() {
        let mut config = AgentConfig::new("k", "deepseek-v4-flash");
        config.tool_choice = Some(ToolChoice::Required);
        let body = serde_json::to_value(responses_request(&config, request(vec![spec()])).unwrap())
            .expect("serialize");
        assert_eq!(body["tool_choice"], "required");
    }

    #[test]
    fn request_tool_choice_overrides_transport_default() {
        let mut config = AgentConfig::new("k", "deepseek-v4-flash");
        config.tool_choice = Some(ToolChoice::Required);
        let mut request = request(vec![spec()]);
        request.tool_choice = Some(ToolChoice::Auto);
        let body =
            serde_json::to_value(responses_request(&config, request).unwrap()).expect("serialize");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn tool_choice_none_is_sent_when_no_tools() {
        let mut config = AgentConfig::new("k", "deepseek-v4-flash");
        config.tool_choice = Some(ToolChoice::Required);
        let mut request = request(vec![]);
        request.tool_choice = Some(ToolChoice::None);
        let body =
            serde_json::to_value(responses_request(&config, request).unwrap()).expect("serialize");
        assert_eq!(body["tool_choice"], "none");
    }

    #[test]
    fn tool_choice_absent_by_default() {
        let config = AgentConfig::new("k", "deepseek-v4-flash");
        let body = serde_json::to_value(responses_request(&config, request(vec![spec()])).unwrap())
            .expect("serialize");
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn stale_chat_endpoint_is_invalid() {
        let err = endpoint("https://api.deepseek.com/chat/completions").expect_err("invalid");
        assert!(err.to_string().contains("explicit /responses endpoint"));
    }
}
