use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

pub(super) struct MockTurn {
    body: String,
}

impl MockTurn {
    pub(super) fn ok(body: String) -> Self {
        Self { body }
    }
}

#[derive(Debug)]
pub(super) struct CapturedRequest {
    body: Value,
}

pub(super) async fn spawn_server(
    turns: Vec<MockTurn>,
) -> (String, Arc<Mutex<Vec<CapturedRequest>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let captured = Arc::new(Mutex::new(Vec::new()));
    let captured_for_task = Arc::clone(&captured);
    tokio::spawn(async move {
        for turn in turns {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let request = read_request(&mut socket).await;
            captured_for_task
                .lock()
                .expect("captured lock")
                .push(request);
            write_response(&mut socket, &turn).await;
        }
    });
    (format!("http://{addr}"), captured)
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> CapturedRequest {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 1024];
    let header_end = loop {
        let n = socket.read(&mut chunk).await.expect("read request");
        assert_ne!(n, 0, "connection closed before headers");
        buffer.extend_from_slice(&chunk[..n]);
        if let Some(index) = find_header_end(&buffer) {
            break index;
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let content_length = content_length(&headers);
    while buffer.len() < header_end + 4 + content_length {
        let n = socket.read(&mut chunk).await.expect("read body");
        assert_ne!(n, 0, "connection closed before body");
        buffer.extend_from_slice(&chunk[..n]);
    }
    let body = &buffer[header_end + 4..header_end + 4 + content_length];
    CapturedRequest {
        body: serde_json::from_slice(body).expect("json body"),
    }
}

async fn write_response(socket: &mut tokio::net::TcpStream, turn: &MockTurn) {
    let response = format!(
        "HTTP/1.1 200\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        turn.body.len(),
        turn.body
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("write response");
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn content_length(headers: &str) -> usize {
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().expect("content length"))
        })
        .expect("content-length")
}

pub(super) fn captured_bodies(captured: &Arc<Mutex<Vec<CapturedRequest>>>) -> Vec<Value> {
    captured
        .lock()
        .expect("captured")
        .iter()
        .map(|request| request.body.clone())
        .collect()
}

pub(super) fn sse(events: impl IntoIterator<Item = Value>) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str("data: ");
        body.push_str(&event.to_string());
        body.push_str("\n\n");
    }
    body.push_str("data: [DONE]\n\n");
    body
}
