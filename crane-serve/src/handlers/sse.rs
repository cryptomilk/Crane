//! SSE stream builders for OpenAI-compatible and native streaming.

use std::convert::Infallible;

use axum::response::sse::Event;
use futures::stream::Stream;
use tokio::sync::mpsc;

use crate::engine::EngineResponse;
use crate::now_epoch;
use crate::openai_api::*;
use crate::sglang_api::*;

// ─────────────────────────────────────────────────────────────
//  Chat completions SSE
// ─────────────────────────────────────────────────────────────

/// `splitter` routes each token delta to `content` or `reasoning_content`; its
/// starting state comes from the rendered prompt (see
/// [`crate::reasoning::ReasoningSplitter::for_prompt`]).
pub fn make_chat_sse_stream(
    request_id: String,
    model_name: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
    include_usage: bool,
    mut splitter: crate::reasoning::ReasoningSplitter,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let created = now_epoch();

    async_stream::stream! {
        // Role announcement chunk.
        let first_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta::role("assistant"),
                finish_reason: None,
            }],
            usage: None,
        };
        yield Ok(Event::default().json_data(&first_chunk).unwrap());

        let mut _prompt_tokens = 0usize;
        let mut _completion_tokens = 0usize;

        // A token may straddle a `</think>` tag, so one token can yield both a
        // reasoning and a content delta — or neither, while a partial tag is
        // buffered.
        // Tool-call markup is withheld from `content` and released as a single
        // `tool_calls` delta at the end — see `crate::tools::ToolCallStream`.
        let mut tool_stream = crate::tools::ToolCallStream::default();

        macro_rules! emit_deltas {
            ($reasoning:expr, $content:expr) => {{
                let (reasoning, content) = ($reasoning, $content);
                let content = tool_stream.push(&content);
                for delta in [
                    (!reasoning.is_empty()).then(|| ChunkDelta {
                        role: None,
                        content: None,
                        reasoning_content: Some(reasoning),
                        tool_calls: None,
                    }),
                    (!content.is_empty()).then(|| ChunkDelta::content(content)),
                ]
                .into_iter()
                .flatten()
                {
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice { index: 0, delta, finish_reason: None }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                }
            }};
        }

        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    _completion_tokens += 1;
                    let (reasoning, content) = splitter.push(&text);
                    emit_deltas!(reasoning, content);
                }
                EngineResponse::Finished {
                    finish_reason,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    ..
                } => {
                    _prompt_tokens = pt;
                    _completion_tokens = ct;

                    // Flush any text held back as a possible partial tag.
                    let (reasoning, content) = splitter.finish();
                    emit_deltas!(reasoning, content);

                    // Then release whatever the tool filter withheld: trailing
                    // prose first, then the completed calls.
                    let (tail, calls) = tool_stream.finish();
                    let has_calls = !calls.is_empty();
                    for delta in [
                        (!tail.trim().is_empty()).then(|| ChunkDelta::content(tail)),
                        has_calls.then(|| ChunkDelta::tool_calls(calls)),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        let chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice { index: 0, delta, finish_reason: None }],
                            usage: None,
                        };
                        yield Ok(Event::default().json_data(&chunk).unwrap());
                    }

                    // A tool turn ends as `tool_calls`, not `stop`, so clients
                    // know to run the tools instead of showing the text.
                    let finish_reason = if has_calls {
                        "tool_calls".to_string()
                    } else {
                        finish_reason
                    };
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta::empty(),
                            finish_reason: Some(finish_reason),
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());

                    if include_usage {
                        let usage_chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens: _prompt_tokens,
                                completion_tokens: _completion_tokens,
                                total_tokens: _prompt_tokens + _completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().json_data(&usage_chunk).unwrap());
                    }

                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: e,
                            r#type: "server_error".into(),
                            code: None,
                        },
                    };
                    yield Ok(Event::default().json_data(&error_response).unwrap());
                    break;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Text completions SSE
// ─────────────────────────────────────────────────────────────

pub fn make_completion_sse_stream(
    request_id: String,
    model_name: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let created = now_epoch();

    async_stream::stream! {
        let mut _prompt_tokens = 0usize;
        let mut _completion_tokens = 0usize;

        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    _completion_tokens += 1;
                    let chunk = CompletionChunk {
                        id: request_id.clone(),
                        object: "text_completion".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text,
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                }
                EngineResponse::Finished {
                    finish_reason,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    ..
                } => {
                    _prompt_tokens = pt;
                    _completion_tokens = ct;

                    let chunk = CompletionChunk {
                        id: request_id.clone(),
                        object: "text_completion".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text: String::new(),
                            finish_reason: Some(finish_reason),
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());

                    if include_usage {
                        let usage_chunk = CompletionChunk {
                            id: request_id.clone(),
                            object: "text_completion".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens: _prompt_tokens,
                                completion_tokens: _completion_tokens,
                                total_tokens: _prompt_tokens + _completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().json_data(&usage_chunk).unwrap());
                    }

                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: e,
                            r#type: "server_error".into(),
                            code: None,
                        },
                    };
                    yield Ok(Event::default().json_data(&error_response).unwrap());
                    break;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Native /generate SSE
// ─────────────────────────────────────────────────────────────

pub fn make_generate_sse_stream(
    request_id: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    let chunk = GenerateStreamChunk {
                        text,
                        meta_info: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                }
                EngineResponse::Finished {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                    ..
                } => {
                    let chunk = GenerateStreamChunk {
                        text: String::new(),
                        meta_info: Some(GenerateMetaInfo {
                            id: request_id.clone(),
                            prompt_tokens,
                            completion_tokens,
                            finish_reason,
                        }),
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    let error_response = ErrorResponse {
                        error: ErrorDetail {
                            message: e,
                            r#type: "server_error".into(),
                            code: None,
                        },
                    };
                    yield Ok(Event::default().json_data(&error_response).unwrap());
                    break;
                }
            }
        }
    }
}
