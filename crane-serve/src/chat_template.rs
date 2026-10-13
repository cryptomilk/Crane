//! Chat template formatting.
//!
//! Converts OpenAI-style `[{role, content}]` messages into the prompt string
//! expected by each model family.  Two strategies:
//!
//! * **[`AutoChatTemplate`]** — uses the Jinja `chat_template` from
//!   `tokenizer_config.json` (works for Qwen, Llama, Mistral, …).
//! * **[`HunyuanChatTemplate`]** — hardcoded template for Hunyuan models.

use crate::openai_api::{ChatMessage, Tool};
use crate::reasoning::ThinkingOptions;
use crane_core::autotokenizer::AutoTokenizer;

// ─────────────────────────────────────────────────────────────
//  Trait
// ─────────────────────────────────────────────────────────────

/// Everything beyond the messages that a template may read.
#[derive(Default)]
pub struct RenderOptions<'a> {
    pub thinking: ThinkingOptions,
    /// Tool specs, handed to the template verbatim. `None` and an empty slice
    /// both mean "render no tool block".
    pub tools: Option<&'a [Tool]>,
}

impl RenderOptions<'_> {
    fn tools_json(&self) -> Option<serde_json::Value> {
        let tools = self.tools?;
        (!tools.is_empty()).then(|| serde_json::json!(tools))
    }
}

/// Formats chat messages into a model-specific prompt string.
pub trait ChatTemplateProcessor: Send + Sync {
    /// # Errors
    ///
    /// Returns an error if the messages cannot be rendered into a prompt,
    /// e.g. because the underlying template engine rejects the input.
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String>;

    /// Render with reasoning controls and tool specs. Defaults to ignoring
    /// both, which is correct for templates that support neither (Hunyuan).
    ///
    /// # Errors
    ///
    /// Returns an error if the messages cannot be rendered into a prompt,
    /// e.g. because the underlying template engine rejects the input or the
    /// requested reasoning options.
    fn apply_with(
        &self,
        messages: &[ChatMessage],
        _opts: &RenderOptions<'_>,
    ) -> Result<String, String> {
        self.apply(messages)
    }
}

/// Convert one message into the object shape the Jinja templates expect.
///
/// More than `{role, content}`: a tool-calling turn has to replay the
/// assistant's `tool_calls` so the template can re-render them, otherwise the
/// model sees its own call vanish from the history and calls again forever.
///
/// `arguments` is translated from `OpenAI`'s JSON *string* into an object,
/// because the Qwen templates iterate it (`tool_call.arguments|items`) and a
/// string would fail that filter.
fn message_to_json(m: &ChatMessage) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("role".into(), serde_json::json!(m.role));
    obj.insert("content".into(), serde_json::json!(m.text_content()));

    if let Some(calls) = &m.tool_calls {
        let rendered: Vec<serde_json::Value> = calls
            .iter()
            .map(|c| {
                let args = serde_json::from_str::<serde_json::Value>(&c.function.arguments)
                    .unwrap_or_else(|_| serde_json::json!(c.function.arguments));
                serde_json::json!({
                    "id": c.id,
                    "type": c.kind,
                    "function": { "name": c.function.name, "arguments": args },
                })
            })
            .collect();
        obj.insert("tool_calls".into(), serde_json::Value::Array(rendered));
    }
    for (key, val) in [("tool_call_id", &m.tool_call_id), ("name", &m.name)] {
        if let Some(v) = val {
            obj.insert(key.into(), serde_json::json!(v));
        }
    }
    serde_json::Value::Object(obj)
}

// ─────────────────────────────────────────────────────────────
//  AutoChatTemplate (Jinja-based)
// ─────────────────────────────────────────────────────────────

/// Uses [`AutoTokenizer`]'s Jinja `chat_template` from `tokenizer_config.json`.
pub struct AutoChatTemplate {
    tokenizer: AutoTokenizer,
}

impl AutoChatTemplate {
    /// # Errors
    ///
    /// Returns an error if the tokenizer fails to load from `model_path`.
    pub fn new(model_path: &str) -> Result<Self, String> {
        let tokenizer = AutoTokenizer::from_pretrained(model_path, None)
            .map_err(|e| format!("Failed to load AutoTokenizer: {e}"))?;
        Ok(Self { tokenizer })
    }
}

impl ChatTemplateProcessor for AutoChatTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String> {
        self.apply_with(messages, &RenderOptions::default())
    }

    fn apply_with(
        &self,
        messages: &[ChatMessage],
        opts: &RenderOptions<'_>,
    ) -> Result<String, String> {
        let template_messages: Vec<serde_json::Value> =
            messages.iter().map(message_to_json).collect();

        self.tokenizer
            .apply_chat_template_full(
                &template_messages,
                opts.tools_json().as_ref(),
                true,
                opts.thinking.enable_thinking,
                opts.thinking.reasoning_effort.as_deref(),
            )
            // Qwen 3.6+ templates `raise_exception` on an unsupported
            // reasoning_effort, so this is a user-input error, not a bug.
            .map_err(|e| format!("Chat template error: {e}"))
    }
}

// ─────────────────────────────────────────────────────────────
//  HunyuanChatTemplate (hardcoded)
// ─────────────────────────────────────────────────────────────

/// Hardcoded chat template for Hunyuan Dense models.
pub struct HunyuanChatTemplate;

impl ChatTemplateProcessor for HunyuanChatTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String> {
        const BOS: &str = "<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>";
        const USER: &str = "<\u{ff5c}hy_User\u{ff5c}>";
        const ASSISTANT: &str = "<\u{ff5c}hy_Assistant\u{ff5c}>";
        const EOS: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}2\u{ff5c}>";
        const SEP: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}3\u{ff5c}>";

        let mut result = String::new();
        result.push_str(BOS);

        let (system_msg, loop_messages) = if !messages.is_empty() && messages[0].role == "system" {
            (Some(messages[0].text_content()), &messages[1..])
        } else {
            (None, &messages[..])
        };

        if let Some(sys) = system_msg {
            result.push_str(&sys);
            result.push_str(SEP);
        }

        for msg in loop_messages {
            match msg.role.as_str() {
                "user" => {
                    result.push_str(USER);
                    result.push_str(&msg.text_content());
                },
                "assistant" => {
                    result.push_str(ASSISTANT);
                    result.push_str(&msg.text_content());
                    result.push_str(EOS);
                },
                _ => {},
            }
        }

        result.push_str(ASSISTANT);
        Ok(result)
    }
}
