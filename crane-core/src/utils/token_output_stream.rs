use candle_core::Result;

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    pub tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    /// Text that has already been safely sent to the client. Keeping this as
    /// text (rather than token indices) lets us wait when a UTF-8 character is
    /// split across byte-fallback tokens.
    emitted: String,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            emitted: String::new(),
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => candle_core::bail!("cannot decode: {err}"),
        }
    }

    /// Return the decodable prefix and withhold an incomplete byte sequence.
    /// Tokenizers use U+FFFD while decoding such a partial sequence, which
    /// must never be forwarded to an SSE client — a later token completes it.
    fn safe_delta(&mut self, decoded: &str) -> Option<String> {
        let safe = match decoded.find('\u{fffd}') {
            Some(index) => &decoded[..index],
            None => decoded,
        };
        if !safe.starts_with(&self.emitted) {
            // A tokenizer should only extend a decoded prefix. If it does not,
            // avoid emitting corrupt text and let the final decode recover it.
            return None;
        }
        let delta = &safe[self.emitted.len()..];
        self.emitted = safe.to_string();
        (!delta.is_empty()).then(|| delta.to_string())
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    /// # Errors
    ///
    /// Returns an error if the tokenizer fails to decode the accumulated
    /// token sequence.
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        self.tokens.push(token);
        let decoded = self.decode(&self.tokens)?;
        Ok(self.safe_delta(&decoded))
    }

    /// # Errors
    ///
    /// Returns an error if the tokenizer fails to decode the accumulated
    /// token sequence.
    pub fn decode_rest(&mut self) -> Result<Option<String>> {
        let decoded = self.decode(&self.tokens)?;
        Ok(self.safe_delta(&decoded))
    }

    /// # Errors
    ///
    /// Returns an error if the tokenizer fails to decode the accumulated
    /// token sequence.
    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.emitted.clear();
    }
}
