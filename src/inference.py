"""Inference: load the trained model and generate support replies.

`ReplyGenerator` reconstructs the encoder/decoder from the saved config and
weights, then decodes a reply for a new customer message either greedily or
with temperature / top-k sampling.
"""
import json
import os

import numpy as np
import tensorflow as tf

from src import config
from src.dataset import Vocab, _pad
from src.model import build_models
from src.text_cleaning import clean_and_tokenize

tf.get_logger().setLevel("ERROR")

# Tokens the decoder must never emit.
_BLOCKED_IDS = (config.PAD_ID, config.SOS_ID, config.UNK_ID)


class ModelNotTrainedError(RuntimeError):
    """Raised when artifacts are missing - i.e. training has not run yet."""


class ReplyGenerator:
    """Loads a trained model once and generates replies for new messages."""

    def __init__(self, artifacts_dir=config.ARTIFACTS_DIR):
        cfg_path = os.path.join(artifacts_dir, "model_config.json")
        for path in (cfg_path, config.VOCAB_PATH,
                     config.ENCODER_WEIGHTS, config.DECODER_WEIGHTS):
            if not os.path.exists(path):
                raise ModelNotTrainedError(
                    f"Missing artifact: {path}\n"
                    f"Run `python -m src.preprocess` then `python -m src.train`."
                )

        with open(cfg_path) as f:
            self.cfg = json.load(f)
        self.vocab = Vocab.load(config.VOCAB_PATH)
        self.max_in = self.cfg["max_input_len"]
        self.max_out = self.cfg["max_output_len"]

        # Rebuild the architecture, then materialise weights with a dummy
        # forward pass before loading the trained parameters.
        self.encoder, self.decoder = build_models(self.cfg["vocab_size"])
        self._build_once()
        self.encoder.load_weights(config.ENCODER_WEIGHTS)
        self.decoder.load_weights(config.DECODER_WEIGHTS)

    def _build_once(self):
        units = self.cfg["units"]
        dummy_q = tf.zeros((1, self.max_in), dtype=tf.int32)
        enc_out, hidden = self.encoder(dummy_q)
        enc_mask = tf.ones((1, self.max_in), dtype=tf.float32)
        self.decoder(tf.zeros((1, 1), dtype=tf.int32), hidden, enc_out, enc_mask)

    # ------------------------------------------------------------------
    def _encode_query(self, text):
        tokens = clean_and_tokenize(text)
        ids = self.vocab.encode(tokens)
        ids = _pad(ids, self.max_in, pre=True)
        return tf.constant([ids], dtype=tf.int32)

    def _next_id(self, logits, decoding, temperature, top_k):
        """Pick the next token id from the decoder logits."""
        logits = logits.numpy().astype(np.float64)
        for blocked in _BLOCKED_IDS:
            logits[blocked] = -1e9
        if decoding == "greedy":
            return int(np.argmax(logits))
        # temperature + top-k sampling for more varied replies
        logits = logits / max(temperature, 1e-6)
        k = min(top_k, len(logits))
        top_idx = np.argpartition(logits, -k)[-k:]
        top_logits = logits[top_idx]
        probs = np.exp(top_logits - top_logits.max())
        probs /= probs.sum()
        return int(np.random.choice(top_idx, p=probs))

    def reply(self, text, decoding=config.DEFAULT_DECODING,
              temperature=config.TEMPERATURE, top_k=config.TOP_K,
              max_len=None):
        """Generate a support reply for one customer message."""
        max_len = max_len or self.max_out
        enc_in = self._encode_query(text)
        enc_mask = tf.cast(tf.not_equal(enc_in, config.PAD_ID), tf.float32)
        enc_out, hidden = self.encoder(enc_in)

        token = tf.constant([[config.SOS_ID]], dtype=tf.int32)
        out_ids = []
        for _ in range(max_len):
            logits, hidden, _ = self.decoder(token, hidden, enc_out, enc_mask)
            next_id = self._next_id(logits[0], decoding, temperature, top_k)
            if next_id == config.EOS_ID:
                break
            out_ids.append(next_id)
            token = tf.constant([[next_id]], dtype=tf.int32)

        reply = self.vocab.decode(out_ids)
        return reply if reply.strip() else "could you share a few more details?"


# Module-level lazy singleton so the API / chat reuse one loaded model.
_GENERATOR = None


def get_generator():
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = ReplyGenerator()
    return _GENERATOR


if __name__ == "__main__":  # quick manual check
    gen = get_generator()
    for q in ["my internet has been down all day, please help",
              "how do i reset my password?",
              "your app keeps crashing when i open it"]:
        print(f"Q: {q}")
        print(f"A: {gen.reply(q)}\n")
