"""Train the sequential conversational model.

Custom teacher-forcing training loop for the GRU encoder-decoder + attention.
After every epoch the model is evaluated on a validation split and the best
weights (lowest validation loss) are checkpointed to artifacts/.

Run:  python -m src.train  [--epochs N] [--batch-size N] [--limit N]
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

from src import config
from src.dataset import Vocab, load_pairs, split_pairs, vectorize
from src.model import build_models

# Quieter, reproducible runs.
tf.get_logger().setLevel("ERROR")
tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def masked_loss(real, logits):
    """Sum of cross-entropy over non-padding tokens, plus the token count."""
    mask = tf.cast(tf.not_equal(real, config.PAD_ID), tf.float32)
    step = _loss_obj(real, logits) * mask
    return tf.reduce_sum(step), tf.reduce_sum(mask)


def make_step_fns(encoder, decoder, optimizer):
    """Build the (compiled) train and eval step functions."""

    def _run(enc_in, dec_in, dec_out, training):
        enc_mask = tf.cast(tf.not_equal(enc_in, config.PAD_ID), tf.float32)
        enc_outputs, hidden = encoder(enc_in, training=training)
        total_loss, total_tokens = 0.0, 0.0
        for t in range(config.MAX_OUTPUT_LEN):
            logits, hidden, _ = decoder(
                dec_in[:, t:t + 1], hidden, enc_outputs, enc_mask, training=training
            )
            step_loss, step_tokens = masked_loss(dec_out[:, t], logits)
            total_loss += step_loss
            total_tokens += step_tokens
        return total_loss, total_tokens

    @tf.function
    def train_step(enc_in, dec_in, dec_out):
        with tf.GradientTape() as tape:
            total_loss, total_tokens = _run(enc_in, dec_in, dec_out, training=True)
            mean_loss = total_loss / total_tokens
        variables = encoder.trainable_variables + decoder.trainable_variables
        grads = tape.gradient(mean_loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return total_loss, total_tokens

    @tf.function
    def eval_step(enc_in, dec_in, dec_out):
        return _run(enc_in, dec_in, dec_out, training=False)

    return train_step, eval_step


def run_epoch(dataset, step_fn, n_batches, label):
    """Run one pass over `dataset`, returning (per-token loss, perplexity)."""
    epoch_loss, epoch_tokens = 0.0, 0.0
    t0 = time.time()
    for i, (enc_in, dec_in, dec_out) in enumerate(dataset, start=1):
        loss, tokens = step_fn(enc_in, dec_in, dec_out)
        epoch_loss += float(loss)
        epoch_tokens += float(tokens)
        if i % 50 == 0 or i == n_batches:
            running = epoch_loss / max(epoch_tokens, 1.0)
            print(f"  [{label}] batch {i}/{n_batches}  "
                  f"loss={running:.4f}  ppl={np.exp(running):.1f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    mean_loss = epoch_loss / max(epoch_tokens, 1.0)
    return mean_loss, float(np.exp(mean_loss))


def main():
    parser = argparse.ArgumentParser(description="Train the reply assistant.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of training pairs (quick runs)")
    args = parser.parse_args()

    if not os.path.exists(config.PAIRS_CSV):
        raise FileNotFoundError(
            "artifacts/pairs.csv not found - run `python -m src.preprocess` first."
        )
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

    # ---- data ------------------------------------------------------------
    pairs = load_pairs()
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)
    if args.limit:
        train_pairs = train_pairs[:args.limit]
    print(f"[train] train={len(train_pairs):,}  val={len(val_pairs):,}  "
          f"test={len(test_pairs):,}")

    # Vocabulary is built from the TRAINING split only (no leakage).
    vocab = Vocab.build(q + r for q, r in train_pairs)
    vocab.save(config.VOCAB_PATH)
    print(f"[train] vocabulary size = {vocab.size:,}  (saved to {config.VOCAB_PATH})")

    enc_tr, din_tr, dout_tr = vectorize(train_pairs, vocab)
    enc_va, din_va, dout_va = vectorize(val_pairs, vocab)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((enc_tr, din_tr, dout_tr))
        .shuffle(len(enc_tr), seed=config.RANDOM_SEED)
        .batch(args.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((enc_va, din_va, dout_va))
        .batch(args.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    n_train_batches = len(enc_tr) // args.batch_size
    n_val_batches = max(len(enc_va) // args.batch_size, 1)

    # ---- model -----------------------------------------------------------
    encoder, decoder = build_models(vocab.size)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE, clipnorm=config.CLIPNORM
    )
    train_step, eval_step = make_step_fns(encoder, decoder, optimizer)

    model_cfg = {
        "vocab_size": vocab.size,
        "embedding_dim": config.EMBEDDING_DIM,
        "units": config.UNITS,
        "max_input_len": config.MAX_INPUT_LEN,
        "max_output_len": config.MAX_OUTPUT_LEN,
    }
    with open(config.MODEL_CONFIG_PATH, "w") as f:
        json.dump(model_cfg, f, indent=2)

    # ---- training loop ---------------------------------------------------
    history, best_val = [], float("inf")
    print(f"[train] starting training for {args.epochs} epoch(s)\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_ppl = run_epoch(train_ds, train_step, n_train_batches, "train")
        va_loss, va_ppl = run_epoch(val_ds, eval_step, n_val_batches, "val")
        secs = time.time() - t0
        print(f"  -> train_loss={tr_loss:.4f} ppl={tr_ppl:.1f} | "
              f"val_loss={va_loss:.4f} ppl={va_ppl:.1f} | {secs:.0f}s")

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_ppl": tr_ppl,
            "val_loss": va_loss, "val_ppl": va_ppl, "seconds": secs,
        })
        with open(config.HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

        if va_loss < best_val:
            best_val = va_loss
            encoder.save_weights(config.ENCODER_WEIGHTS)
            decoder.save_weights(config.DECODER_WEIGHTS)
            print(f"  -> new best (val_loss={va_loss:.4f}); weights checkpointed")
        print()

    print(f"[train] done. best val_loss={best_val:.4f}. "
          f"artifacts in {config.ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
