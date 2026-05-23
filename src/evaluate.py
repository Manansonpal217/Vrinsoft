"""Evaluation: measure the trained model on the held-out test split.

Metrics
-------
* Perplexity  - teacher-forced; lower is better (how well the model predicts
                the reference reply token by token).
* BLEU-4      - corpus BLEU of greedy generations vs. the reference replies.
* Distinct-1/2- ratio of unique uni-/bi-grams in the generations; a low value
                means the model collapsed to a few generic replies.

Results are printed and saved to artifacts/eval_metrics.json.

Run:  python -m src.evaluate  [--limit N]
"""
import argparse
import json

import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from src import config
from src.dataset import load_pairs, split_pairs, vectorize
from src.inference import get_generator
from src.text_cleaning import tokenize
from src.train import masked_loss


def compute_perplexity(gen, test_pairs, batch_size=64):
    """Teacher-forced perplexity over the test split."""
    enc, din, dout = vectorize(test_pairs, gen.vocab)
    ds = tf.data.Dataset.from_tensor_slices((enc, din, dout)).batch(batch_size)
    total_loss, total_tokens = 0.0, 0.0
    for e, di, do in ds:
        enc_mask = tf.cast(tf.not_equal(e, config.PAD_ID), tf.float32)
        enc_out, hidden = gen.encoder(e, training=False)
        for t in range(gen.max_out):
            logits, hidden, _ = gen.decoder(
                di[:, t:t + 1], hidden, enc_out, enc_mask, training=False
            )
            step_loss, step_tokens = masked_loss(do[:, t], logits)
            total_loss += float(step_loss)
            total_tokens += float(step_tokens)
    return float(np.exp(total_loss / max(total_tokens, 1.0)))


def distinct_n(token_lists, n):
    """Fraction of unique n-grams across all generated replies."""
    total, unique = 0, set()
    for toks in token_lists:
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model.")
    parser.add_argument("--limit", type=int, default=1500,
                        help="max test pairs to score (BLEU generation is slow)")
    args = parser.parse_args()

    gen = get_generator()
    _, _, test_pairs = split_pairs(load_pairs())
    if args.limit:
        test_pairs = test_pairs[:args.limit]
    print(f"[eval] scoring {len(test_pairs):,} test pairs ...")

    # ---- perplexity (teacher-forced) ------------------------------------
    ppl = compute_perplexity(gen, test_pairs)
    print(f"[eval] test perplexity = {ppl:.2f}")

    # ---- BLEU + diversity over greedy generations -----------------------
    smooth = SmoothingFunction().method1
    references, hypotheses, hyp_tokens, samples = [], [], [], []
    for i, (q_tok, r_tok) in enumerate(test_pairs):
        query = " ".join(q_tok)
        generated = gen.reply(query, decoding="greedy")
        gen_tok = tokenize(generated)
        references.append([r_tok])
        hypotheses.append(gen_tok)
        hyp_tokens.append(gen_tok)
        if i < 12:
            samples.append({
                "query": query,
                "reference": " ".join(r_tok),
                "generated": generated,
            })
        if (i + 1) % 250 == 0:
            print(f"  generated {i + 1}/{len(test_pairs)} ...", flush=True)

    bleu = corpus_bleu(references, hypotheses, smoothing_function=smooth)
    d1 = distinct_n(hyp_tokens, 1)
    d2 = distinct_n(hyp_tokens, 2)
    avg_len = float(np.mean([len(t) for t in hyp_tokens]))

    metrics = {
        "test_pairs": len(test_pairs),
        "perplexity": round(ppl, 3),
        "bleu4": round(bleu, 4),
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "avg_reply_tokens": round(avg_len, 2),
        "samples": samples,
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(" EVALUATION SUMMARY")
    print("=" * 60)
    print(f" test pairs        : {metrics['test_pairs']:,}")
    print(f" perplexity        : {metrics['perplexity']}")
    print(f" BLEU-4            : {metrics['bleu4']}")
    print(f" distinct-1        : {metrics['distinct_1']}")
    print(f" distinct-2        : {metrics['distinct_2']}")
    print(f" avg reply tokens  : {metrics['avg_reply_tokens']}")
    print("=" * 60)
    print("\n sample generations:")
    for s in samples[:6]:
        print(f"  Q   : {s['query']}")
        print(f"  REF : {s['reference']}")
        print(f"  GEN : {s['generated']}\n")
    print(f"[eval] metrics saved to {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
