"""Data preprocessing pipeline.

Turns the raw Twitter Customer Support (TWCS) CSV into a clean table of
(customer_query, support_reply) pairs ready for sequential modelling.

Pipeline stages
---------------
1. Read the raw tweets (tweet_id, author_id, inbound, text, in_response_to...).
2. Reconstruct conversations: every support tweet (inbound == False) points at
   the tweet it answers via `in_response_to_tweet_id`.
3. Keep only support replies whose parent is a customer tweet (inbound == True)
   -> that is one (query, reply) training example.
4. Clean the noisy social-media text of both sides.
5. Drop pairs that are too short / too long, de-duplicate, sample.
6. Write the result to artifacts/pairs.csv.

Run:  python -m src.preprocess  [--max-pairs N] [--input PATH]
"""
import argparse
import os

import pandas as pd

from src import config
from src.text_cleaning import clean_text, tokenize


def _ntokens(text: str) -> int:
    return len(tokenize(text))


def build_pairs(input_csv: str, max_pairs: int) -> pd.DataFrame:
    """Reconstruct and clean (query, reply) pairs from the raw TWCS CSV."""
    print(f"[preprocess] reading raw tweets from {input_csv} ...")
    df = pd.read_csv(
        input_csv,
        usecols=["tweet_id", "inbound", "text", "in_response_to_tweet_id"],
        dtype={"tweet_id": "int64", "text": "string"},
    )
    print(f"[preprocess] {len(df):,} raw tweets loaded")

    # `inbound` arrives as the strings "True"/"False".
    df["inbound"] = df["inbound"].astype(str).str.strip().str.lower().eq("true")
    df = df.dropna(subset=["text"])

    # Customer tweets (the queries) and support tweets (the replies).
    customers = (
        df[df["inbound"]][["tweet_id", "text"]]
        .rename(columns={"tweet_id": "query_id", "text": "query"})
    )
    support = df[~df["inbound"]][["in_response_to_tweet_id", "text"]].copy()
    support = support.dropna(subset=["in_response_to_tweet_id"])
    support["in_response_to_tweet_id"] = support["in_response_to_tweet_id"].astype("int64")
    support = support.rename(columns={"text": "reply"})

    # Join each support reply onto the customer tweet it answers.
    pairs = support.merge(
        customers,
        left_on="in_response_to_tweet_id",
        right_on="query_id",
        how="inner",
    )[["query", "reply"]]
    print(f"[preprocess] {len(pairs):,} customer->support pairs reconstructed")

    # Sample a workable amount *before* the (slower) text cleaning.
    oversample = min(len(pairs), int(max_pairs * 1.6) + 1000)
    pairs = pairs.sample(n=oversample, random_state=config.RANDOM_SEED)

    print("[preprocess] cleaning noisy social-media text ...")
    pairs["query"] = pairs["query"].map(clean_text)
    pairs["reply"] = pairs["reply"].map(clean_text)

    # Length filtering keeps the model focused on learnable, well-formed turns.
    q_len = pairs["query"].map(_ntokens)
    r_len = pairs["reply"].map(_ntokens)
    keep = (
        (q_len >= config.MIN_TOKENS)
        & (r_len >= config.MIN_TOKENS)
        & (q_len <= config.MAX_INPUT_LEN)
        & (r_len <= config.MAX_OUTPUT_LEN - 1)  # leave room for <eos>
    )
    pairs = pairs[keep]
    pairs = pairs.drop_duplicates(subset=["query", "reply"])
    print(f"[preprocess] {len(pairs):,} pairs left after length filter + dedup")

    pairs = pairs.head(max_pairs).reset_index(drop=True)
    print(f"[preprocess] keeping {len(pairs):,} pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build (query, reply) pairs.")
    parser.add_argument("--input", default=config.RAW_CSV,
                        help="raw TWCS csv (default: sample_data/twcs/twcs.csv)")
    parser.add_argument("--max-pairs", type=int, default=config.MAX_PAIRS)
    parser.add_argument("--output", default=config.PAIRS_CSV)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Raw dataset not found at {args.input}. "
            f"Point --input at the TWCS csv."
        )

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    pairs = build_pairs(args.input, args.max_pairs)
    pairs.to_csv(args.output, index=False)
    print(f"[preprocess] wrote {args.output}")

    # Show a few examples so the result is easy to eyeball.
    print("\n[preprocess] sample pairs:")
    for _, row in pairs.head(3).iterrows():
        print(f"  Q: {row['query']}")
        print(f"  A: {row['reply']}\n")


if __name__ == "__main__":
    main()
