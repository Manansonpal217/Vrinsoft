"""Vocabulary, sequence encoding and train/val/test splitting.

Converts the cleaned (query, reply) pairs from `preprocess.py` into the
integer tensors the sequential model consumes:

  * encoder input  - the customer query, PRE-padded to MAX_INPUT_LEN so the
                     final GRU state always lands on a real token.
  * decoder input  - <sos> + reply tokens, POST-padded.
  * decoder target - reply tokens + <eos>, POST-padded.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

from src import config
from src.text_cleaning import tokenize


class Vocab:
    """Word <-> integer-id mapping with a fixed special-token block."""

    def __init__(self, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def size(self):
        return len(self.word2idx)

    @classmethod
    def build(cls, token_lists, max_size=config.VOCAB_SIZE):
        """Build a vocabulary from training token lists (most-frequent first)."""
        counter = Counter()
        for toks in token_lists:
            counter.update(toks)

        word2idx = {tok: i for i, tok in enumerate(config.SPECIAL_TOKENS)}
        budget = max_size - len(config.SPECIAL_TOKENS)
        for word, _ in counter.most_common(budget):
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        idx2word = {i: w for w, i in word2idx.items()}
        return cls(word2idx, idx2word)

    def encode(self, tokens):
        """Token list -> id list (out-of-vocabulary tokens become <unk>)."""
        unk = config.UNK_ID
        return [self.word2idx.get(t, unk) for t in tokens]

    def decode(self, ids, strip_special=True):
        """Id list -> readable string."""
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), config.UNK)
            if strip_special and w in (config.PAD, config.SOS, config.EOS):
                continue
            words.append(w)
        return " ".join(words)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            word2idx = json.load(f)["word2idx"]
        idx2word = {i: w for w, i in word2idx.items()}
        return cls(word2idx, idx2word)


def _pad(seq, maxlen, pad_id=config.PAD_ID, pre=False):
    """Pad / truncate a single id sequence to `maxlen`."""
    seq = list(seq)[:maxlen]
    pad = [pad_id] * (maxlen - len(seq))
    return (pad + seq) if pre else (seq + pad)


def load_pairs(path=config.PAIRS_CSV):
    """Read pairs.csv and return a list of (query_tokens, reply_tokens)."""
    df = pd.read_csv(path).dropna()
    return [(tokenize(q), tokenize(r)) for q, r in zip(df["query"], df["reply"])]


def split_pairs(pairs, seed=config.RANDOM_SEED):
    """Deterministic shuffle + train/val/test split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_test = int(len(pairs) * config.TEST_FRACTION)
    n_val = int(len(pairs) * config.VAL_FRACTION)
    test = [pairs[i] for i in idx[:n_test]]
    val = [pairs[i] for i in idx[n_test:n_test + n_val]]
    train = [pairs[i] for i in idx[n_test + n_val:]]
    return train, val, test


def vectorize(pairs, vocab):
    """Turn (query_tokens, reply_tokens) pairs into model-ready int arrays."""
    enc_in, dec_in, dec_out = [], [], []
    for q_tok, r_tok in pairs:
        q_ids = vocab.encode(q_tok)
        r_ids = vocab.encode(r_tok)
        # encoder input: pre-padded query
        enc_in.append(_pad(q_ids, config.MAX_INPUT_LEN, pre=True))
        # decoder input/target: teacher-forcing shifted by one position
        dec_in.append(_pad([config.SOS_ID] + r_ids, config.MAX_OUTPUT_LEN))
        dec_out.append(_pad(r_ids + [config.EOS_ID], config.MAX_OUTPUT_LEN))
    return (
        np.array(enc_in, dtype=np.int32),
        np.array(dec_in, dtype=np.int32),
        np.array(dec_out, dtype=np.int32),
    )


if __name__ == "__main__":  # quick stats
    pairs = load_pairs()
    train, val, test = split_pairs(pairs)
    vocab = Vocab.build(t[0] + t[1] for t in train)
    print(f"pairs={len(pairs):,}  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"vocab size = {vocab.size:,}")
    enc, di, do = vectorize(train[:3], vocab)
    print("encoder input  :", enc[0])
    print("decoder input  :", di[0])
    print("decoder target :", do[0])
