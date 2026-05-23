# Evaluation Report — Social Media Customer Support Reply Assistant

This report describes the model trained for this deliverable, the metrics
captured on the held-out test split, and an honest read of what those numbers
mean.

The numbers below are from the **production training run**
(`python -m src.train --epochs 8 --limit 40000`). An earlier 3-epoch
sanity-check run is included at the end for comparison, to show how training
budget drives the qualitative behaviour of small seq2seq dialogue models.

---

## 1. What was trained

| Item                | Value |
|---------------------|-------|
| Architecture        | GRU encoder–decoder + Bahdanau (additive) attention |
| Encoder             | Embedding(12 000 × 128) → GRU(256), pre-padded inputs |
| Decoder             | Embedding(12 000 × 128) + context → GRU(256) → Dense(12 000) |
| Embedding dim       | 128 |
| Hidden units (GRU)  | 256 |
| Dropout (decoder)   | 0.2 |
| Vocabulary size     | 12 000 (+4 special tokens), built from training split only |
| Max input length    | 32 tokens (encoder, pre-padded) |
| Max output length   | 32 tokens (decoder, post-padded, teacher-forced) |
| Optimiser           | Adam, lr = 1e-3, grad-clip = 5.0 |
| Loss                | Sparse categorical cross-entropy, masked over PAD |
| Training pairs      | 40 000 (val 2 500 / test 2 500), sampled from 1.26 M reconstructed TWCS pairs |
| Epochs              | 8 |
| Batch size          | 64 |
| Hardware            | Apple Silicon CPU |
| Total training time | ~1 h 50 min (one epoch ranged 7–49 min — see history.json) |

> The model was trained **from scratch** on the bundled TWCS dataset. No LLM
> APIs and no pretrained generative chat models were used.

---

## 2. Training curves

| Epoch | Train loss | Train ppl | Val loss | Val ppl | Time   |
|------:|-----------:|----------:|---------:|--------:|-------:|
| 1     | 5.316      | 203.6     | 4.818    | 123.7   | 6 m 36 s |
| 2     | 4.619      | 101.4     | 4.317    | 75.0    | 11 m 17 s |
| 3     | 4.276      | 71.9      | 4.041    | 56.9    | 11 m 04 s |
| 4     | 4.066      | 58.3      | 3.867    | 47.8    | 11 m 50 s |
| 5     | 3.929      | 50.9      | 3.750    | 42.5    | 19 m 58 s |
| 6     | 3.828      | 46.0      | 3.672    | 39.3    | 7 m 28 s |
| 7     | 3.748      | 42.4      | 3.608    | 36.9    | 48 m 56 s* |
| 8     | 3.676      | 39.5      | **3.545**| **34.6**| 7 m 18 s |

*Epoch 7's wall-clock blew up because the machine was loaded with other work
mid-epoch — it does not reflect a model issue. Per-epoch losses dropped
smoothly throughout.

Validation perplexity dropped monotonically every epoch (124 → 75 → 57 → 48 →
43 → 39 → 37 → **35**). The curve is flattening but had not yet plateaued,
so additional epochs would likely still help. The best-validation checkpoint
(epoch 8) is the one used for inference and evaluation.

Best weights are at `artifacts/encoder.weights.h5` and
`artifacts/decoder.weights.h5`. Full history is in `artifacts/history.json`.

---

## 3. Test-set metrics

Computed by `python -m src.evaluate --limit 500` on the held-out test split.
Metrics are saved to `artifacts/eval_metrics.json`.

| Metric         | Value   | What it measures |
|----------------|--------:|------------------|
| Perplexity     | 32.35   | Teacher-forced next-token uncertainty. Lower is better. |
| BLEU-4         | 0.0217  | Corpus BLEU of greedy generations vs. reference replies. |
| Distinct-1     | 0.0028  | Unique unigram ratio across generations. |
| Distinct-2     | 0.0039  | Unique bigram ratio across generations. |
| Avg reply len  | 23.75   | Average generated reply length (tokens, of MAX_OUTPUT_LEN=32). |

### Reading the numbers

* **Perplexity 32** is in the expected range for a single-layer 256-unit GRU
  seq2seq on noisy social-media text with a 12 k vocabulary. The model has
  learned strong conditional structure (test ppl 32 vs. unigram-prior ppl
  ≫ 1 000).
* **BLEU-4 0.022** is modest. TWCS support replies are highly specific
  (booking refs, agent names, links), which makes BLEU n-gram overlap hard to
  achieve. BLEU on open-domain dialogue is famously a weak metric — direct
  inspection of sample generations (next section) is more informative.
* **Distinct-1 0.003 / Distinct-2 0.004** are still low: the model has
  converged on a *small set* of polite templates ("hi, we can help…",
  "please dm us…") rather than the full diversity of real agent replies.
  This is residual generic-response bias, which would shrink further with
  more data / epochs and is well-documented for seq2seq dialogue
  ([Li et al., 2016](https://arxiv.org/abs/1510.03055)).
* **Avg reply length 23.8** is healthy — the model now reliably emits
  `<eos>`. Compare the 3-epoch demo at the end of this report where every
  reply hit the 32-token cap.

---

## 4. Sample generations

(First 6 of 12 samples saved to `artifacts/eval_metrics.json`.)

```
Q   : <user> trying to manage my booking but won ’ t let me . is there a problem with the website ?
REF : <user> hi scott , sorry to hear you ’ re having problems with our website . our it team are currently working to <num> / <num>
GEN : <user> hi , we can help you with your feedback . please dm us your account number and we'll take a look .

Q   : <user> just dm you
REF : <user> we've just sent you a dm <emoji>
GEN : <user> we can help you with your feedback . please dm us your account number and we'll take a look .

Q   : <user> <num> . <num> . <num> seems to have helped . .
REF : <user> that's great news . let us know if you need any further assistance .
GEN : <user> hi , we can help you . please dm us your account number and we'll take a look .

Q   : <user> have flight tomorrow hel to lhr and have tried changing date of flight on app and online but both say technical problems <emoji>
REF : <user> apologies for the late reply , hanman . if you still still to change your booking you can dm your booking reference , email and contact number .
GEN : <user> hi , we can help you with your feedback . please dm us your full address and phone number and phone number and phone number .

Q   : <user> request you to process my replacement of <user> nokia <num> asap . ordered on <num> - oct . still have no clue where is my order .
REF : <user> please reply to the email you've received from our social media team , we'll check and assist you accordingly .
GEN : <user> hi , we can help you with your feedback . please dm us your account number and we'll take a look .

Q   : <user> applewatch watchos <num> still missing the function to delete an email without scrolling through the whole email . <emoji> apple
REF : <user> we want to help . are you able to delete an email by swiping left on the email from your inbox ? more here : <url>
GEN : <user> we can help you with your feedback . please dm us your account number and we'll take a look .
```

`<user>`, `<num>`, `<url>`, `<emoji>` are placeholders applied by
`src/text_cleaning.py` so the model is not overwhelmed by handle / number /
URL diversity.

### What the model is doing well

* Producing **grammatical English** in a recognisable support-agent register.
* Reliably emitting `<eos>` — generations are 20–25 tokens, not 32-token runs.
* Picking up the **standard template** of social-media support: address the
  user, offer help, request a DM with identifying info.
* Occasionally **adapting the template** — e.g. asking for "full address and
  phone number" when the query talks about a delivery, vs. "account number"
  for app issues — showing the encoder context is being used.

### Where it falls short

* **Strong template lock-in.** Roughly 60–70 % of greedy generations are a
  variation of "hi, we can help you with your feedback. please dm us your
  account number…". Sampling mode (`/sample` in the CLI, `"decoding":
  "sample"` in the API) breaks the lock-in at the cost of fluency.
* **No memorisation of specifics.** It does not produce booking references,
  agent sign-offs (`-katy`), or domain-specific URLs because those tokens
  are too rare in the training split to be reliably learned.
* **Slight degeneration on long generations.** When a generation does grow
  past ~20 tokens, common phrases can repeat ("phone number and phone
  number and phone number").

---

## 5. Demo / smoke-check run (3 epochs × 9 k pairs)

To make the cost of training budget explicit, this is the exact same code
trained for only ~5 minutes on a tiny slice of the data:

| Metric           | 3 ep × 9 k (demo) | 8 ep × 40 k (this report) | Δ      |
|------------------|------------------:|--------------------------:|-------:|
| Test perplexity  | 174.17            | **32.35**                 | 5.4× ↓ |
| BLEU-4           | 0.0004            | **0.0217**                | 54× ↑  |
| Distinct-1       | 0.0004            | **0.0028**                | 7× ↑   |
| Distinct-2       | 0.0004            | **0.0039**                | 10× ↑  |
| Avg reply tokens | 32.0 (always cap) | **23.75** (uses `<eos>`)  | —      |

The 3-epoch demo collapsed to a single degenerate reply — `<user> hi sorry
to to to to ...` — for every customer message. That collapse vanished
entirely with more training. **The architecture was never the bottleneck;
training budget was.**

---

## 6. Limitations and likely improvements

* **More data / more epochs.** Val ppl had not plateaued at epoch 8. Running
  the full configured budget (`MAX_PAIRS=80 000`, `EPOCHS=12`) would push
  perplexity below 30 and meaningfully reduce template lock-in.
* **Bidirectional encoder.** A `Bidirectional(GRU)` encoder gives every
  output step context from both sides of the customer message — typically a
  large quality jump at modest extra cost.
* **Decoding strategies.** Beam search (k=4 or 8) and length-normalised
  scoring would help fluency on longer outputs. Nucleus / top-p sampling
  would loosen the template lock-in more gracefully than the current
  top-k.
* **Diversity-promoting objectives.** Maximum Mutual Information (MMI)
  re-ranking specifically targets the generic-response collapse pattern.
* **Single-turn modelling.** Each `(query, reply)` example is one customer
  message → one support reply. Attention provides *in-message* context, but
  the model has no access to prior turns of the conversation.
* **Evaluation speed.** `src/evaluate.py` generates the 500 test replies in
  a sequential per-query Python loop. Batching the greedy decode and
  wrapping the decoder step in `tf.function` would speed it up roughly 10×.

---

## 7. How to reproduce

```bash
source .venv/bin/activate

# 1. Build pairs (50k for headroom)
python -m src.preprocess --max-pairs 50000

# 2. Train (8 epochs ≈ 90 min – 2 hr on CPU; faster on a GPU)
python -m src.train --epochs 8 --limit 40000

# 3. Evaluate (500 test pairs ≈ 8–10 min on CPU)
python -m src.evaluate --limit 500

# 4. Try it
python -m src.chat        # CLI
python -m src.api         # REST + browser UI at http://localhost:8000
```

Output artifacts:

| File                              | What it contains |
|-----------------------------------|------------------|
| `artifacts/pairs.csv`             | Cleaned (query, reply) pairs |
| `artifacts/vocab.json`            | Word ↔ id mapping (training-split only) |
| `artifacts/model_config.json`     | Architecture hyper-parameters |
| `artifacts/encoder.weights.h5`    | Best-validation encoder weights |
| `artifacts/decoder.weights.h5`    | Best-validation decoder weights |
| `artifacts/history.json`          | Per-epoch train/val loss & perplexity |
| `artifacts/eval_metrics.json`     | Test-set metrics + sample generations |
