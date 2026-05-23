# Social Media Customer Support — Reply Assistant

An AI system that learns from real customer–support Twitter conversations and
generates contextual support replies using a **sequential deep-learning model
(RNN/GRU encoder–decoder with attention)**.

> **Constraint honoured:** No LLM APIs and no pretrained generative chat models
> are used. The model is a recurrent seq2seq network trained from scratch on the
> [Twitter Customer Support (TWCS)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
> dataset that ships with this project (`sample_data/twcs/twcs.csv`).

---

## 1. What it does

```
Customer:  my internet has been down all day, please help
Support :  sorry to hear that ! please dm us your account details so we can look into this
```

The pipeline:

1. **Reads & preprocesses** the raw TWCS tweets.
2. **Reconstructs conversations** — links every support reply to the customer
   tweet it answers using the `in_response_to_tweet_id` thread pointers.
3. **Cleans noisy social-media text** — @mentions, t.co links, HTML entities,
   emoji, agent sign-offs (`^RR`, `/AY`), elongated words (`soooo`).
4. **Trains a sequential conversational model** — a GRU encoder–decoder with
   Bahdanau attention, using teacher forcing.
5. **Generates replies** for new customer messages (greedy or sampled decoding).
6. **Serves** the model through an interactive CLI chatbot and a REST API with
   a browser chat UI.

---

## 2. Architecture

```
 customer message
        │
        ▼
 ┌──────────────┐     cleaned + tokenised + padded ids
 │ Preprocessing│ ───────────────────────────────────────┐
 └──────────────┘                                         ▼
                                              ┌────────────────────┐
                                              │ Encoder            │
                                              │  Embedding → GRU   │
                                              └─────────┬──────────┘
                                            encoder outputs + state
                                                        │
                              ┌─────────────────────────▼───────────┐
                              │ Decoder (step by step)               │
                              │  Bahdanau attention over encoder     │
                              │  outputs → GRU → Dense → vocab logits │
                              └─────────────────────────┬───────────┘
                                                        ▼
                                                support reply
```

* **Encoder** — `Embedding → GRU`. Reads the customer query; inputs are
  pre-padded so the final GRU state lands on a real token.
* **Bahdanau (additive) attention** — at every decoding step the decoder
  re-weights the whole encoder output sequence, giving it **context-awareness**
  instead of relying on a single fixed vector.
* **Decoder** — `Embedding + context → GRU → Dense`. Emits the reply one token
  at a time; trained with teacher forcing, masked cross-entropy loss.

GRU (a gated RNN) is used as the recurrent cell — it trains faster than LSTM
with comparable quality. The architecture is cell-agnostic.

---

## 3. Project structure

```
senior_ai-ml_practical/
├── README.md                 # this file
├── EVALUATION_REPORT.md       # metrics, analysis, sample outputs
├── requirements.txt
├── sample_data/
│   ├── sample.csv             # 100-row preview
│   └── twcs/twcs.csv          # full TWCS dataset (~3M tweets)
├── src/
│   ├── config.py              # all hyper-parameters & paths
│   ├── text_cleaning.py       # noisy social-media text cleaning + tokeniser
│   ├── preprocess.py          # raw CSV → cleaned (query, reply) pairs
│   ├── dataset.py             # vocabulary, encoding, train/val/test split
│   ├── model.py               # GRU encoder–decoder + Bahdanau attention
│   ├── train.py               # teacher-forcing training loop
│   ├── inference.py           # load model + generate replies
│   ├── evaluate.py            # perplexity / BLEU / diversity metrics
│   ├── chat.py                # interactive CLI chatbot
│   └── api.py                 # Flask REST API + browser chat UI
└── artifacts/                 # created by training (model, vocab, metrics)
```

---

## 4. Setup

Requires **Python 3.9+**.

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

The full TWCS dataset (`sample_data/twcs/twcs.csv`, ~504 MB) is **not committed**
to the repo because it exceeds GitHub's 100 MB file limit. Download it from
Kaggle: <https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter>
and place `twcs.csv` at `sample_data/twcs/twcs.csv`. A 100-row preview
(`sample_data/sample.csv`) is included so the code can be smoke-tested
without the download.

---

## 5. Usage

All commands are run from the project root with the virtualenv active.

### Step 1 — Preprocess (build the training pairs)

```bash
python -m src.preprocess
# options: --max-pairs N   --input PATH   --output PATH
```

Reconstructs and cleans `(customer_query, support_reply)` pairs and writes
`artifacts/pairs.csv`.

### Step 2 — Train the sequential model

```bash
python -m src.train
# options: --epochs N   --batch-size N   --limit N   (limit = quick run)
```

Builds the vocabulary, trains the GRU seq2seq model and checkpoints the best
weights to `artifacts/` (`encoder.weights.h5`, `decoder.weights.h5`,
`vocab.json`, `model_config.json`, `history.json`).

### Step 3 — Evaluate

```bash
python -m src.evaluate
# options: --limit N   (number of test pairs to score)
```

Reports perplexity, BLEU-4 and diversity, and writes
`artifacts/eval_metrics.json`. See **EVALUATION_REPORT.md** for analysis.

### Step 4a — Chat in the terminal

```bash
python -m src.chat
```

```
Customer> my order still hasn't arrived
Support > sorry to hear that ! please dm us your order number so we can help
```

Commands inside the chat: `/greedy`, `/sample`, `/reset`, `/help`, `/quit`.

### Step 4b — Run the REST API + browser chat UI

```bash
python -m src.api
# then open http://localhost:8000  in a browser
```

---

## 6. API reference

| Method | Path      | Body / Notes |
|--------|-----------|--------------|
| `GET`  | `/`       | Browser chat interface |
| `GET`  | `/health` | `{"status": "...", "model_loaded": true}` |
| `POST` | `/reply`  | `{"message": "...", "decoding": "greedy"\|"sample"}` |

Example:

```bash
curl -s -X POST http://localhost:8000/reply \
  -H 'Content-Type: application/json' \
  -d '{"message": "my internet is not working", "decoding": "greedy"}'
# {"reply": "sorry to hear that ! please dm us ...", "decoding": "greedy"}
```

---

## 7. Configuration

Every tunable parameter lives in [`src/config.py`](src/config.py): dataset size
(`MAX_PAIRS`), sequence lengths, `VOCAB_SIZE`, model dims (`EMBEDDING_DIM`,
`UNITS`), training settings (`EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`) and
decoding defaults (`TEMPERATURE`, `TOP_K`).

---

## 8. Handling noisy social-media text

`src/text_cleaning.py` normalises raw tweets before modelling:

| Noise                         | Handling                              |
|-------------------------------|---------------------------------------|
| `@AppleSupport`, `@115712`    | → `<user>` placeholder                |
| `https://t.co/...`            | → `<url>` placeholder                 |
| `&amp;`, `&gt;`, `&#39;`      | HTML-unescaped to `&`, `>`, `'`       |
| 😡😡😡 emoji                   | → `<emoji>` placeholder               |
| `^RR`, `/AY`, `-KC` sign-offs | stripped from support replies         |
| `soooo`, `!!!!`               | elongation collapsed (`soo`, `!!`)    |
| phone numbers / versions      | digit runs → `<num>` placeholder      |
| `#refund`                     | hashtag word kept (`refund`)          |

Rare words beyond `VOCAB_SIZE` map to `<unk>`; the decoder is blocked from ever
emitting `<unk>`, `<pad>` or `<sos>`.

---

## 9. Deliverables map

| Required deliverable        | Where |
|-----------------------------|-------|
| Working implementation      | `src/` (run via the steps above) |
| Source code                 | `src/*.py` |
| Trained sequential model    | `artifacts/encoder.weights.h5`, `decoder.weights.h5` |
| Data preprocessing pipeline | `src/preprocess.py`, `src/text_cleaning.py`, `src/dataset.py` |
| README with setup           | this file |
| Chatbot demo / API          | `src/chat.py` (CLI), `src/api.py` (REST + web UI) |
| Evaluation report           | `EVALUATION_REPORT.md`, `src/evaluate.py` |

---

## 10. Limitations & possible improvements

* Trained on a subset of TWCS for tractable CPU training — more data, more
  epochs and a bidirectional encoder would improve fluency.
* Single-turn modelling (one customer message → one reply). Attention provides
  in-message context; full multi-turn dialogue history is a natural extension.
* Greedy decoding favours safe, generic replies — sampling mode adds variety;
  beam search is a further improvement.
