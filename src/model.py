"""Sequential seq2seq model: GRU encoder-decoder with Bahdanau attention.

This is a classic recurrent (RNN-family) conversational model - no LLM APIs
and no pretrained generative models are used.

  Encoder  : Embedding -> GRU            (reads the customer query)
  Attention: Bahdanau additive attention (gives the decoder context-awareness
             by letting every output step look back at the whole query)
  Decoder  : Embedding + context -> GRU -> Dense  (emits the reply token by
             token)

GRU is used as the recurrent cell: it is a gated RNN that trains faster than
LSTM with comparable quality.  The architecture is cell-agnostic - swapping
`tf.keras.layers.GRU` for `LSTM` only requires carrying the extra cell state.
"""
import tensorflow as tf

from src import config


class Encoder(tf.keras.Model):
    """Embeds the customer query and runs it through a GRU."""

    def __init__(self, vocab_size, embedding_dim=config.EMBEDDING_DIM,
                 units=config.UNITS):
        super().__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x, training=False):
        # x: (batch, MAX_INPUT_LEN) -> encoder inputs are PRE-padded, so the
        # final GRU state corresponds to a real query token.
        x = self.embedding(x)
        outputs, state = self.gru(x, training=training)
        return outputs, state          # (batch, T, units), (batch, units)


class BahdanauAttention(tf.keras.layers.Layer):
    """Additive (Bahdanau) attention scoring decoder state vs. encoder outputs."""

    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)   # over encoder outputs
        self.W2 = tf.keras.layers.Dense(units)   # over decoder state (query)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, value_mask=None):
        # query : (batch, units)        - current decoder hidden state
        # values: (batch, T, units)     - all encoder outputs
        query_t = tf.expand_dims(query, 1)                       # (batch,1,units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_t)))  # (batch,T,1)
        if value_mask is not None:
            # push padded positions to -inf before softmax
            mask = tf.cast(tf.expand_dims(value_mask, -1), score.dtype)
            score += (1.0 - mask) * -1e9
        weights = tf.nn.softmax(score, axis=1)                   # (batch,T,1)
        context = tf.reduce_sum(weights * values, axis=1)        # (batch,units)
        return context, weights


class Decoder(tf.keras.Model):
    """Single-step GRU decoder that attends over the encoder outputs."""

    def __init__(self, vocab_size, embedding_dim=config.EMBEDDING_DIM,
                 units=config.UNITS, dropout=config.DROPOUT):
        super().__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(units)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, token, hidden, enc_outputs, enc_mask=None, training=False):
        # token: (batch, 1) - the previously emitted / teacher-forced token
        context, weights = self.attention(hidden, enc_outputs, enc_mask)
        x = self.embedding(token)                                # (batch,1,emb)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)  # (batch,1,emb+units)
        output, state = self.gru(x, initial_state=hidden, training=training)
        output = tf.reshape(output, (-1, self.units))            # (batch,units)
        output = self.dropout(output, training=training)
        logits = self.fc(output)                                 # (batch,vocab)
        return logits, state, weights


def build_models(vocab_size):
    """Construct a fresh encoder/decoder pair."""
    encoder = Encoder(vocab_size)
    decoder = Decoder(vocab_size)
    return encoder, decoder
