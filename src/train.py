import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import tensorflow as tf

import os
os.makedirs("models/usecase-2: LSTM/checkpoints", exist_ok=True)

max_words = 5000
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_words)

max_len = 500
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

wandb.init(
    project="dnn_lab5",
    config={
        "max_len": max_len,
        "max_words": max_words,
        "embedding_dim": 32,
        "lstm_units": 100,
        "batch_size": 64,
        "epochs": 5,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
    }
)

config = wandb.config

class RNN(tf.keras.Model):
    def __init__(
        self,
        max_len: int = config.max_len,
        max_words: int = config.max_words,
        embedding_dim: int = config.embedding_dim,
        lstm_units: int = config.lstm_units,
        **kwargs,
    ):
        super(RNN, self).__init__(**kwargs)

        self.max_len = max_len
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units

        self.network = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_len,), name="inputs"),
            tf.keras.layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len, name="embedding"),
            tf.keras.layers.LSTM(units=lstm_units, name="lstm"),
            tf.keras.layers.Dense(units=1, name="output"),
            tf.keras.layers.Activation("sigmoid", name="sigmoid"),
        ], name="rnn_sequential")

    def call(self, x):
        return self.network(x)

    def get_config(self):
        config = super(RNN, self).get_config()
        config.update({
            "max_len": self.max_len,
            "max_words": self.max_words,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

model = RNN()
model.compile(
    loss=config.loss,
    optimizer=config.optimizer,
    metrics=["accuracy"],
)

print(model.network.summary())

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=5,
        restore_best_weights=True,
    ),
    WandbMetricsLogger(log_freq="epoch"),
    WandbModelCheckpoint(
        filepath="models/usecase-2: LSTM/checkpoints/rnn_{epoch:02d}.keras",
        monitor="val_loss"
    ),
]

history = model.fit(
    X_train,
    y_train,
    batch_size=config.batch_size,
    epochs=config.epochs,
    validation_split=0.2,
    callbacks=callbacks,
)

scores = model.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))

wandb.finish()