import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import random
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
os.makedirs("models/usecase-1/checkpoints", exist_ok=True)

df = pd.read_csv('data/spam.csv', delimiter=',', encoding='latin-1')
df.drop(df.columns[[2, 3, 4]], axis=1, inplace=True)

X = df.v2
Y = df.v1
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

max_words = 1000
max_len = 150
tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

wandb.init(
    project="dnn_lab5",
    config={
        "max_len": max_len,
        "max_words": max_words,
        "embedding_dim": 50,
        "lstm_units": 64,
        "dense_units": 256,
        "dropout_rate": 0.5,
        "batch_size": 128,
        "epochs": 10,
        "optimizer": "rmsprop",
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
        dense_units: int = config.dense_units,
        dropout_rate: float = config.dropout_rate,
        **kwargs,
    ):
        super(RNN, self).__init__(**kwargs)

        self.max_len = max_len
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        self.network = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_len,), name="inputs"),
            tf.keras.layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len, name="embedding"),
            tf.keras.layers.LSTM(units=lstm_units, name="lstm"),
            tf.keras.layers.Dense(units=dense_units, name="fc1"),
            tf.keras.layers.Activation("relu", name="relu"),
            tf.keras.layers.Dropout(rate=dropout_rate, name="dropout"),
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
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = RNN()
model.compile(
    loss=config.loss,
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=3,
        restore_best_weights=True,
    ),
    WandbMetricsLogger(log_freq="epoch"),
    WandbModelCheckpoint(
        filepath="models/usecase-1/checkpoints/rnn_{epoch:02d}.keras",
        monitor="val_loss"
    ),
]

history = model.fit(
    sequences_matrix,
    Y_train,
    batch_size=config.batch_size,
    epochs=config.epochs,
    validation_split=0.2,
    callbacks=callbacks,
)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = tf.keras.preprocessing.sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,Y_test)

wandb.finish()