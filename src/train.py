import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import random
import numpy as np
import tensorflow as tf

wandb.init(
    project="dnn_lab5",
    config={
        # conv block 1
        "conv1_filters": 32,
        "conv1_kernel_size": 5,
        "conv1_activation": "relu",
        # conv block 2
        "conv2_filters": 64,
        "conv2_kernel_size": 5,
        "conv2_activation": "relu",
        # dense layers
        "dense_1": 1000,
        "dense_2": 500,
        "dense_3": 250,
        "dense_activation": "relu",
        "output_units": 10,
        # regularization
        "dropout": 0.5,
        # training
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "from_logits": True,
        "metric": "accuracy",
        "epoch": 10,
        "batch_size": 256
    }
)

config = wandb.config

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(filters=config.conv1_filters, kernel_size=(config.conv1_kernel_size, config.conv1_kernel_size), activation=config.conv1_activation),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=config.conv2_filters, kernel_size=(config.conv2_kernel_size, config.conv2_kernel_size), activation=config.conv2_activation),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(config.dense_1, activation=config.dense_activation),
    tf.keras.layers.Dropout(rate=config.dropout),
    tf.keras.layers.Dense(config.dense_2, activation=config.dense_activation),
    tf.keras.layers.Dropout(rate=config.dropout),
    tf.keras.layers.Dense(config.dense_3, activation=config.dense_activation),
    tf.keras.layers.Dropout(rate=config.dropout),
    tf.keras.layers.Dense(config.output_units)
])

model.compile(optimizer=config.optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=config.from_logits),
              metrics=[config.metric])

history = model.fit(x=x_train, y=y_train,
                    epochs=config.epoch,
                    batch_size=config.batch_size,
                    validation_split=0.2,
                    callbacks=[
                      WandbMetricsLogger(log_freq=5),
                      WandbModelCheckpoint("models/model.keras")
                    ])

wandb.finish()