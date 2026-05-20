import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import tensorflow as tf
import argparse
import yaml
import os
import numpy as np

np.random.seed(7)
tf.random.set_seed(7)

from data  import load_data
from model import RNN

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/usecase-1/baseline.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg_file = yaml.safe_load(f)

config_name = os.path.splitext(os.path.basename(args.config))[0]
os.makedirs("models/usecase-1/checkpoints", exist_ok=True)
os.makedirs("logs/usecase-1", exist_ok=True)

wandb.init(
    project=cfg_file["project"],
    config=cfg_file["config"],
    job_type="train",
    name=config_name,
)
config = wandb.config

(X_train, y_train), (X_test, y_test) = load_data(
    max_words=config.max_words,
    max_len=config.max_len,
)

model = RNN(
    max_len=config.max_len,
    max_words=config.max_words,
    embedding_dim=config.embedding_dim,
    lstm_units=config.lstm_units,
    dense_units=config.dense_units,
    dropout_rate=config.dropout_rate,
)
model.compile(
    loss=config.loss,
    optimizer=config.optimizer,
    metrics=["accuracy"],
)
model.network.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=3,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir="logs/usecase-1",
        histogram_freq=1,
        write_graph=True,
        write_images=True,
    ),
    WandbMetricsLogger(log_freq="epoch"),
    WandbModelCheckpoint(
        filepath=f"models/usecase-1/checkpoints/{config_name}_{{epoch:02d}}.keras",
        monitor="val_loss",
        save_best_only=False,
    ),
]

model.fit(
    X_train,
    y_train,
    batch_size=config.batch_size,
    epochs=config.epochs,
    validation_split=0.2,
    callbacks=callbacks,
)

model.save(f"models/usecase-1/{config_name}.keras")
wandb.finish()