import wandb
import tensorflow as tf
import argparse
import yaml

from model import RNN
from data import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/usecase-1/baseline.yaml")
parser.add_argument("--model",  type=str, default="models/usecase-1/rnn_final.keras")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg_file = yaml.safe_load(f)

wandb.init(
    project=cfg_file["project"],
    config=cfg_file["config"],
    job_type="eval",
)
config = wandb.config

_, (X_test, y_test) = load_data(
    max_words=config.max_words,
    max_len=config.max_len,
)

model = tf.keras.models.load_model(args.model, custom_objects={"RNN": RNN})

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss:     {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

wandb.log({"test_loss": loss, "test_accuracy": accuracy})
wandb.finish()