import wandb
import tensorflow as tf
import argparse
import yaml
import os

from model import RNN
from data import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/usecase-2/lstm+cnn+dropout/baseline.yaml")
parser.add_argument("--model",  type=str, default=None)
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg_file = yaml.safe_load(f)

usecase = cfg_file["usecase"]
arch = cfg_file["arch"]
config_name = os.path.splitext(os.path.basename(args.config))[0]
run_name = f"eval/{usecase}/{arch}/{config_name}"

if args.model is None:
    args.model = f"models/{usecase}/{arch}/{config_name}.keras"

wandb.init(
    project=cfg_file["project"],
    config=cfg_file["config"],
    job_type="eval",
    name=run_name
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