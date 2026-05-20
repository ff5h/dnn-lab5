import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(max_words: int, max_len: int):
    df = pd.read_csv("data/spam.csv", delimiter=",", encoding="latin-1")
    df.drop(df.columns[[2, 3, 4]], axis=1, inplace=True)
    df.columns = ["label", "text"]

    le = LabelEncoder()
    y = le.fit_transform(df["label"]).reshape(-1, 1)
    X = df["text"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)

    X_train = tf.keras.preprocessing.sequence.pad_sequences(
        tok.texts_to_sequences(X_train), maxlen=max_len
    )
    X_test = tf.keras.preprocessing.sequence.pad_sequences(
        tok.texts_to_sequences(X_test), maxlen=max_len
    )

    return (X_train, y_train), (X_test, y_test)