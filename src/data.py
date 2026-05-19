import tensorflow as tf

def load_data(max_words: int, max_len: int):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_words)
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
    return (X_train, y_train), (X_test, y_test)