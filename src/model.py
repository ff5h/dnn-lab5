import tensorflow as tf

class RNN(tf.keras.Model):
    def __init__(
        self,
        max_len: int,
        max_words: int,
        embedding_dim: int,
        lstm_units: int,
        filters: int,
        kernel_size: int,
        **kwargs,
    ):
        super(RNN, self).__init__(**kwargs)

        self.max_len = max_len
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.filters       = filters
        self.kernel_size   = kernel_size

        self.network = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(max_len,), name="inputs"),
            tf.keras.layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len, name="embedding"),
            tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', name='conv1d'),
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
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)