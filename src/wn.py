import numpy as np
import tensorflow as tf
import pathlib

VOCAB = "GPAVLIMCFYWHKRQNEDSTBXZ"
VOCAB_SIZE = len(VOCAB)
SEQ_SIZE = 72

AA_MAPPER = {aa: i for i, aa in enumerate(VOCAB, 1)}


class Wavenet:
    def preprocess_sequences(self, sequences):
        truncated = sequences.str.slice(0, SEQ_SIZE)

        features = np.zeros((len(sequences), SEQ_SIZE, 1))

        for i, seq in enumerate(truncated):
            for j, aa in enumerate(seq):
                features[i, j, 0] = AA_MAPPER[aa]

        return features

    def predict(self, df_test):
        X = self.preprocess_sequences(df_test["sequence"])
        return tf.keras.models.load_model(
            (pathlib.Path(__file__).parent / "wn2.h5").resolve().as_posix()
        ).predict(X).flatten()
