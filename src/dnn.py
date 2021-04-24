from collections import Counter

import numpy as np
import pandas as pd
from keras.models import load_model

VOCAB = "GPAVLIMCFYWHKRQNEDSTBXZ"
VOCAB_SIZE = len(VOCAB)


class DeepNeuralNetwork1:
    def preprocess_sequences(self, sequences):
        lens = sequences.apply(len)

        features = np.zeros((len(sequences), VOCAB_SIZE))

        for i, seq in enumerate(sequences):
            counts = dict.fromkeys(VOCAB, 0)
            counts.update(Counter(seq))

            features[i, :] = [*counts.values()]

        x_data = pd.DataFrame(features, columns=list(VOCAB))
        x_data = x_data.div(lens, axis=0)
        return x_data

    def predict(self, df_test):
        X = self.preprocess_sequences(df_test["sequence"])
        return load_model("dnn.h5").predict(X).flatten()
