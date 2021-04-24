import numpy as np
import pathlib
from collections import Counter
import pandas as pd
import xgboost

VOCAB = "GPAVLIMCFYWHKRQNEDSTBXZ"
VOCAB_SIZE = len(VOCAB)
SEQ_SIZE = 72

AA_MAPPER = {aa: i for i, aa in enumerate(VOCAB, 1)}


class Xgboost:
    def preprocess_sequences(self, sequences):
        VOCAB = "GPAVLIMCFYWHKRQNEDSTBXZ"
        VOCAB_SIZE = len(VOCAB)
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
        print(X)
        xgb = xgboost.XGBRegressor()
        xgb.load_model(
            (pathlib.Path(__file__).parent /
                "xgboost.bin").resolve().as_posix())
        return xgb.predict(X)
