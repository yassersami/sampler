from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class MixedMinMaxScaler:
    def __init__(
            self,
            features: List[str],
            targets: List[str],
            scale: List = None,
    ):
        self.features = features
        self.targets = targets

        self.scale = scale
        if scale is not None:
            assert all([v in ["lin", "log"] for v in scale]),\
                    f"Only allowed values for scale are 'lin' or 'log', but got:\n{zip(features+targets, scale)}"
            self.lin_vars = np.array([v == "lin" for v in scale])
            self.log_vars = np.array([v == "log" for v in scale])

        self.lin_on = self.log_on = False
        if self.lin_vars.any():
            self.lin_on = True
            self.scaler_lin = MinMaxScaler()
        if self.log_vars.any():
            self.log_on = True
            self.scaler_log = MinMaxScaler()

    def fit(self, X):
        if self.lin_on:
            self.scaler_lin.fit(X[:, self.lin_vars])
        if self.log_on:
            self.scaler_log.fit(np.log(X[:, self.log_vars]))

    def transform(self, X: np.ndarray):
        if X.shape[0] == 0:
            return X
        if self.lin_on:
            x_lin_norm = self.scaler_lin.transform(X[:, self.lin_vars])
        else:
            x_lin_norm = np.empty(len(X))
        if self.log_on:
            x_log_norm = self.scaler_log.transform(np.log(X[:, self.log_vars]))
        else:
            x_log_norm = np.empty(len(X))
        return self._merge(x_lin_norm, x_log_norm)

    def transform_with_nans(self, X_nan: np.ndarray) -> np.ndarray:
        """ Transform handling NaNs by filling temporarily with zeroes"""
        cols = self.features + self.targets
        df = pd.DataFrame(X_nan, columns=cols)
        nan_mask = df.isna()
        df = df.fillna(0)  # 0 is a dummy value
        df[cols] = self.transform(df.values)
        df = df.mask(nan_mask)
        return df.values
    
    def transform_features(self, X_feat: np.ndarray):
        """ Transforms a features only array filling the rest with ones """
        ones = np.ones((len(X_feat), len(self.targets)))
        X = np.concatenate((X_feat, ones), axis=1)
        # Return only the first len(features) columns
        return self.transform(X)[:, :len(self.features)]

    def transform_targets(self, X_tar: np.ndarray):
        """ Transforms a targets only array filling the rest with ones """
        ones = np.ones((len(X_tar), len(self.features)))
        X = np.concatenate((ones, X_tar), axis=1)
        # Return only the last len(targets) columns
        return self.transform(X)[:, -len(self.targets):]

    def inverse_transform(self, X: np.ndarray):
        if self.lin_on:
            x_lin = self.scaler_lin.inverse_transform(X[:, self.lin_vars])
        else:
            x_lin = np.empty(len(X))
        if self.log_on:
            x_log = self.scaler_log.inverse_transform(X[:, self.log_vars])
            x_log = np.exp(x_log)
        else:
            x_log = np.empty(len(X))
        return self._merge(x_lin, x_log)

    def _merge(self, x_lin_norm, x_log_norm):
        dim_col = len(self.lin_vars + self.log_vars)
        dim_row = len(x_log_norm) if self.log_on else len(x_lin_norm)
        x = np.ones((dim_row, dim_col))
        j_lin = j_log = 0
        for j in range(dim_col):
            if self.lin_vars[j]:
                x[:, j] = x_lin_norm[:, j_lin]
                j_lin += 1
            elif self.log_vars[j]:
                x[:, j] = x_log_norm[:, j_log]
                j_log += 1
        return x
