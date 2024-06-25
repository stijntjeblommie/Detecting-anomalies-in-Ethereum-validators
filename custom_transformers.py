import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.sparse import issparse

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop = []

    def fit(self, X, y=None):
        self.columns_to_drop = [
            col for col in X.columns if "pubkey" in col or col == "validator"
        ]
        self.columns_to_drop.append("date")
        return self

    def transform(self, X):
        print("DropColumns transform shape:", X.shape)
        return X.drop(columns=self.columns_to_drop)


class RemoveDuplicates(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("RemoveDuplicates transform shape:", X.shape)
        return X.drop_duplicates()


class DateConversion(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if "date" in X.columns:
            X["date"] = pd.to_datetime(X["date"])
            X["day"] = X["date"].dt.date
        print("DateConversion transform shape:", X.shape)
        return X


class CalculateEarnedGas(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if "gas_price" in X.columns:
            condition_1 = X["gas_price"] <= 0
            condition_2 = (
                X["max_fee_per_gas"]
                >= X["base_fee_per_gas"] + X["max_priority_fee_per_gas"]
            )

            earned_gas_1 = (
                X["gas"] * (X["base_fee_per_gas"] + X["max_priority_fee_per_gas"])
                - X["base_fee_per_gas"] * X["gas"]
            )
            earned_gas_2 = (
                X["max_fee_per_gas"] * X["gas"] - X["base_fee_per_gas"] * X["gas"]
            )
            earned_gas_3 = X["gas"] * X["gas_price"] - X["base_fee_per_gas"] * X["gas"]

            X["gas_earned"] = np.where(
                condition_1 & condition_2,
                earned_gas_1,
                np.where(condition_1 & ~condition_2, earned_gas_2, earned_gas_3),
            )

        print("CalculateEarnedGas transform shape:", X.shape)
        return X


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["burnt"] = X["gas_used"] * X["base_fee_per_gas"]
        X["hash"] = X.groupby("block_number")["hash"].transform("nunique")
        X["first_seen_ts"] = X.groupby("block_number")["first_seen_ts"].transform(
            lambda x: x.max() - x.min()
        )

        X.rename(columns={"first_seen_ts": "time_span"}, inplace=True)
        X["transaction_frequency"] = np.where(
            X["time_span"] == 0, 0, X["n_transactions"] / X["time_span"]
        )
        X["reverted"] = X["n_transactions"] - X["hash"]
        X.drop(columns=["timestamp", "gas", "gas_price"], inplace=True)

        print("FeatureEngineering transform shape:", X.shape)
        return X


class ConditionalSampler(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_method=None):
        self.sampling_method = sampling_method

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.sampling_method is not None and y is not None and len(set(y)) > 1:
            X_res, y_res = self.sampling_method.fit_resample(X, y)
            return X_res, y_res
        return X, y

    def fit_resample(self, X, y=None):
        return self.transform(X, y)


class ConditionalVarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.no_features_selected = False

    def fit(self, X, y=None):
        try:
            self.selector.fit(X)
            self.variances_ = self.selector.variances_
            self.no_features_selected = not np.any(self.variances_ > self.threshold)
        except ValueError:
            self.no_features_selected = True
        return self

    def transform(self, X):
        if self.no_features_selected:
            # If no features meet the threshold, return the original data
            return X
        else:
            return self.selector.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        if self.no_features_selected:
            return (
                np.ones(X.shape[1], dtype=bool)
                if not indices
                else np.arange(X.shape[1])
            )
        else:
            return self.selector.get_support(indices=indices)


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation="relu")(input_layer)
    encoded = Dense(32, activation="relu")(encoded)
    decoded = Dense(64, activation="relu")(encoded)
    output_layer = Dense(input_dim, activation="sigmoid")(decoded)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer=Adam(), loss="mean_squared_error")
    return autoencoder


class AutoencoderModel(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim, epochs=50, batch_size=32):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = build_autoencoder(input_dim)

    def fit(self, X, y=None):
        self.autoencoder.fit(
            X, X, epochs=self.epochs, batch_size=self.batch_size, shuffle=True
        )
        return self

    def transform(self, X):
        return self.autoencoder.predict(X)

    def fit_predict(self, X, y=None):
        self.fit(X)
        reconstructed = self.transform(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        threshold = np.percentile(mse, 95)
        return np.where(mse > threshold, -1, 1)



class CustomDense(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):

        return self

    def transform(self, X):
        if issparse(X):
            return X.toarray()
        else:
            return X

