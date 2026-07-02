import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders_ = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns

        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le

        return self

    def transform(self, X):
        X_enc = X.copy()

        for col in self.columns:
            le = self.encoders_[col]

            values = X_enc[col].astype(str)

            if "unknown" not in le.classes_:
                le.classes_ = np.append(le.classes_, "unknown")

            values = values.apply(
                lambda x: x if x in le.classes_ else "unknown"
            )

            X_enc[col] = le.transform(values)

        return X_enc


def load_data(filepath):
    """Load raw dataset."""
    return pd.read_csv(filepath)


def preprocess_data(df, fit_encoders=False, encoders=None):
    """
    Preprocess data for training and prediction.
    """

    data = df.copy()

    # Remove customerID if present
    data.drop("customerID", axis=1, inplace=True, errors="ignore")

    # TotalCharges cleanup
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce"
        )
        data["TotalCharges"] = data["TotalCharges"].fillna(0)

    # Target handling
    if "Churn" in data.columns:
        y = data["Churn"].map({"No": 0, "Yes": 1})
        X = data.drop("Churn", axis=1)
    else:
        y = None
        X = data

    numeric_cols = [
        col for col in ["tenure", "MonthlyCharges", "TotalCharges"]
        if col in X.columns
    ]

    categorical_cols = [
        col for col in X.columns
        if col not in numeric_cols
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_cols
            ),
            (
                "cat",
                MultiColumnLabelEncoder(columns=categorical_cols),
                categorical_cols
            )
        ]
    )

    if fit_encoders:
        X_processed = preprocessor.fit_transform(X)
        encoders = preprocessor
    else:
        if encoders is None:
            raise ValueError(
                "encoders must be provided when fit_encoders=False"
            )

        X_processed = encoders.transform(X)

    feature_names = numeric_cols + categorical_cols

    X_processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names
    )

    if fit_encoders:
        return X_processed_df, y, encoders
    else:
        return X_processed_df, y