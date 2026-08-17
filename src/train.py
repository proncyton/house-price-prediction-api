import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler

)

from features import feature_engineering


# CONFIGURATION

DATA_PATH = "../data/housing.csv"
MODEL_PATH = "../models/house_price_xgb.joblib"

RANDOM_STATE = 42


# LOAD DATA

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# SEPARATE FEATURES AND TARGET

X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]


#FEATURE ENGINEERING

X = feature_engineering(X)

print("Features after engineering:", X.shape[1])


#TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)


#IDENTIFY FEATURE TYPES

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns


print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))


#PREPROCESSING

preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="median"
                )),
                ("scaler", StandardScaler())
            ]),
            numerical_features
        ),

        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ]),
            categorical_features
        )
    ]
)


#FINAL XGBOOST MODEL

xgb_model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE
)


#COMPLETE ML PIPELINE

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", xgb_model)
    ]
)


#5-FOLD CROSS-VALIDATION

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = cross_validate(
    model,
    X,
    y,
    cv=kf,
    scoring={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    },
    n_jobs=-1
)


cv_mae = -cv_results["test_mae"]
cv_rmse = -cv_results["test_rmse"]
cv_r2 = cv_results["test_r2"]


print("\n===== 5-FOLD CROSS-VALIDATION =====")

print("Fold MAE:")
print(cv_mae)

print("Average MAE:", cv_mae.mean())
print("MAE Std:", cv_mae.std())

print("\nFold RMSE:")
print(cv_rmse)

print("Average RMSE:", cv_rmse.mean())
print("RMSE Std:", cv_rmse.std())

print("\nFold R²:")
print(cv_r2)

print("Average R²:", cv_r2.mean())
print("R² Std:", cv_r2.std())


#TRAIN FINAL MODEL

model.fit(
    X_train,
    y_train
)


#TEST SET PREDICTIONS

predictions = model.predict(X_test)

#FINAL EVALUATION

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n===== FINAL TEST PERFORMANCE =====")

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)


#TRAINING PERFORMANCE

train_predictions = model.predict(X_train)

train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

train_r2 = r2_score(
    y_train,
    train_predictions
)


print("\n===== TRAIN PERFORMANCE =====")

print("Train MAE:", train_mae)
print("Train R²:", train_r2)


#SAVE MODEL
os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved to:", MODEL_PATH)