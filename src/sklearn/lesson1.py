from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import pandas as pd


df = pd.read_csv("../../data/housing.csv")

# Features and target
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Identify column types
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["str"]
).columns

# Numerical preprocessing
numeric_transformer = SimpleImputer(strategy="median")

# Categorical preprocessing
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Full ML pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

#print(len(numerical_features))
#print(len(categorical_features))

#print(categorical_features)

regressor = model.named_steps["regressor"]

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

#print("Intercept:", regressor.intercept_)
#print("Number of coefficients:", len(regressor.coef_))

#for feature, coefficient in zip(feature_names[:20], regressor.coef_[:20]):
#    print(feature, coefficient)

#for feature, coefficient in zip(feature_names, regressor.coef_):
#    if feature.startswith("cat__Neighborhood"):
#        print(feature, coefficient)



tree_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor(
                max_depth=5,
                random_state=42
            ))
        ]
    )

tree_model.fit(X_train, y_train)

train_predictions = tree_model.predict(X_train)
test_predictions = tree_model.predict(X_test)

train_mae = mean_absolute_error(
        y_train,
        train_predictions
    )

test_mae = mean_absolute_error(
        y_test,
        test_predictions
    )

train_r2 = r2_score(
        y_train,
        train_predictions
    )

test_r2 = r2_score(
        y_test,
        test_predictions
    )

print("Train MAE:", train_mae)
print("Test MAE:", test_mae)
print("Train R²:", train_r2)
print("Test R²:", test_r2)

from sklearn.model_selection import cross_val_score


scores = cross_val_score(
    tree_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error"
)

mae_scores = -scores

print("Fold MAE:", mae_scores)
print("Average MAE:", mae_scores.mean())

r2_scores = cross_val_score(
    tree_model,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

print("Fold R²:", r2_scores)
print("Average R²:", r2_scores.mean())

from sklearn.model_selection import GridSearchCV

param_grid = {
    "regressor__max_depth": [2, 5, 10, None],
    "regressor__min_samples_split": [2, 5, 10],
    "regressor__min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    tree_model,
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)


print("Best parameters:")
print(grid_search.best_params_)

print("\nBest CV MAE:")
print(-grid_search.best_score_)


best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\n===== TUNED DECISION TREE =====")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)


from sklearn.ensemble import RandomForestRegressor

random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

random_forest.fit(
    X_train,
    y_train
)

rf_predictions = random_forest.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = mean_squared_error(
    y_test,
    rf_predictions
) ** 0.5

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\n===== RANDOM FOREST =====")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R²:", rf_r2)

rf_train_predictions = random_forest.predict(X_train)

rf_train_mae = mean_absolute_error(
    y_train,
    rf_train_predictions
)

rf_train_r2 = r2_score(
    y_train,
    rf_train_predictions
)

print("RF Train MAE:", rf_train_mae)
print("RF Test MAE:", rf_mae)

print("RF Train R²:", rf_train_r2)
print("RF Test R²:", rf_r2)



xgb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_predictions = xgb_model.predict(
    X_test
)

xgb_mae = mean_absolute_error(
    y_test,
    xgb_predictions
)

xgb_rmse = mean_squared_error(
    y_test,
    xgb_predictions
) ** 0.5

xgb_r2 = r2_score(
    y_test,
    xgb_predictions
)

print("\n===== XGBOOST =====")
print("MAE:", xgb_mae)
print("RMSE:", xgb_rmse)
print("R²:", xgb_r2)

xgb_train_predictions = xgb_model.predict(X_train)

xgb_train_mae = mean_absolute_error(
    y_train,
    xgb_train_predictions
)

xgb_train_r2 = r2_score(
    y_train,
    xgb_train_predictions
)

print("XGB Train MAE:", xgb_train_mae)
print("XGB Test MAE:", xgb_mae)

print("XGB Train R²:", xgb_train_r2)
print("XGB Test R²:", xgb_r2)



param_grid = {
    "regressor__n_estimators": [100, 200, 300, 400],
    "regressor__learning_rate": [0.05, 0.1, 0.2, 0.3],
    "regressor__max_depth": [3, 5, 7, 9]
}

xgb_grid = GridSearchCV(
    xgb_model,
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

xgb_grid.fit(
    X_train,
    y_train
)

print("Best parameters:")
print(xgb_grid.best_params_)

print("\nBest CV MAE:")
print(-xgb_grid.best_score_)

best_xgb = xgb_grid.best_estimator_

xgb_predictions = best_xgb.predict(
    X_test
)

xgb_mae = mean_absolute_error(
    y_test,
    xgb_predictions
)

xgb_rmse = mean_squared_error(
    y_test,
    xgb_predictions
) ** 0.5

xgb_r2 = r2_score(
    y_test,
    xgb_predictions
)

print("\n===== TUNED XGBOOST =====")
print("MAE:", xgb_mae)
print("RMSE:", xgb_rmse)
print("R²:", xgb_r2)

import shap

# =========================
# SHAP EXPLAINABILITY
# =========================

# Get the fitted preprocessor from the best XGBoost pipeline
preprocessor_fitted = best_xgb.named_steps["preprocessor"]

# Transform the test data using the same preprocessing
X_test_transformed = preprocessor_fitted.transform(X_test)

# Get the actual XGBoost model from the pipeline
xgb_regressor = best_xgb.named_steps["regressor"]

# Create SHAP explainer for XGBoost
explainer = shap.TreeExplainer(xgb_regressor)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_transformed)

# Get names of all transformed features
feature_names = preprocessor_fitted.get_feature_names_out()

print("Number of features:", len(feature_names))
print("SHAP shape:", shap_values.shape)


# =========================
# COLORED SHAP SUMMARY PLOT
# =========================

# Convert sparse matrix to normal array if necessary
X_test_shap = X_test_transformed.toarray() if hasattr(
    X_test_transformed, "toarray"
) else X_test_transformed

# Put transformed features into a DataFrame
X_test_shap = pd.DataFrame(
    X_test_shap,
    columns=feature_names,
    index=X_test.index
)

# Global SHAP importance
shap.summary_plot(
    shap_values,
    X_test_shap,
    feature_names=feature_names
)


# =========================
# INDIVIDUAL HOUSE
# =========================

# Select one house from the test set
house_index = 0

# SHAP values for this particular house
house_shap_values = shap_values[house_index]

# Feature values for this particular house
house_data = X_test_shap.iloc[house_index]

# Display the original house information
print("\n===== ORIGINAL HOUSE =====")
print(X_test.iloc[house_index])

# Actual sale price
actual_price = y_test.iloc[house_index]

# Predicted sale price
predicted_price = best_xgb.predict(
    X_test.iloc[[house_index]]
)[0]

print("\nActual price:", actual_price)
print("Predicted price:", predicted_price)
print("Error:", actual_price - predicted_price)


# =========================
# WATERFALL PLOT
# =========================

shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    house_shap_values,
    feature_names=feature_names
)


# =========================
# RESIDUAL ANALYSIS
# =========================

predictions = best_xgb.predict(X_test)

residuals = y_test - predictions

residual_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions,
    "Residual": residuals,
    "AbsoluteError": abs(residuals)
})

print("\n===== LARGEST ERRORS =====")

print(
    residual_df
    .sort_values("AbsoluteError", ascending=False)
    .head(10)
)

import matplotlib.pyplot as plt

# =========================
# ACTUAL VS PREDICTED
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.6
)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Prices")

plt.show()

# =========================
# RESIDUAL PLOT
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(
    predictions,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Sale Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted Sale Price")

plt.show()

import numpy as np

# ==========================================
# LOG-TRANSFORMED XGBOOST
# ==========================================

# Transform the target
y_train_log = np.log1p(y_train)

# Create XGBoost model
log_xgb = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Create pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", log_xgb)
    ]
)

# Train on log-transformed SalePrice
model.fit(
    X_train,
    y_train_log
)

# Predict log(SalePrice)
predictions_log = model.predict(X_test)

# Convert predictions back to dollars
predictions = np.expm1(predictions_log)


# ==========================================
# EVALUATION
# ==========================================

mae_log = mean_absolute_error(
    y_test,
    predictions
)

rmse_log = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2_log = r2_score(
    y_test,
    predictions
)

print("\n===== LOG XGBOOST =====")
print("MAE:", mae_log)
print("RMSE:", rmse_log)
print("R²:", r2_log)


# ==========================================
# TRAINING PERFORMANCE
# ==========================================

train_predictions_log = model.predict(X_train)

train_predictions = np.expm1(
    train_predictions_log
)

train_mae_log = mean_absolute_error(
    y_train,
    train_predictions
)

train_r2_log = r2_score(
    y_train,
    train_predictions
)

print("Log XGB Train MAE:", train_mae_log)
print("Log XGB Test MAE:", mae_log)

print("Log XGB Train R²:", train_r2_log)
print("Log XGB Test R²:", r2_log)

# ==========================================
# LOG MODEL RESIDUAL ANALYSIS
# ==========================================

log_residuals = y_test - predictions

log_residual_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions,
    "Residual": log_residuals,
    "AbsoluteError": abs(log_residuals)
})

print("\n===== LOG XGBOOST LARGEST ERRORS =====")

print(
    log_residual_df
    .sort_values("AbsoluteError", ascending=False)
    .head(10)
)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.6
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Log XGBoost: Actual vs Predicted")

plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(
    predictions,
    log_residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Sale Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Log XGBoost: Residuals vs Predicted")

plt.show()

# Inspect the worst prediction
worst_index = 691

print("\n===== WORST PREDICTION =====")
print(X_test.loc[worst_index])

print("\nActual:", y_test.loc[worst_index])
print("Predicted:", predictions.loc[worst_index] if hasattr(predictions, "loc") 
      else predictions[X_test.index.get_loc(worst_index)])
raw_predictions = best_xgb.predict(X_test)

print("\nActual:", y_test.loc[worst_index])
print("Raw XGBoost prediction:", raw_predictions[X_test.index.get_loc(worst_index)])


# ============================================================
# 5-FOLD CROSS-VALIDATION - TUNED XGBOOST
# ============================================================

from sklearn.model_selection import KFold, cross_validate

# Use 5 different train/validation splits
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Cross-validation metrics
cv_results = cross_validate(
    best_xgb,
    X,
    y,
    cv=kf,
    scoring={
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    },
    return_train_score=True,
    n_jobs=-1
)

# Convert negative MAE back to positive
cv_mae = -cv_results["test_mae"]
cv_r2 = cv_results["test_r2"]

print("\n===== 5-FOLD CV - TUNED XGBOOST =====")

print("Fold MAE:", cv_mae)
print("Average MAE:", cv_mae.mean())
print("MAE Std:", cv_mae.std())

print("\nFold R²:", cv_r2)
print("Average R²:", cv_r2.mean())
print("R² Std:", cv_r2.std())

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):

    df = df.copy()

    # Total square footage
    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    # Age of the house when it was sold
    df["HouseAge"] = (
        df["YrSold"]
        - df["YearBuilt"]
    )

    # Years since the house was last remodeled
    df["YearsSinceRemod"] = (
        df["YrSold"]
        - df["YearRemodAdd"]
    )

    # Convert bathrooms into one comparable measure
    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # Total porch/deck area
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
    )

    # Total finished basement area
    df["TotalBsmtFinished"] = (
        df["BsmtFinSF1"]
        + df["BsmtFinSF2"]
    )

    return df


X_engineered = feature_engineering(X)

print("\nOriginal number of features:", X.shape[1])
print("Engineered number of features:", X_engineered.shape[1])

X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
    X_engineered,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# PREPROCESSING - FEATURE ENGINEERED DATA
# ============================================================

numerical_features_fe = X_engineered.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features_fe = X_engineered.select_dtypes(
    include=["str", "object"]
).columns

print("\nNumerical features:", len(numerical_features_fe))
print("Categorical features:", len(categorical_features_fe))


preprocessor_fe = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical_features_fe
        ),

        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ]),
            categorical_features_fe
        )
    ]
)

# ============================================================
# FEATURE-ENGINEERED XGBOOST
# ============================================================

xgb_fe = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

xgb_fe_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor_fe),
        ("regressor", xgb_fe)
    ]
)

xgb_fe_model.fit(
    X_train_fe,
    y_train_fe
)

predictions_fe = xgb_fe_model.predict(
    X_test_fe
)

# ============================================================
# FEATURE-ENGINEERED MODEL EVALUATION
# ============================================================

mae_fe = mean_absolute_error(
    y_test_fe,
    predictions_fe
)

rmse_fe = np.sqrt(
    mean_squared_error(
        y_test_fe,
        predictions_fe
    )
)

r2_fe = r2_score(
    y_test_fe,
    predictions_fe
)

print("\n===== FEATURE-ENGINEERED XGBOOST =====")

print("MAE:", mae_fe)
print("RMSE:", rmse_fe)
print("R²:", r2_fe)

# ============================================================
# 5-FOLD CV - FEATURE-ENGINEERED XGBOOST
# ============================================================

cv_results_fe = cross_validate(
    xgb_fe_model,
    X_engineered,
    y,
    cv=kf,
    scoring={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    },
    n_jobs=-1
)

cv_mae_fe = -cv_results_fe["test_mae"]
cv_rmse_fe = -cv_results_fe["test_rmse"]
cv_r2_fe = cv_results_fe["test_r2"]

print("\n===== 5-FOLD CV - FEATURE-ENGINEERED XGBOOST =====")

print("\nFold MAE:")
print(cv_mae_fe)
print("Average MAE:", cv_mae_fe.mean())
print("MAE Std:", cv_mae_fe.std())

print("\nFold RMSE:")
print(cv_rmse_fe)
print("Average RMSE:", cv_rmse_fe.mean())
print("RMSE Std:", cv_rmse_fe.std())

print("\nFold R²:")
print(cv_r2_fe)
print("Average R²:", cv_r2_fe.mean())
print("R² Std:", cv_r2_fe.std())

# ============================================================
# SHAP - FEATURE-ENGINEERED XGBOOST
# ============================================================

# Get the fitted preprocessor from the feature-engineered pipeline
preprocessor_fe_fitted = xgb_fe_model.named_steps["preprocessor"]

# Transform the test data
X_test_fe_transformed = preprocessor_fe_fitted.transform(X_test_fe)

# Extract the actual XGBoost model from the pipeline
xgb_fe_regressor = xgb_fe_model.named_steps["regressor"]

# Create SHAP explainer
explainer_fe = shap.TreeExplainer(
    xgb_fe_regressor
)

# Calculate SHAP values
shap_values_fe = explainer_fe.shap_values(
    X_test_fe_transformed
)

# Get names of all transformed features
feature_names_fe = (
    preprocessor_fe_fitted.get_feature_names_out()
)

print("\n===== SHAP - FEATURE-ENGINEERED XGBOOST =====")

print(
    "Number of transformed features:",
    len(feature_names_fe)
)

print(
    "SHAP shape:",
    shap_values_fe.shape
)


# ============================================================
# CONVERT TRANSFORMED DATA TO DATAFRAME
# ============================================================

X_test_fe_shap = (
    X_test_fe_transformed.toarray()
    if hasattr(X_test_fe_transformed, "toarray")
    else X_test_fe_transformed
)

X_test_fe_shap = pd.DataFrame(
    X_test_fe_shap,
    columns=feature_names_fe,
    index=X_test_fe.index
)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

shap.summary_plot(
    shap_values_fe,
    X_test_fe_shap,
    feature_names=feature_names_fe
)

# ============================================================
# INDIVIDUAL HOUSE SHAP
# ============================================================

house_index = 0

house_shap_values_fe = shap_values_fe[house_index]

house_data_fe = X_test_fe_shap.iloc[house_index]

print("\n===== FEATURE-ENGINEERED HOUSE =====")

print(X_test_fe.iloc[house_index])

actual_price_fe = y_test_fe.iloc[house_index]

predicted_price_fe = xgb_fe_model.predict(
    X_test_fe.iloc[[house_index]]
)[0]

print("\nActual price:", actual_price_fe)
print("Predicted price:", predicted_price_fe)
print(
    "Error:",
    actual_price_fe - predicted_price_fe
)


# ============================================================
# WATERFALL PLOT
# ============================================================

shap.plots._waterfall.waterfall_legacy(
    explainer_fe.expected_value,
    house_shap_values_fe,
    feature_names=feature_names_fe
)