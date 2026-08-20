# 🏠 House Price Prediction API

> A production-style machine learning API for predicting residential house prices using **XGBoost**, **FastAPI**, **Docker**, **MLflow**, **GitHub Actions**, **GHCR**, and **Render**.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-orange?logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow&logoColor=white)
![Pytest](https://img.shields.io/badge/Tests-5%20passed-brightgreen?logo=pytest)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render&logoColor=black)

</p>

---

## 🚀 Live Demo

<table>
<tr>
<td align="center" width="50%">

### 🌐 Live API

[**Open API**](https://house-price-api-u7zo.onrender.com)

`https://house-price-api-u7zo.onrender.com`

</td>

<td align="center" width="50%">

### 📖 Swagger UI

[**Open Swagger**](https://house-price-api-u7zo.onrender.com/docs)

`/docs`

</td>
</tr>
</table>

### Example Prediction

A real request to the deployed API successfully returned:

```json
{
  "predicted_price": 208587.9375
}



┌──────────────────────┐
│       Housing        │
│       Dataset        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Feature Engineering  │
│                      │
│ • TotalSF            │
│ • HouseAge           │
│ • TotalBathrooms     │
│ • TotalPorchSF       │
│ • TotalBsmtFinished  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────┐
│       Preprocessing Pipeline         │
│                                      │
│  Numerical             Categorical   │
│  ─────────             ───────────   │
│  Median Imputation     Most Frequent │
│  Standard Scaling      Imputation   │
│                        One-Hot Encode│
└──────────────────┬───────────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ XGBoost Regressor│
          └────────┬────────┘
                   │
                   ▼
             ┌───────────┐
             │  MLflow   │
             │           │
             │ Tracking  │
             │ Metrics   │
             │ Registry  │
             └─────┬─────┘
                   │
                   ▼
             ┌───────────┐
             │  FastAPI  │
             │    API    │
             └─────┬─────┘
                   │
                   ▼
             ┌───────────┐
             │   Docker  │
             └─────┬─────┘
                   │
                   ▼
           ┌────────────────┐
           │ GitHub Actions │
           │      CI/CD     │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │      GHCR      │
           │ Docker Registry│
           └───────┬────────┘
                   │
                   ▼
             ┌───────────┐
             │  Render   │
             │   Cloud   │
             └─────┬─────┘
                   │
                   ▼
          🌐 Public REST API

```

📊 Model Performance

The model was evaluated using 5-fold cross-validation and a held-out test set.

<table> <tr> <th>Metric</th> <th>5-Fold CV</th> <th>Final Test</th> <th>Training</th> </tr> <tr> <td><strong>MAE</strong></td> <td>$15,393.65</td> <td>$15,440.61</td> <td>$7,948.18</td> </tr> <tr> <td><strong>RMSE</strong></td> <td>$28,670.75</td> <td>$25,637.48</td> <td>—</td> </tr> <tr> <td><strong>R²</strong></td> <td>0.8483</td> <td><strong>0.9143</strong></td> <td>0.9798</td> </tr> </table>
📌 Interpretation

The final test-set R² of 0.9143 indicates that the model explains approximately 91.4% of the variance in house sale prices on the held-out test data.

The difference between training and test performance also indicates some degree of overfitting, leaving room for further regularization and hyperparameter optimization.


🧠 Feature Engineering

Several domain-informed features are generated before training.

TotalSF
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF

Represents the total residential floor area.

HouseAge
HouseAge = YrSold - YearBuilt

Represents the age of the property when it was sold.

YearsSinceRemod
YearsSinceRemod = YrSold - YearRemodAdd

Represents how long it had been since the property was remodeled.

TotalBathrooms
TotalBathrooms =
    FullBath
    + 0.5 × HalfBath
    + BsmtFullBath
    + 0.5 × BsmtHalfBath
TotalPorchSF
TotalPorchSF =
    OpenPorchSF
    + EnclosedPorch
    + 3SsnPorch
    + ScreenPorch
TotalBsmtFinished
TotalBsmtFinished =
    BsmtFinSF1 + BsmtFinSF2


🤖 Machine Learning Pipeline

The project uses a Scikit-learn Pipeline combining preprocessing and the final estimator.

Numerical Features
Missing Values
      ↓
Median Imputation
      ↓
StandardScaler
Categorical Features
Missing Values
      ↓
Most-Frequent Imputation
      ↓
OneHotEncoder
      ↓
handle_unknown="ignore"
XGBoost
n_estimators = 400
learning_rate = 0.05
max_depth = 3
random_state = 42

The preprocessing and model are saved together as a single pipeline, ensuring that inference uses the same transformations applied during training.


⚡ FastAPI

The trained model is exposed through a REST API.

<table> <tr> <th>Method</th> <th>Endpoint</th> <th>Description</th> </tr> <tr> <td><code>GET</code></td> <td><code>/</code></td> <td>API health/basic endpoint</td> </tr> <tr> <td><code>POST</code></td> <td><code>/predict</code></td> <td>Predict house price</td> </tr> <tr> <td><code>GET</code></td> <td><code>/docs</code></td> <td>Interactive Swagger documentation</td> </tr> </table>


Example Request
{
  "MSSubClass": 60,
  "MSZoning": "RL",
  "LotArea": 8450,
  "OverallQual": 7,
  "OverallCond": 5,
  "YearBuilt": 2003,
  "GrLivArea": 1710,
  "FullBath": 2,
  "GarageCars": 2
}
Example Response
{
  "predicted_price": 208587.9375
}

The API uses Pydantic validation and supports the complete Ames Housing feature schema.

🧪 Testing

The project uses pytest for automated testing.

tests/
├── test_api.py
├── test_features.py
└── test_prediction.py

The test suite covers:

API endpoints
Request validation
Feature engineering
Model prediction
Prediction output

Current Result: 5 passed


🐳 Docker

The application is fully containerized.

Build
docker build -t house-price-api .
Run
docker run -p 8000:8000 house-price-api
Docker Compose
docker compose up

The application uses the PORT environment variable when supplied by the deployment platform and defaults to port 8000 locally.



🔄 CI/CD Pipeline

GitHub Actions automatically validates changes pushed to main.

                 git push
                    │
                    ▼
             GitHub Actions
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Install deps           Checkout
          │
          ▼
        pytest
          │
          ▼
     Docker Build
          │
          ▼
     ┌─────────────┐
     │   SUCCESS   │
     └──────┬──────┘
            │
            ▼
     Publish to GHCR
            │
            ▼
        Render

The pipeline prevents a failing test or Docker build from being published as the latest container image.

📦 GitHub Container Registry

The Docker image is published to GitHub Container Registry:

ghcr.io/proncyton/house-price-prediction-api:main

Pull the image:

docker pull ghcr.io/proncyton/house-price-prediction-api:main

This allows the deployment platform to consume the same container image produced by the CI/CD pipeline.


📈 MLflow Experiment Tracking

MLflow is used to track machine learning experiments.

The training pipeline records:
Model parameters
Cross-validation metrics
Test metrics
Training metrics
Model artifacts
Training runs

Experiment
house-price-prediction

Tracked Metrics
cv_mae_mean
cv_mae_std
cv_rmse_mean
cv_rmse_std
cv_r2_mean
cv_r2_std
test_mae
test_rmse
test_r2
train_mae
train_r2


🗂️ MLflow Model Registry

The model is registered as:

house-price-model

with the alias:

@champion

Example MLflow model URI:

models:/house-price-model@champion

The registry provides a mechanism for managing model versions independently from the training code.

Implementation note: MLflow tracking and model registry are implemented in the training workflow. The currently deployed FastAPI service uses the packaged house_price_xgb.joblib model. Direct MLflow Registry loading from the production inference service is a planned improvement.


☁️ Deployment

The API is deployed on Render using the Docker image published through GHCR.

GitHub
   │
   ▼
GitHub Actions
   │
   ├── pytest
   ├── Docker build
   └── GHCR publish
          │
          ▼
        GHCR
          │
          ▼
       Render
          │
          ▼
      FastAPI
          │
          ▼
      XGBoost
          │
          ▼
     Prediction

Production API: https://house-price-api-u7zo.onrender.com/

Interactive Swagger: https://house-price-api-u7zo.onrender.com/docs



📁 Project Structure


house-price-prediction-api/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── housing.csv
│
├── models/
│   └── house_price_xgb.joblib
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── predict.py
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   └── test_prediction.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md


🛠️ Local Setup
1. Clone
git clone https://github.com/proncyton/house-price-prediction-api.git
cd house-price-prediction-api
2. Create Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Run Tests
pytest

Expected:
5 passed
5. Start API
uvicorn app.main:app --reload

Open:
http://localhost:8000/docs


🔮 Future Improvements
 Load production models directly from MLflow Model Registry
 Automated model evaluation gates
 Automated model retraining
 Hyperparameter optimization
 Reduce model overfitting
 Data-quality validation
 Data drift monitoring
 Model drift monitoring
 Production logging and monitoring
 API authentication
 API rate limiting
 Integration tests against the deployed API
 Replace legacy XGBoost pickle/joblib serialization


🧰 Tech Stack
<p align="center">
Category	Technology
🐍 Language	Python
📊 Data	Pandas, NumPy
🧠 Machine Learning	Scikit-learn, XGBoost
⚡ API	FastAPI, Pydantic
🧪 Testing	Pytest
📈 Experiment Tracking	MLflow
🐳 Containerization	Docker
🔄 CI/CD	GitHub Actions
📦 Registry	GitHub Container Registry
☁️ Deployment	Render
💾 Model Persistence	Joblib
</p>

⭐ Key Features
<table> <tr> <td align="center">🧠<br><strong>End-to-End ML</strong></td> <td align="center">📊<br><strong>Model Evaluation</strong></td> <td align="center">📈<br><strong>MLflow Tracking</strong></td> <td align="center">🗂️<br><strong>Model Registry</strong></td> </tr> <tr> <td align="center">⚡<br><strong>REST API</strong></td> <td align="center">🐳<br><strong>Dockerized</strong></td> <td align="center">🔄<br><strong>CI/CD</strong></td> <td align="center">☁️<br><strong>Cloud Deployed</strong></td> </tr> </table>


👤 Author
Sanchit Surve
Machine Learning / Data Science
Built as an end-to-end ML engineering project demonstrating:
Machine Learning → API Development → Testing → Containerization → Experiment Tracking → CI/CD → Cloud Deployment