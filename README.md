# House Price Prediction API


A production-style machine learning project for predicting residential house prices using **XGBoost**, exposed through a **FastAPI REST API**, containerized with **Docker**, tested with **pytest**, tracked with **MLflow**, and deployed to the cloud through **GitHub Actions, GHCR, and Render**.


## 🚀 Live Demo


- **Live API:** https://house-price-api-u7zo.onrender.com
- **Swagger UI:** https://house-price-api-u7zo.onrender.com/docs


### Example Prediction


A request to the deployed `/predict` endpoint successfully returned:


```json
{
  "predicted_price": 208587.9375
}
🏗️ Architecture
                         Ames Housing Dataset
                                  │
                                  ▼
                         Feature Engineering
                                  │
                                  ▼
                    Data Preprocessing Pipeline
                    ┌────────────┴────────────┐
                    │                         │
               Numerical                  Categorical
               Features                   Features
                    │                         │
             Median Imputation          Most-Frequent
             Standard Scaling            Imputation
                                             │
                                        One-Hot Encoding
                    └────────────┬────────────┘
                                 ▼
                         XGBoost Regressor
                                 │
                                 ▼
                       Model Evaluation
                                 │
                                 ▼
                              MLflow
                       ┌─────────┴─────────┐
                       │                   │
                 Experiment Tracking   Model Registry
                                           │
                                      @champion
                                           │
                                           ▼
                                      Model Artifact
                                           │
                                           ▼
                                      FastAPI API
                                           │
                                           ▼
                                         Docker
                                           │
                              ┌────────────┴────────────┐
                              │                         │
                       GitHub Actions                 GHCR
                       CI/CD Pipeline            Docker Registry
                              │                         │
                              └────────────┬────────────┘
                                           ▼
                                         Render
                                           │
                                           ▼
                                  Public REST API

📊 Dataset

The project uses the Ames Housing dataset, containing 1,460 residential property records and 81 columns.

The target variable is: SalePrice
The model uses the remaining housing attributes as input features.

Examples include:

Overall house quality
Living area
Basement area
Garage characteristics
Number of bathrooms
Year built
Year remodeled
Neighborhood
Exterior materials
Porch/deck areas
Lot characteristics


🧠 Feature Engineering

Domain-informed features are created before model training.

TotalSF

Combines basement, first-floor, and second-floor square footage:

TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
HouseAge

Age of the property when it was sold:

HouseAge = YrSold - YearBuilt
YearsSinceRemod

Years since the property's last remodeling:

YearsSinceRemod = YrSold - YearRemodAdd
TotalBathrooms

Converts bathrooms into a single numerical feature:

TotalBathrooms =
    FullBath
    + 0.5 × HalfBath
    + BsmtFullBath
    + 0.5 × BsmtHalfBath
TotalPorchSF

Combines the different outdoor/porch areas:

TotalPorchSF =
    OpenPorchSF
    + EnclosedPorch
    + 3SsnPorch
    + ScreenPorch
TotalBsmtFinished

Combines finished basement areas:

TotalBsmtFinished =
    BsmtFinSF1 + BsmtFinSF2



🤖 Machine Learning Pipeline

The project uses an end-to-end Scikit-learn pipeline.

Numerical features
Missing values are replaced using median imputation.
Features are standardized using StandardScaler.
Categorical features
Missing values are replaced using the most frequent category.
Categories are converted using OneHotEncoder.
Unknown categories are safely ignored during inference.
Model

The final estimator is an XGBoost Regressor.

Current configuration:

n_estimators = 400
learning_rate = 0.05
max_depth = 3
random_state = 42

The preprocessing and model are stored together in a single Scikit-learn pipeline.



📈 Model Evaluation

The model was evaluated using 5-fold cross-validation and a held-out test set.

5-Fold Cross-Validation
Metric	Average
MAE	$15,393.65
RMSE	$28,670.75
R²	0.8483
Final Test Set
Metric	Result
MAE	$15,440.61
RMSE	$25,637.48
R²	0.9143
Training Set
Metric	Result
MAE	$7,948.18
R²	0.9798

The difference between training and test performance indicates some degree of overfitting. Further hyperparameter tuning and regularization could be used to improve generalization.



⚡ FastAPI

The trained model is exposed through a REST API using FastAPI.

Endpoints
GET /

Health/basic API endpoint.

Example response:

{
  "message": "House Price Prediction API"
}

POST /predict

Accepts housing characteristics and returns a predicted house price.

Example response:

{
  "predicted_price": 208587.9375
}

Interactive API Documentation

Swagger UI is available at:

https://house-price-api-u7zo.onrender.com/docs

The API uses Pydantic for request validation.

Fields containing dataset column names that begin with numbers are handled using Pydantic aliases, for example:

1stFlrSF
2ndFlrSF
3SsnPorch


🧪 Testing

The project uses pytest for automated testing.

Current test suite:

tests/
├── test_api.py
├── test_features.py
└── test_prediction.py

The test suite covers:

API endpoints
Request validation
Feature engineering
Model prediction
Prediction output type

Current result:

5 passed


🐳 Docker

The application is containerized using Docker.

Build the image:

docker build -t house-price-api .

Run the application:

docker run -p 8000:8000 house-price-api

The container is configured to use the PORT environment variable when provided by the deployment platform.

For local development, the application defaults to port 8000.


🔄 CI/CD

GitHub Actions is used to automatically validate the project when changes are pushed.

The CI pipeline performs:

Git Push
   ↓
GitHub Actions
   ↓
Install Dependencies
   ↓
Run pytest
   ↓
Build Docker Image
   ↓
Publish Docker Image

The Docker image is published to GitHub Container Registry (GHCR) after the tests and Docker build succeed.

This prevents a failing test or Docker build from being published as the latest container image.



📦 Docker Image

The Docker image is published to GitHub Container Registry:

ghcr.io/proncyton/house-price-prediction-api:main

It can be pulled using:

docker pull ghcr.io/proncyton/house-price-prediction-api:main



📊 MLflow

MLflow is used for experiment tracking and model management.

The training pipeline records:

Experiment information
Training runs
Model parameters
Evaluation metrics
Model artifacts

The experiment is:

house-price-prediction


🗂️ MLflow Model Registry

The trained model is registered in MLflow as:

house-price-model

The project uses a model alias:

@champion

The alias allows model versions to be referenced logically rather than hard-coding a specific model version.

For example:

models:/house-price-model@champion

This makes it possible to promote a newer validated model without changing the application code that references the model.

Current implementation note: MLflow experiment tracking and model registry are implemented in the training workflow. The currently deployed FastAPI service uses the packaged house_price_xgb.joblib model. Integrating MLflow Model Registry directly into the deployed inference service is planned as a future improvement.


☁️ Deployment

The API is deployed using Render.

Deployment architecture:

GitHub
   ↓
GitHub Actions
   ↓
Docker Build
   ↓
GitHub Container Registry
   ↓
Render
   ↓
FastAPI
   ↓
XGBoost Prediction

The deployed service successfully handles real prediction requests through the public /predict endpoint.

Live API:

https://house-price-api-u7zo.onrender.com

Swagger:

https://house-price-api-u7zo.onrender.com/docs


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
1. Clone the repository
git clone https://github.com/proncyton/house-price-prediction-api.git
cd house-price-prediction-api
2. Create a virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run tests
pytest

Expected result:

5 passed
5. Start the API
uvicorn app.main:app --reload

Open:

http://localhost:8000/docs


🐋 Running with Docker Compose
docker compose up

The API will be available at:
http://localhost:8000

Swagger:
http://localhost:8000/docs



🔮 Future Improvements
 Serve the production model directly from the MLflow Model Registry
 Add automated model performance gates to CI/CD
 Add automated model retraining
 Add hyperparameter optimization
 Investigate and reduce model overfitting
 Add data-quality validation
 Add data and model drift monitoring
 Replace legacy XGBoost pickle/joblib serialization with a more robust model persistence approach
 Add production logging and monitoring
 Add API authentication and rate limiting
 Add integration tests for the deployed service




🧰 Tech Stack
Category	Technology
Language	Python
Data Processing	Pandas, NumPy
Machine Learning	Scikit-learn, XGBoost
API	FastAPI, Pydantic
Testing	Pytest
Experiment Tracking	MLflow
Containerization	Docker
CI/CD	GitHub Actions
Container Registry	GitHub Container Registry
Cloud Deployment	Render
Model Persistence	Joblib