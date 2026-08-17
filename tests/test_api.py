import json

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "House Price Prediction API"
    }


def test_predict():
    df = pd.read_csv("data/housing.csv")

    house = json.loads(
        df.drop(columns=["SalePrice"])
        .iloc[[0]]
        .to_json(orient="records")
    )[0]

    response = client.post(
        "/predict",
        json=house
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_price" in data
    assert data["predicted_price"] > 0


def test_predict_invalid_input():
    response = client.post(
        "/predict",
        json={
            "MSSubClass": "hello"
        }
    )

    assert response.status_code == 422