import numpy as np
import pandas as pd

from src.predict import predict_house_price


def test_prediction():
    df = pd.read_csv("data/housing.csv")

    house = df.drop(columns=["SalePrice"]).iloc[[0]]

    prediction = predict_house_price(house)

    assert len(prediction) == 1
    assert isinstance(prediction[0], (float, np.floating))
    assert prediction[0] > 0