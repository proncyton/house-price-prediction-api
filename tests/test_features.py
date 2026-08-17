import pandas as pd

from src.features import feature_engineering


def test_feature_engineering():
    house = pd.DataFrame([
        {
            "TotalBsmtSF": 800,
            "1stFlrSF": 900,
            "2ndFlrSF": 700,
            "YrSold": 2010,
            "YearBuilt": 2000,
            "YearRemodAdd": 2005,
            "FullBath": 2,
            "HalfBath": 1,
            "BsmtFullBath": 1,
            "BsmtHalfBath": 0,
            "OpenPorchSF": 50,
            "EnclosedPorch": 20,
            "3SsnPorch": 0,
            "ScreenPorch": 30,
            "BsmtFinSF1": 500,
            "BsmtFinSF2": 100
        }
    ])

    result = feature_engineering(house)

    assert result["TotalSF"].iloc[0] == 2400
    assert result["HouseAge"].iloc[0] == 10
    assert result["YearsSinceRemod"].iloc[0] == 5
    assert result["TotalBathrooms"].iloc[0] == 3.5
    assert result["TotalPorchSF"].iloc[0] == 100
    assert result["TotalBsmtFinished"].iloc[0] == 600