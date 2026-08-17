import joblib
import pandas as pd
import os
from dotenv import load_dotenv

from src.features import feature_engineering


#MODEL_PATH = "models/house_price_xgb.joblib"
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/house_price_xgb.joblib"
)

# Load trained model
model = joblib.load(MODEL_PATH)


def predict_house_price(house_data):
    """
    Predict the sale price of one or more houses.
    """

    df = pd.DataFrame(house_data)

    # Apply exactly the same feature engineering
    df = feature_engineering(df)

    predictions = model.predict(df)

    return predictions


if __name__ == "__main__":

    test_house = {
        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotFrontage": 70,
        "LotArea": 8450,
        "Street": "Pave",
        "Alley": None,
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "CollgCr",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "BrkFace",
        "MasVnrArea": 196,
        "ExterQual": "Gd",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "Gd",
        "BsmtCond": "TA",
        "BsmtExposure": "Gd",
        "BsmtFinType1": "GLQ",
        "BsmtFinSF1": 706,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 150,
        "TotalBsmtSF": 856,
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "1stFlrSF": 856,
        "2ndFlrSF": 854,
        "LowQualFinSF": 0,
        "GrLivArea": 1710,
        "BsmtFullBath": 1,
        "BsmtHalfBath": 0,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "KitchenQual": "Gd",
        "TotRmsAbvGrd": 8,
        "Functional": "Typ",
        "Fireplaces": 0,
        "FireplaceQu": "None",
        "GarageType": "Attchd",
        "GarageYrBlt": 2003,
        "GarageFinish": "RFn",
        "GarageCars": 2,
        "GarageArea": 548,
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "WoodDeckSF": 0,
        "OpenPorchSF": 61,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "PoolQC": None,
        "Fence": None,
        "MiscFeature": None,
        "MiscVal": 0,
        "MoSold": 2,
        "YrSold": 2008,
        "SaleType": "WD",
        "SaleCondition": "Normal"
    }

    prediction = predict_house_price([test_house])

    print("Predicted house price:", prediction[0])