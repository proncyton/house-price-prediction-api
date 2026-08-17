from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_house_price


app = FastAPI(
    title="House Price Prediction API",
    description="XGBoost house price prediction service",
    version="1.0"
)


class HouseInput(BaseModel):
    MSSubClass: int
    MSZoning: str
    LotFrontage: float | None = None
    LotArea: int
    Street: str
    Alley: str | None = None
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str | None = None
    MasVnrArea: float | None = None
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str | None = None
    BsmtCond: str | None = None
    BsmtExposure: str | None = None
    BsmtFinType1: str | None = None
    BsmtFinSF1: float | None = None
    BsmtFinType2: str | None = None
    BsmtFinSF2: float | None = None
    BsmtUnfSF: float | None = None
    TotalBsmtSF: float | None = None
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str | None = None
    first_floor_sf: int= Field(alias="1stFlrSF")
    second_floor_sf: int= Field(alias="2ndFlrSF")
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: float | None = None
    BsmtHalfBath: float | None = None
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: str | None = None
    GarageType: str | None = None
    GarageYrBlt: float | None = None
    GarageFinish: str | None = None
    GarageCars: float | None = None
    GarageArea: float | None = None
    GarageQual: str | None = None
    GarageCond: str | None = None
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    three_season_porch:int = Field(alias="3SsnPorch")
    ScreenPorch: int
    PoolArea: int
    PoolQC: str | None = None
    Fence: str | None = None
    MiscFeature: str | None = None
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API"
    }


@app.post("/predict")
def predict(data: HouseInput):

    prediction = predict_house_price(
        [data.model_dump(by_alias=True)]
    )[0]

    return {
        "predicted_price": float(prediction)
    }