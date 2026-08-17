import pandas as pd


def feature_engineering(df):
    """
    Add domain-informed features to the Ames housing dataset.
    """

    df = df.copy()

    # Total usable square footage
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

    # Years since last renovation
    df["YearsSinceRemod"] = (
        df["YrSold"]
        - df["YearRemodAdd"]
    )

    # Convert bathrooms into a single numerical measure
    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # Total outdoor/porch area
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