import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# read the clean Model 2 table
data = pd.read_csv(r"data/processed/model2_routes_data.csv")
data["date"] = pd.to_datetime(data["date"])

routes = ["shanghai_rotterdam", "rotterdam_shanghai", "shanghai_los_angeles",
          "los_angeles_shanghai", "shanghai_genoa", "new_york_rotterdam", "rotterdam_new_york"]

# oil inputs (the same for every route)
data["oil_lastweek"] = data["DCOILBRENTEU"].shift(1)
data["oil_roll3"] = data["DCOILBRENTEU"].shift(1).rolling(3).mean()

# route inputs
for route in routes:
    data[f"{route}_lastweek"] = data[route].shift(1)
    data[f"{route}_lag2"] = data[route].shift(2)
    data[f"{route}_lag3"] = data[route].shift(3)
    data[f"{route}_change"] = data[route].shift(1).diff(1)
    data[f"{route}_roll3"] = data[route].shift(1).rolling(3).mean()

data = data.dropna().reset_index(drop=True)

# train one model for each route and save it
for route in routes:
    features = [f"{route}_lastweek", f"{route}_lag2", f"{route}_lag3", f"{route}_change",
                f"{route}_roll3", "oil_lastweek", "oil_roll3"]

    model = LinearRegression()
    model.fit(data[features], data[route])

    joblib.dump(model, f"models/model2_{route}.pkl")
    print(route, "saved.")