import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

#data is loaded from the processed folder
data = pd.read_csv(r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\processed\model2_routes_data.csv")
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
#splitting the data into train and test sets
split_point = int(len(data) * 0.8)
train = data[:split_point]
test = data[split_point:]
tscv = TimeSeriesSplit(n_splits=5)
#training the models using the features DCOILBRENTEU and wci_lastweek to predict wci
features = ["wci_lastweek", "wci_lag2", "wci_lag3", "wci_change", "wci_roll3", "oil_lastweek", "oil_roll3"]

# train and save one model for each route
for route in routes:
    features = [f"{route}_lastweek", f"{route}_lag2", f"{route}_lag3", f"{route}_change",
                f"{route}_roll3", "oil_lastweek", "oil_roll3"]
    print("-----", route)
    #training using linear regression
    lr = LinearRegression().fit(train[features], train[route])
    print("Linear Regression MAE:", mean_absolute_error(test[route], lr.predict(test[features])))

    #training using random forest 
    rf = GridSearchCV(RandomForestRegressor(), {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
                  cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1).fit(train[features], train[route])
    print("Random Forest MAE:", mean_absolute_error(test[route], rf.predict(test[features])))

    #training using XGBoost
    xgb = GridSearchCV(XGBRegressor(), {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
                   "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 1.0]},
                   cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1).fit(train[features], train[route])
    print("XGBoost MAE:", mean_absolute_error(test[route], xgb.predict(test[features])))

    naive_mae = mean_absolute_error(test[route], test[features].iloc[:, 0])
    print("Naive guess MAE:", naive_mae)
