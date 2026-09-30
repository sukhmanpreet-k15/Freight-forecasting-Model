import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

model1_data = pd.read_csv("C:\\Users\\sukhm\\OneDrive\\Desktop\\pydev\\Freight-forecasting-Model\\data\\processed\\model1_data.csv")

ship_types = ["HSI", "SI", "PI", "CI"]

# create feature columns for each ship type
for ship in ship_types:
    model1_data[f"{ship}_yesterday"] = model1_data[ship].shift(1)
    model1_data[f"{ship}_lag2"] = model1_data[ship].shift(2)
    model1_data[f"{ship}_lag3"] = model1_data[ship].shift(3)
    model1_data[f"{ship}_change"] = model1_data[ship].shift(1).diff(1)
    model1_data[f"{ship}_roll3"] = model1_data[ship].shift(1).rolling(3).mean()

model1_data["oil_yesterday"] = model1_data["DCOILBRENTEU"].shift(1)
model1_data["oil_roll3"] = model1_data["DCOILBRENTEU"].shift(1).rolling(3).mean()

model1_data = model1_data.dropna().reset_index(drop=True)

split_point = int(len(model1_data) * 0.8)
train = model1_data[:split_point]
test = model1_data[split_point:]

tscv = TimeSeriesSplit(n_splits=5)

results = {}

# ---- compare models for each ship type ----
for ship in ship_types:
    ship_features = [f"{ship}_yesterday", f"{ship}_lag2", f"{ship}_lag3", f"{ship}_change", f"{ship}_roll3", "oil_yesterday", "oil_roll3"]

    X_train = train[ship_features]
    y_train = train[ship]
    X_test = test[ship_features]
    y_test = test[ship]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_mae = mean_absolute_error(y_test, lr.predict(X_test))

    rf_grid = GridSearchCV(RandomForestRegressor(), {"n_estimators": [100, 200], "max_depth": [5, 10, None]}, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_mae = mean_absolute_error(y_test, rf_grid.predict(X_test))

    xgb_grid = GridSearchCV(XGBRegressor(), {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 1.0]}, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    xgb_mae = mean_absolute_error(y_test, xgb_grid.predict(X_test))

    naive_mae = mean_absolute_error(y_test, X_test[f"{ship}_yesterday"])

    results[ship] = {"Linear Regression": lr_mae, "Random Forest": rf_mae, "XGBoost": xgb_mae, "Naive Guess": naive_mae}
    print(ship, "done")

print(pd.DataFrame(results))

# ---- save final models ----
final_models = {}

for ship in ship_types:
    ship_features = [f"{ship}_yesterday", f"{ship}_lag2", f"{ship}_lag3", f"{ship}_change", f"{ship}_roll3", "oil_yesterday", "oil_roll3"]

    X_train = train[ship_features]
    y_train = train[ship]

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    final_models[ship] = lr
    joblib.dump(lr, f"C:\\Users\\sukhm\\OneDrive\\Desktop\\pydev\\Freight-forecasting-Model\\models\\model1_{ship}.pkl")

print("All 4 models saved.")