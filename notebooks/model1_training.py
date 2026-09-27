import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Load processed data
model1_data = pd.read_csv("C:\\Users\\sukhm\\OneDrive\\Desktop\\pydev\\Freight-forecasting-Model\\data\\processed\\model1_data.csv")
model1_data["CI_yesterday"] = model1_data["CI"].shift(1)
model1_data = model1_data.dropna().reset_index(drop=True)

split_point = int(len(model1_data) * 0.8)
train = model1_data[:split_point]
test = model1_data[split_point:]

X_train = train[["DCOILBRENTEU", "CI_yesterday"]]
y_train = train["CI"]
X_test = test[["DCOILBRENTEU", "CI_yesterday"]]
y_test = test["CI"]

# Time-respecting cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))

# 2. Random Forest
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None]
}
rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)
print("Random Forest best params:", rf_grid.best_params_)
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))

# 3. XGBoost
xgb_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0]
}
xgb_grid = GridSearchCV(XGBRegressor(), xgb_params, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test)
print("XGBoost best params:", xgb_grid.best_params_)
print("XGBoost MAE:", mean_absolute_error(y_test, xgb_pred))