import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

model2_data = pd.read_csv(r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\processed\model2_data.csv")

# last week's price as a feature
model2_data["wci_lastweek"] = model2_data["wci"].shift(1)
model2_data = model2_data.dropna().reset_index(drop=True)

split_point = int(len(model2_data) * 0.8)
train = model2_data[:split_point]
test = model2_data[split_point:]

X_train = train[["DCOILBRENTEU", "wci_lastweek"]]
y_train = train["wci"]
X_test = test[["DCOILBRENTEU", "wci_lastweek"]]
y_test = test["wci"]

tscv = TimeSeriesSplit(n_splits=5)

lr = LinearRegression().fit(X_train, y_train)
print("Linear Regression MAE:", mean_absolute_error(y_test, lr.predict(X_test)))

rf = GridSearchCV(RandomForestRegressor(), {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
                  cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train, y_train)
print("Random Forest MAE:", mean_absolute_error(y_test, rf.predict(X_test)))

xgb = GridSearchCV(XGBRegressor(), {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
                   "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 1.0]},
                   cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train, y_train)
print("XGBoost MAE:", mean_absolute_error(y_test, xgb.predict(X_test)))

#saving model2 using joblib
import joblib

lr = LinearRegression()
lr.fit(X_train, y_train)

joblib.dump(lr, r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\models\model2_wci.pkl")
print("Model 2 saved.")