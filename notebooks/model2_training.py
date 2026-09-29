import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
#data is loaded from the processed folder
model2_data = pd.read_csv(r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\processed\model2_data.csv")

# last week's price as a feature
model2_data["wci_lastweek"] = model2_data["wci"].shift(1)
model2_data["oil_lastweek"] = model2_data["DCOILBRENTEU"].shift(1)
model2_data = model2_data.dropna().reset_index(drop=True)
#splitting the data into train and test sets
split_point = int(len(model2_data) * 0.8)
train = model2_data[:split_point]
test = model2_data[split_point:]
#training the models using the features DCOILBRENTEU and wci_lastweek to predict wci
X_train = train[["oil_lastweek", "wci_lastweek"]]
y_train = train["wci"]
X_test = test[["oil_lastweek", "wci_lastweek"]]
y_test = test["wci"]

tscv = TimeSeriesSplit(n_splits=5)
#training and evaluating the models

#training using linear regression
lr = LinearRegression().fit(X_train, y_train)
print("Linear Regression MAE:", mean_absolute_error(y_test, lr.predict(X_test)))

#training using random forest 
rf = GridSearchCV(RandomForestRegressor(), {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
                  cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train, y_train)
print("Random Forest MAE:", mean_absolute_error(y_test, rf.predict(X_test)))

#training using XGBoost
xgb = GridSearchCV(XGBRegressor(), {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
                   "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 1.0]},
                   cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train, y_train)
print("XGBoost MAE:", mean_absolute_error(y_test, xgb.predict(X_test)))

import joblib

lr = LinearRegression()
lr.fit(X_train, y_train)

joblib.dump(lr, r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\models\model2_wci.pkl")
print("Model 2 saved.")