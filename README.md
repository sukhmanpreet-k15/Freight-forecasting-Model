# Freight Forecasting Model (SIH26006)

## What does this project do?

Ships carry goods all over the world. The price to hire a ship (the freight rate) changes all the time. This project **predicts the next freight rate** and shows it in a simple dashboard.

There are two models:

- **Model 1 (ship cost):** predicts the daily rate for 4 ship types: Handysize (HSI), Supramax (SI), Panamax (PI) and Capesize (CI). It uses past ship prices and the Brent oil price.
- **Model 2 (route cost):** predicts the weekly container freight price (Drewry WCI) for 7 sea routes. It uses past prices of the route and the Brent oil price.

## Data used

| Data | What it is |
|------|------------|
| Baltic sub-indices (HSI, SI, PI, CI) | Daily price index for each ship type |
| Brent oil price (`DCOILBRENTEU`) | Daily price of Brent crude oil |
| Drewry WCI (per route) | Weekly container freight price for each route |

The WCI data is downloaded with the `akshare` library.

## Routes in Model 2

1. Shanghai to Rotterdam
2. Rotterdam to Shanghai
3. Shanghai to Los Angeles
4. Los Angeles to Shanghai
5. Shanghai to Genoa
6. New York to Rotterdam
7. Rotterdam to New York

## How the models work

**Model 1** uses 7 inputs for each ship type:
- yesterday's value
- the value 2 days ago
- the value 3 days ago
- the change from the day before
- the 3-day average
- yesterday's oil price
- the 3-day average of the oil price

It uses Linear Regression. There is one model for each ship type (4 models).

**Model 2** uses the same 7 inputs for each route, but with weeks instead of days (last week, 2 weeks ago, 3 weeks ago, change, 3-week average, and the oil price of last week and its 3-week average). There is one model for each route (7 models).

For each route, three algorithms are tested:
- Linear Regression
- Random Forest
- XGBoost

The algorithm with the lowest error on the test data is saved. The error is also compared with a "naive guess" (copy last week's value), to check that the model is really useful.

The old 80% of the data is used for learning, and the newest 20% is used for testing. Only past values are used as inputs, so the model never sees the future.

## Dashboard

The dashboard is made with Streamlit and Plotly. It has a sidebar where you choose what to predict:

- **Ship cost (Model 1):**
  - 4 boxes showing the expected rate for each ship type
  - 4 line charts, one for each ship type (past to now)
  - an oil price chart
  - a donut chart comparing the 4 ship types
- **Route cost (Model 2):**
  - a dropdown to pick a route
  - top boxes: current rate, oil price and predicted next week
  - a line chart of the chosen route, with month and year range buttons
  - an oil price chart

The time range buttons work the same for both models.

## Folders and files

```
Freight-forecasting-Model/
├── data/
│   ├── raw/
│   │   └── brent_oil.csv
│   └── processed/
│       ├── model1_data.csv
│       └── model2_routes_data.csv
├── models/
│   ├── model1_HSI.pkl
│   ├── model1_SI.pkl
│   ├── model1_PI.pkl
│   ├── model1_CI.pkl
│   ├── model2_shanghai_rotterdam.pkl
│   ├── model2_rotterdam_shanghai.pkl
│   ├── model2_shanghai_los_angeles.pkl
│   ├── model2_los_angeles_shanghai.pkl
│   ├── model2_shanghai_genoa.pkl
│   ├── model2_new_york_rotterdam.pkl
│   └── model2_rotterdam_new_york.pkl
├── notebooks/
│   ├── download_oil.py
│   ├── download_wci.py
│   ├── cleaning_data.py
│   ├── cleaning_data2.py
│   ├── joining_datasets.py
│   ├── model1_training.py
│   ├── model2_algo_checck.py
│   ├── model2_training.py
│   └── update_data.py
├── dashboard.py
└── README.md
```

### `data/raw/`
Original data, before any cleaning.
- `brent_oil.csv` – Brent oil price, correct up to 2026-09-17.

### `data/processed/`
Clean data, ready for training the models.
- `model1_data.csv` – daily data for Model 1 (weekdays). Columns: date, HSI, SI, PI, CI, DTI, DCOILBRENTEU. It has 1749 rows, from 2012-08-01 up to 2019. The DTI column is not used for training.
- `model2_routes_data.csv` – weekly (Thursday) data for Model 2. Columns: date, the 7 routes, and DCOILBRENTEU (oil price). It has 550 rows, from 2016-03-10 to 2026-09-17.

### `models/`
Trained models, saved as `.pkl` files.
- `model1_<ship>.pkl` – 4 files, one for each ship type.
- `model2_<route>.pkl` – 7 files, one for each route.

### `notebooks/`
Python scripts used to get the data, clean it and train the models. Run them in this order:

| File | What it does |
|------|--------------|
| `download_oil.py` | Downloads the Brent oil price |
| `download_wci.py` | Downloads the Drewry WCI data for the routes (with `akshare`) |
| `cleaning_data.py` | Cleans the data for Model 1 |
| `cleaning_data2.py` | Cleans the data for Model 2 |
| `joining_datasets.py` | Joins the ship, route and oil data into the processed files |
| `model1_training.py` | Trains and saves the 4 ship type models (Model 1) |
| `model2_algo_checck.py` | Tests Linear Regression, Random Forest and XGBoost for the routes |
| `model2_training.py` | Trains and saves the 7 route models (Model 2) |
| `update_data.py` | Meant to update the data with the latest values (does not work for fresh values yet, see "Known limitation") |

### `dashboard.py`
The code of the dashboard.

## Libraries needed

| Library | Why it is needed |
|---------|------------------|
| `pandas` | Read and prepare the data |
| `numpy` | Number work |
| `scikit-learn` | Linear Regression, Random Forest, testing and error checks |
| `xgboost` | XGBoost model |
| `joblib` | Save and load the `.pkl` models |
| `plotly` | Draw the charts |
| `streamlit` | Run the dashboard |
| `akshare` | Download the WCI data |

Install all of them with one command:

```
pip install pandas numpy scikit-learn xgboost joblib plotly streamlit akshare
```

## How to run

1. Open the project folder in VS Code.
2. Install the libraries (command above).
3. Run the dashboard:

```
streamlit run dashboard.py
```

## Known limitation

**The models do not work on the latest values.** We could not find any source that gives updated data for free. Because of this:

- The data files stop at a fixed date:
  - Model 1 (ship types) data ends in 2019.
  - Model 2 (routes) data ends on 2026-09-17.
- The dashboard shows predictions from this older data, and not from today's prices.
- The `update_data.py` script cannot bring in fresh values until a source for updated data is found.

This will be fixed when a source with daily or weekly updated data is available. Then the data can be refreshed and the models can be trained again.

## Note

Freight prices change slowly, so a simple "copy last week" guess is hard to beat. That is why each model is compared with the naive guess before it is trusted.