# Freight Forecasting Model (SIH26006)

## What does this project do?

Ships carry goods all over the world. The price to hire a ship (the freight rate) changes all the time. This project **predicts the next freight rate** and shows it in a dashboard.

There are two models:

- **Model 1 (ship types):** predicts the daily rate for 4 ship types: Handysize (HSI), Supramax (SI), Panamax (PI) and Capesize (CI). It uses past ship prices and the Brent oil price.
- **Model 2 (routes):** predicts the weekly container freight price (Drewry WCI) for 7 sea routes. It uses past prices of the route and the Brent oil price.

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

**Model 1** uses 7 inputs for each ship type: yesterday's value, 2 days ago, 3 days ago, the change, the 3-day average, yesterday's oil price and the 3-day average oil price. It uses Linear Regression. There is one model for each ship type (4 models).

**Model 2** uses 7 inputs for each route: last week's value, 2 weeks ago, 3 weeks ago, the change, the 3-week average, last week's oil price and the 3-week average oil price. There is one model for each route (7 models).

For each route, three algorithms are tested:
- Linear Regression
- Random Forest
- XGBoost

The algorithm with the lowest error on the test data is saved. The error is also compared with a "naive guess" (copy last week's value) to check that the model is really useful.

The old 80% of the data is used for learning, and the newest 20% is used for testing.

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
├── dashboard.py
└── README.md
```

### `data/raw/`
Original data, before any cleaning.
- `brent_oil.csv` – Brent oil price, correct up to 2026-09-17.

### `data/processed/`
Clean data, ready for training the models.
- `model1_data.csv` – daily data for Model 1. Columns: date, HSI, SI, PI, CI, DTI, DCOILBRENTEU. It has 1749 rows, starting from 2012-08-01. (The DTI column is not used for training.)
- `model2_routes_data.csv` – weekly (Thursday) data for Model 2. Columns: date, the 7 routes, and DCOILBRENTEU (oil price). It has 550 rows, from 2016-03-10 to 2026-09-17.

### `models/`
Trained models, saved as `.pkl` files.
- `model1_<ship>.pkl` – 4 files, one for each ship type.
- `model2_<route>.pkl` – 7 files, one for each route.

### `dashboard.py`
The dashboard. It shows the expected value for each ship type and line charts of past prices. You can change the time range from the sidebar.

## Libraries needed

| Library | Why it is needed |
|---------|------------------|
| `pandas` | Read and prepare the data |
| `numpy` | Number work |
| `scikit-learn` | Linear Regression, Random Forest, testing and error checks |
| `xgboost` | XGBoost model |
| `joblib` | Save and load the `.pkl` models |
| `plotly` | Draw the line charts |
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