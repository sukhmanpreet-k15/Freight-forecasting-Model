import pandas as pd
import joblib
import streamlit as st

st.title("Freight Forecasting - Model 1")

df = pd.read_csv("C:\\Users\\sukhm\\OneDrive\\Desktop\\pydev\\Freight-forecasting-Model\\data\\processed\\model1_data.csv")
df["date"] = pd.to_datetime(df["date"])

last = df.iloc[-1]

st.write("Latest data date:", last["date"].date())
st.caption("Expected values are for the next working day after this date.")

ships = {
    "HSI": "Handysize (HSI)",
    "SI": "Supramax (SI)",
    "PI": "Panamax (PI)",
    "CI": "Capesize (CI)",
}

folder = "C:\\Users\\sukhm\\OneDrive\\Desktop\\pydev\\Freight-forecasting-Model\\models\\"
oil = df["DCOILBRENTEU"]

columns = st.columns(4)

for col, (ship, name) in zip(columns, ships.items()):
    model = joblib.load(folder + f"model1_{ship}.pkl")
    s = df[ship]

    # same 7 features the model was trained with, for the next day
    row = pd.DataFrame([{
        f"{ship}_yesterday": s.iloc[-1],
        f"{ship}_lag2": s.iloc[-2],
        f"{ship}_lag3": s.iloc[-3],
        f"{ship}_change": s.iloc[-1] - s.iloc[-2],
        f"{ship}_roll3": s.iloc[-3:].mean(),
        "oil_yesterday": oil.iloc[-1],
        "oil_roll3": oil.iloc[-3:].mean(),
    }])

    pred = model.predict(row)[0]
    col.metric(name + " - expected", int(round(pred)), int(round(pred - last[ship])))