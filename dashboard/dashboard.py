import pandas as pd
import streamlit as st

st.title("Freight Forecasting - Model 1")

df = pd.read_csv("C:\\Users\\sukhm\\OneDrive\\Desktop\\pydev\\Freight-forecasting-Model\\data\\processed\\model1_data.csv")
df["date"] = pd.to_datetime(df["date"])

last = df.iloc[-1]
prev = df.iloc[-2]

st.write("Latest data date:", last["date"].date())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Handysize (HSI)", int(last["HSI"]), int(last["HSI"] - prev["HSI"]))
c2.metric("Supramax (SI)", int(last["SI"]), int(last["SI"] - prev["SI"]))
c3.metric("Panamax (PI)", int(last["PI"]), int(last["PI"] - prev["PI"]))
c4.metric("Capesize (CI)", int(last["CI"]), int(last["CI"] - prev["CI"]))