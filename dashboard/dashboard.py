import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
 
# ---------- SIDEBAR (used by both models) ----------
page = st.sidebar.radio("What do you want to see?", ["Ship cost (Model 1)", "Container cost - WCI (Model 2)"])
choice = st.sidebar.radio("Time range", ["1M", "3M", "6M", "1Y", "All"], index=3)
months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}
 
folder = "C:\\Users\\sukhm\\OneDrive\\Desktop\\pydev\\Freight-forecasting-Model\\models\\"
 
 
# keeps only the rows inside the chosen time range (counted back from the last date of that data)
def get_view(data, last_date):
    if choice == "All":
        return data
    start = last_date - pd.DateOffset(months=months[choice])
    return data[data["date"] >= start]
 
 
# =====================================================================
# PAGE 1: SHIP COST (MODEL 1)
# =====================================================================
if page == "Ship cost (Model 1)":
    st.title("Freight Forecasting Model")
 
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
 
    oil = df["DCOILBRENTEU"]
    preds = {}
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
        preds[ship] = pred
        col.metric(name + " - expected", int(round(pred)), int(round(pred - last[ship])))
 
    view = get_view(df, last["date"])
 
    # part 3 for graphs
    def make_chart(ship, name):
        next_day = last["date"] + pd.offsets.BDay(1)
        fig = go.Figure()
 
        # blue line with color under it (the past)
        fig.add_scatter(x=view["date"], y=view[ship], name="Actual", fill="tozeroy",
                        line=dict(color="#08eef6"), hovertemplate="%{y:.0f}")
 
        # dotted line to the prediction (no hover on the line itself)
        fig.add_scatter(x=[last["date"], next_day], y=[last[ship], preds[ship]],
                        line=dict(color="#860cf0", dash="dot"), hoverinfo="skip")
 
        # the predicted point (this one shows "Expected" on hover)
        fig.add_scatter(x=[next_day], y=[preds[ship]], name="Expected", mode="markers",
                        marker=dict(color="#93c5fd", size=9), hovertemplate="%{y:.0f}")
 
        fig.update_yaxes(range=[min(view[ship].min(), preds[ship]) * 0.95, max(view[ship].max(), preds[ship]) * 1.05])
        fig.update_xaxes(hoverformat="%d %b %Y")
        fig.update_layout(template="plotly_dark", title=name, height=300,
                          showlegend=False, hovermode="x unified",
                          hoverlabel=dict(bgcolor="#830aed", bordercolor="#04f9c0",
                                          font=dict(size=15, color="white")))
        return fig
 
    left, right = st.columns(2)
 
    for i, (ship, name) in enumerate(ships.items()):
        place = left if i % 2 == 0 else right
        place.plotly_chart(make_chart(ship, name))
 
 
# =====================================================================
# PAGE 2: CONTAINER COST - WCI (MODEL 2)
# =====================================================================
else:
    st.title("Freight Forecasting Model - WCI")
 
    df2 = pd.read_csv(r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\processed\model2_data.csv")
    df2["date"] = pd.to_datetime(df2["date"])
 
    last2 = df2.iloc[-1]
 
    st.write("Latest data date:", last2["date"].date())
    st.caption("Expected value is for the next week after this date.")
 
    wci = df2["wci"]
    oil2 = df2["DCOILBRENTEU"]
 
    model2 = joblib.load(folder + "model2_wci.pkl")
 
    # same 7 features the model was trained with, for the next week
    row2 = pd.DataFrame([{
        "wci_lastweek": wci.iloc[-1],
        "wci_lag2": wci.iloc[-2],
        "wci_lag3": wci.iloc[-3],
        "wci_change": wci.iloc[-1] - wci.iloc[-2],
        "wci_roll3": wci.iloc[-3:].mean(),
        "oil_lastweek": oil2.iloc[-1],
        "oil_roll3": oil2.iloc[-3:].mean(),
    }])
 
    pred2 = model2.predict(row2)[0]
 
    st.metric("WCI - expected next week", int(round(pred2)), int(round(pred2 - last2["wci"])))

    view2 = get_view(df2, last2["date"])
    next_week = last2["date"] + pd.Timedelta(weeks=1)
 
    fig2 = go.Figure()
 
    # line with color under it (the past)
    fig2.add_scatter(x=view2["date"], y=view2["wci"], name="Actual", fill="tozeroy",
                     line=dict(color="#08eef6"), hovertemplate="%{y:.0f}")
 
    # dotted line to the prediction (no hover on the line itself)
    fig2.add_scatter(x=[last2["date"], next_week], y=[last2["wci"], pred2],
                     line=dict(color="#860cf0", dash="dot"), hoverinfo="skip")