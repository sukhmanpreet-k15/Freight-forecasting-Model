import pandas as pd
import akshare as ak
def update_oil():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
    df = pd.read_csv(url)
    df.columns = ["date", "DCOILBRENTEU"]
    df["date"] = pd.to_datetime(df["date"])
    df["DCOILBRENTEU"] = pd.to_numeric(df["DCOILBRENTEU"], errors="coerce")
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    df["DCOILBRENTEU"] = df["DCOILBRENTEU"].ffill().bfill()
    return df

