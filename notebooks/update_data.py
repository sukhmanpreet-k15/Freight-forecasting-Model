import pandas as pd
import akshare as ak

def update_model1(oil):
    baltic = pd.read_excel("data/raw/baltic_indices.xls")
    baltic["Date"] = pd.to_datetime(baltic["Date"])
    baltic = baltic.rename(columns={"Date": "date"})
    baltic = baltic.sort_values("date").reset_index(drop=True)
    baltic = baltic.drop(columns=["CTI"])

    model1 = baltic.merge(oil, on="date", how="left")
    model1["DCOILBRENTEU"] = model1["DCOILBRENTEU"].ffill()
    return model1

def update_oil():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
    df = pd.read_csv(url)
    df.columns = ["date", "DCOILBRENTEU"]
    df["date"] = pd.to_datetime(df["date"])
    df["DCOILBRENTEU"] = pd.to_numeric(df["DCOILBRENTEU"], errors="coerce")
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    df["DCOILBRENTEU"] = df["DCOILBRENTEU"].ffill().bfill()
    return df

ROUTES = {
    "shanghai_rotterdam": "shanghai-rotterdam",
    "rotterdam_shanghai": "rotterdam-shanghai",
    "shanghai_los_angeles": "shanghai-los angeles",
    "los_angeles_shanghai": "los angeles-shanghai",
    "shanghai_genoa": "shanghai-genoa",
    "new_york_rotterdam": "new york-rotterdam",
    "rotterdam_new_york": "rotterdam-new york",
}
def clean_route(symbol):
    r = ak.drewry_wci_index(symbol=symbol)
    r = r.iloc[:, :2]
    r.columns = ["date", "wci"]
    r["date"] = pd.to_datetime(r["date"])

    # A: give rows with no date a date (the date above + 7 days)
    dates = r["date"].tolist()
    for i in range(1, len(dates)):
        if pd.isna(dates[i]):
            dates[i] = dates[i - 1] + pd.Timedelta(days=7)
    r["date"] = dates

    # B: move every date to Thursday
    r["date"] = r["date"] + pd.to_timedelta(3 - r["date"].dt.weekday, unit="D")
    r = r.drop_duplicates(subset="date", keep="last")

    # C: add the missing weeks and fill their prices
    r = r.set_index("date")
    all_weeks = pd.date_range(r.index.min(), r.index.max(), freq="7D")
    r = r.reindex(all_weeks)
    r["wci"] = r["wci"].interpolate()
    return r["wci"]


def update_wci(oil):
    prices = {}
    for name, symbol in ROUTES.items():
        prices[name] = clean_route(symbol)

    table = pd.DataFrame(prices)
    table.index.name = "date"
    table = table.reset_index()
    table = table.merge(oil, on="date", how="left")
    table["DCOILBRENTEU"] = table["DCOILBRENTEU"].ffill()
    return table
if __name__ == "__main__":
    oil = update_oil()
    wci = update_wci(oil)

    wci.to_csv(r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\processed\model2_routes_data.csv", index=False)

    print("oil last date:", oil["date"].max())
    print("wci last date:", wci["date"].max(), "| rows:", len(wci))
