import pandas as pd

folder = r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\raw"

routes = ["shanghai-rotterdam", "rotterdam-shanghai", "shanghai-los angeles",
          "los angeles-shanghai", "shanghai-genoa", "new york-rotterdam", "rotterdam-new york"]


def clean_route(route):
    r = pd.read_csv(folder + rf"\drewry_{route.replace(' ', '_')}.csv")
    r["date"] = pd.to_datetime(r["date"])

    # A: give rows with no date a date (the date above + 7 days)
    dates = r["date"].tolist()
    for i in range(1, len(dates)):
        if pd.isna(dates[i]):
            dates[i] = dates[i - 1] + pd.Timedelta(days=7)
    r["date"] = dates

    # B: move every date to Thursday
    r["date"] = r["date"] + pd.to_timedelta(3 - r["date"].dt.weekday, unit="D")
    if r["date"].duplicated().any():
        print(route, "has repeated dates after the Thursday step")

    # C: add the missing weeks and fill their prices
    r = r.set_index("date")
    all_weeks = pd.date_range(r.index.min(), r.index.max(), freq="7D")
    r = r.reindex(all_weeks)
    r["wci"] = r["wci"].interpolate()
    return r["wci"]


prices = {}

for route in routes:
    prices[route] = clean_route(route)
    print(route, "| rows:", len(prices[route]), "| empty:", prices[route].isna().sum())

table = pd.DataFrame(prices)
table.index.name = "date"

print(table.tail())
print(table.shape)
print("empty values in the whole table:", table.isna().sum().sum())
#Step 4: add the oil price. Each week (Thursday) will get the oil price of that Thursday.
oil = pd.read_csv(folder + r"\brent_oil.csv")
oil["DATE"] = pd.to_datetime(oil["DATE"])
oil = oil.rename(columns={"DATE": "date"})
oil["DCOILBRENTEU"] = oil["DCOILBRENTEU"].ffill()

table = table.reset_index()
table = table.merge(oil, on="date", how="left")

table.columns = table.columns.str.replace(" ", "_").str.replace("-", "_")

print(table.tail(10))
print("empty oil values:", table["DCOILBRENTEU"].isna().sum())
print("different oil prices in the last 20 weeks:", table["DCOILBRENTEU"].tail(20).nunique())
print(table.shape)

table.to_csv(r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\processed\model2_routes_data.csv", index=False)