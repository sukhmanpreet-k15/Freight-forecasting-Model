import akshare as ak
import pandas as pd

routes = ["shanghai-rotterdam", "rotterdam-shanghai", "shanghai-los angeles",
          "los angeles-shanghai", "shanghai-genoa", "new york-rotterdam", "rotterdam-new york"]

prices = {}

for route in routes:
    r = ak.drewry_wci_index(symbol=route)
    r = r.drop_duplicates(subset="date", keep="last")
    prices[route] = r.set_index("date")["wci"]

table = pd.DataFrame(prices)
table.index = pd.to_datetime(table.index)
table.index.name = "date"

table.to_csv(r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\raw\drewry_routes.csv")

print(table.tail())
print(table.isna().sum())