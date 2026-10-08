import pandas as pd

folder = r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\raw"

routes = ["shanghai-rotterdam", "rotterdam-shanghai", "shanghai-los angeles",
          "los angeles-shanghai", "shanghai-genoa", "new york-rotterdam", "rotterdam-new york"]

for route in routes:
    r = pd.read_csv(folder + rf"\drewry_{route.replace(' ', '_')}.csv")
    r["date"] = pd.to_datetime(r["date"])
    print(route)
    print("  rows:", len(r))
    print("  rows with no date:", r["date"].isna().sum())
    print("  repeated dates:", r["date"].duplicated().sum())
    print("  empty prices:", r["wci"].isna().sum())