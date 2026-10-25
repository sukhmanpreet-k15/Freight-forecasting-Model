import akshare as ak

folder = r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\raw"

routes = ["shanghai-rotterdam", "rotterdam-shanghai", "shanghai-los angeles",
          "los angeles-shanghai", "shanghai-genoa", "new york-rotterdam", "rotterdam-new york"]

for route in routes:
    data = ak.drewry_wci_index(symbol=route)
    data.to_csv(folder + rf"\drewry_{route.replace(' ', '_')}.csv", index=False)
    print(route, "saved, rows:", len(data))