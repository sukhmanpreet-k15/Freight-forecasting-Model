import akshare as ak

wci = ak.drewry_wci_index()
wci.to_csv("data/raw/drewry_wci.csv", index=False)
print(wci.head())