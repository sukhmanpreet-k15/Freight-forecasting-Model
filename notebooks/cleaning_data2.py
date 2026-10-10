import pandas as pd

folder = r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\raw"

r = pd.read_csv(folder + r"\drewry_shanghai-genoa.csv")
r["date"] = pd.to_datetime(r["date"])

# print("total rows:", len(r))

nodate = r[r["date"].isna()]
# print("rows with no date:", len(nodate))
# print("of these, rows that have a price:", nodate["wci"].notna().sum())
# print("row numbers:", nodate.index.tolist())

first = nodate.index[0]
print(r.iloc[max(first - 2, 0) : first + 3])


d = r.dropna(subset=["date"])
gap = d["date"].diff().dt.days

print(gap.value_counts())
print("weeks missing between dated rows:", (gap / 7 - 1).sum())
print("rows with no date:", r["date"].isna().sum())

dates = r["date"].tolist()

for i in range(1, len(dates)):
    if pd.isna(dates[i]):
        dates[i] = dates[i - 1] + pd.Timedelta(days=7)

r["date"] = dates

print("rows with no date now:", r["date"].isna().sum())
print(r["date"].diff().dt.days.value_counts())