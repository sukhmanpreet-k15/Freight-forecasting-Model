import pandas as pd

# load datasets
baltic = pd.read_excel(r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\raw\baltic_indices.xls")
oil = pd.read_csv(r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\raw\brent_oil.csv")
wci = pd.read_csv(r"C:\Users\sukhm\OneDrive\Desktop\pydev\Freight-forecasting-Model\data\raw\drewry_wci.csv")

# for printing column names and first few rows of each dataset
# print(baltic.columns.tolist())
# print(baltic.head())

# print(oil.columns.tolist())
# print(oil.head())

# print(wci.columns.tolist())
# print(wci.head())

# make all date columns the same name and proper date type
baltic["Date"] = pd.to_datetime(baltic["Date"])
baltic = baltic.rename(columns={"Date": "date"})

oil["DATE"] = pd.to_datetime(oil["DATE"])
oil = oil.rename(columns={"DATE": "date"})

wci["date"] = pd.to_datetime(wci["date"])

# datatypes of each dataset
print(baltic.dtypes)
print(oil.dtypes)
print(wci.dtypes)

#  Clean — check for issues
print("Missing values and duplicates in each dataset:")
print(baltic.isna().sum())
print(oil.isna().sum())
print(wci.isna().sum())

#duplicated values in date column
print("Duplicated values in date column:")
print(baltic["date"].duplicated().sum())
print(oil["date"].duplicated().sum())
print(wci["date"].duplicated().sum())

# Fix sort order
baltic = baltic.sort_values("date").reset_index(drop=True)
oil = oil.sort_values("date").reset_index(drop=True)
wci = wci.sort_values("date").reset_index(drop=True)