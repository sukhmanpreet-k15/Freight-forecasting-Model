import pandas_datareader.data as web
import datetime

start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2026, 9, 17)

oil = web.DataReader("DCOILBRENTEU", "fred", start, end)
oil.to_csv("data/raw/brent_oil.csv")
print(oil.head())