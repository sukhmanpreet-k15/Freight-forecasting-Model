from cleaning_data import baltic, oil, wci
import pandas as pd

#Join Baltic + Oil =  Model 1 data
model1_data = pd.merge(baltic, oil, on="date", how="left")
model1_data["DCOILBRENTEU"] = model1_data["DCOILBRENTEU"].ffill()

print(model1_data.head())
print(model1_data.shape)
print(model1_data.isna().sum())

#Join WCI + Oil = Model 2 data
model2_data = pd.merge(wci, oil, on="date", how="left")
model2_data["DCOILBRENTEU"] = model2_data["DCOILBRENTEU"].ffill()

print(model2_data.head())
print(model2_data.shape)
print(model2_data.isna().sum())

#still checking for monotonic increasing dates in both datasets after joining
print(model1_data["date"].is_monotonic_increasing)
print(model2_data["date"].is_monotonic_increasing)

#processed data is saved in the processed folder for further analysis and model building
model1_data.to_csv("data/processed/model1_data.csv", index=False)
model2_data.to_csv("data/processed/model2_data.csv", index=False)