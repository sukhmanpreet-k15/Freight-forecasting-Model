# Freight-forecasting-Model
Freight Forecasting Model (SIH26006)
What does this project do?

Ships carry goods all over the world, and the price to hire a ship (the freight rate) changes almost every day. This project tries to predict the next freight rate for different types of ships.

The model looks at:

the recent past prices of the ship type (yesterday, 2 days ago, 3 days ago)
how much the price changed
the average of the last 3 days
the Brent oil price (because fuel cost affects shipping)

Then it predicts the freight rate for the next day.

Ship types covered:

Code	Ship type
HSI  Handysize
SI	 Supramax
PI	 Panamax
CI	 Capesize

One separate model is trained for each ship type, so there are 4 models in total.

Data used
Data	What it is
Baltic sub-indices (HSI, SI, PI, CI)	Daily freight price index for each ship type
Brent oil price (DCOILBRENTEU)	Daily price of Brent crude oil

Both are joined into one file: data/processed/model1_data.csv

Features used by the model

For each ship type, the model uses 7 inputs:

yesterday – ship index value of yesterday
lag2 – ship index value 2 days ago
lag3 – ship index value 3 days ago
change – change in ship index from the last day
roll3 – average of the last 3 days of the ship index
oil_yesterday – Brent oil price of yesterday
oil_roll3 – average Brent oil price of the last 3 days

Tools and libraries used:

Python – main language
pandas – to load and prepare the data
scikit-learn – Linear Regression model
pickle (.pkl files) – to save the trained models
VS Code – to write the code
Git and GitHub – to save and share the project