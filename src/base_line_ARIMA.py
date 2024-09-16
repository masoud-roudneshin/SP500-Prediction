import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas_datareader.data as web

sp500Data_Train = web.DataReader('SP500', 'fred', start= "2019", end= "2022")
sp500Data_Test = web.DataReader('SP500', 'fred', start= "2022", end= "2023")
df = web.DataReader('SP500', 'fred', start= "2019", end= "2023")
train = sp500Data_Train["SP500"]
test = sp500Data_Test["SP500"]

sequence_length = 60
model = ARIMA(train, order=(3,1,0))
model_fit = model.fit()
# Fit ARIMA model on training data
history = train
predictions = list()
# walk-forward validation

for t in range(len(test),0,-1):
    print(t)
    model = ARIMA(df[-t-sequence_length:-t], order=(3,1,0));
    model_fit = model.fit();
    output = model_fit.forecast();
    yhat = output.iloc[0]
    predictions.append(yhat)
