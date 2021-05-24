import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
df1=pd.read_csv("covid_ready.csv")
X=df1[df1.columns[0:33]].to_numpy()
y=df1['risk_factor'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=4)
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_test,y_test)
reg.coef_
preds=reg.predict(X_test)
X_test.shape
print("r2_score:",r2_score(y_test, preds))
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print("mean square error is",mse)
print("mean absolute error is",mae)
pickle.dump(reg,open('model.pkl','wb'))

