import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("2.3Multiple_linear_regression.csv")
x = data[['SAT','Rand 1,2,3']]
y = data['GPA']

#! standardization
scaler = StandardScaler()

# scaler.fit(x)
# x_scaled = scaler.transform(x)
x_scaled = scaler.fit_transform(x)

#! feature selection through standardization of weights
# regression with scaled features

reg = LinearRegression()
reg.fit(x_scaled,y)

# creating a summary table to see the difference

reg_summary = pd.DataFrame([['Bias'], ['SAT'], ['Rand 1,2,3']], columns=['Features'])  # intercept = bias
reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]
# the bigger the weight, the bigger its impact on regression
# Rand 1,2,3's weight is -0.007, so it barely affects our regression
print(reg_summary)

#! making predictions with the standardized coefficients (weights)

new_data = pd.DataFrame(data=[[1700,2],[1800,1]], columns=['SAT','Rand 1,2,3'])
new_data_scaled = scaler.transform(new_data)

reg.predict(new_data_scaled)

#* what if we dont have Rand 1,2,3?         almost nothing changes since it is unneeded
reg_simple = LinearRegression()
x_simple_matrix = x_scaled[:0].reshape(-1,1)
reg_simple.fit(x_simple_matrix,y)
reg_simple.predict(new_data_scaled[:,0]).reshape(-1,1)
