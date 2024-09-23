import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.linear_model import LinearRegression

data = pd.read_csv("2.0simpleLinearRegression.csv")
print(data.head())

x = data['SAT'] # feature
y = data['GPA'] # target

# reshaping x to create a 2D array for the linear regression
x_matrix = x.values.reshape(-1,1) # 1D to 2D

# regression itself
reg = LinearRegression()
reg.fit(x_matrix,y)

#standardization: the process of subtracting the mean and dividing by the standard deviation, a type of normalization

reg.score(x_matrix,y) # returns the r-squared of a linear regression
reg.coef_ # the array of coefficients
reg.intercept_ # intercept, a simple linear regression always has a single intercept

# making predictions

# reg.predict(1748) # returns an array of predictions

new_data = pd.DataFrame(data=[1740,1760],columns=['SAT'])
new_data['Predicted_GPA'] = reg.predict(new_data)
print(new_data)


# ploting our regression -with the same code-
plt.scatter(x,y)
yhat = reg.coef_*x_matrix + reg.intercept_
fig = plt.plot(x, yhat, lw=4, c='orange', label="regression line")
plt.xlabel("SAT", fontsize=20)
plt.ylabel("GPA", fontsize=20)
plt.show()