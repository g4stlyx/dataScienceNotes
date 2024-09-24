import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

def calculate_adj_r2(r2, n, p):
    return 1 - (1-r2) * (n-1) / (n-p-1)


data = pd.read_csv("2.3Multiple_linear_regression.csv")
print("********************************")
print(data.head())
print("********************************")
print(data.describe()) # gives info for its mean etc.

x = data[['SAT','Rand 1,2,3']]
y = data['GPA']

# regression

reg = LinearRegression()
reg.fit(x,y)

print("********************************")
print("coefficients: ", reg.coef_)
print("intercept: ", reg.intercept_)

# calculating the r-squared and adjusted r-squared
# no default function for adj. r2, so time for a little math. 
# Adj r2 = 1-(1-R2)*(n-1)/(n-p) , where n = number of observations and p = number of predictors

r2 = reg.score(x,y)
print("r-squared: ", r2) # 0.406

adj_r2 = calculate_adj_r2(r2,x.shape[0],x.shape[1])
print("adj. r2: ", adj_r2) # 0.392

#! since adj_r2 < r2, one or more variables dont increase the explanotary power of the regression. so we gotta find and deal with it.
#* we use "feature selection" to get rid of unneeded variables, to simplify the regression
#* we will use P-VALUES to achieve that. if a variables' p-value is over than 0.5, it is disgardable.

print("********************************")
print(f_regression(x,y)) # runs the regression for each variable, so we could find the one we dont need
# returns 2 arrays, first is for f-statistics and 2nd for p-values: (array([56.04804786,  0.17558437]), array([7.19951844e-11, 6.76291372e-01]))
p_values = f_regression(x,y)[1] 
print(p_values.round(3)) # round the numbers to 3 digits max. since the p-value of "Rand 1,2,3" is 0.676, we should disgard it


#! creating a summary table
reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features'])
reg_summary['Coefficients'] = reg.coef_
reg_summary['p-values'] = p_values.round(3)
print("********************************")
print(reg_summary)