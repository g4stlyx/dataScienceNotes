"""
*good models require multiple regressions, in order to address the higher complexity of problems.
*e.g basic regression analysis calculates the house price only considering the size, in multiple regression we considr size, location, year etc. for the house price.

! y = b0 + b1*x1 + b2*x2 +... + bk*xk

! after 3 dimensions, there is no visual way to represent the data
* so the aim is min SSE

! finding the optimal number of variables, using ADJUSTED R-SQUARED
* multiple regressions are always better than simple ones
* adjusted R-Squared penalizes excessive use of variables

! basis for comparing models
* same dependent variable(y) and same dataset required.

! testing significance of the model (F-Test)
* used for testing the overall significance of the model
    * h0: b1=b2=b3=..=bk=0              all betas are 0
    * h1: at least one bi != 0          at least one beta is not 0
* the lower the F-statistic, the closer to a non-significant model

! OLS Assumptions
    * linearity: linear regression, simplest one. y= b0 + b1x1 + .. + bkxk+ E
    * no endogeneity: 
        * omitted variable bias: you forget to include a relevant variable => biased estimates
    * normality and homoscedasticity: E ~ N(0, σ2)  , no errors on average
        * 0 mean and equal variance
    * no autocorrelation: no serial correlation, cannot be relaxed
        * change every day, same underlying asset.
        * popular in finance sector
        * durbin-watson in the summary is 2 = no autocorrelation
        * it's less than 1 or more than 3 = problem
    * no multicollinearity
        * 2 or more variables have high correlation = e.g. A can be represented by B
        * so there is no need for keeping both A and B. choose and remove 1 OR transform them into a variable c.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set_theme()

data = pd.read_csv("2.3Multiple_linear_regression.csv")
print(data)

#! our new model: GPA = b0 + b1*SAT + b2*Rand1,2,3

y = data['GPA']
x1 = data[['SAT','Rand 1,2,3']]

x= sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary()) #* adding "Rand 1,2,3" field made R^2 lower. but it worsens the explanotary power, but is also INSIGNIFICANT.
#* removing insignificant fields from the model is very important