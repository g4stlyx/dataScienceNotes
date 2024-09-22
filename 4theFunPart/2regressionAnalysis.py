"""
!one of the most common methods of prediction,to be continued 

*linear approximation of a causal relationship between two or more variables

*process: get sample data -> design a model works for that sample data -> make predictions for the whole population

*formula: https://miro.medium.com/v2/resize:fit:1400/1*yTXAPLXaCYzuocaalZOt3g.png , y = b_0 + b_1*x_1

! correlation vs regression

* relationship          vs               one variable affects the other
* movement together     vs               cause and effect
* p(x,y) = p(y,x)       vs              one way
* single point          vs              line

!libraries

* numpy, pandas, scipy, statsmodels.api, matplotlib, seaborn, sklearn

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

sns.set_theme() # to use seaborn as the library used to plot
data = pd.read_csv('simpleLinearRegression.csv')
# print(data.describe())



#! gives info about the data
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
# print(results.summary())

#! interpreting the regression table
# we get b0 and b1 from the results.summary(), specifically from the constants part of it.

#! ploting
# y = b0 + b1*x1
y= data["GPA"]
x1= data["SAT"]

plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275 #! the function of the regression (y=b0+b1*x1)
fig = plt.plot(x1, yhat, lw=4, c='orange', label='regression line')
plt.xlabel("SAT",fontsize=20)
plt.ylabel("GPA",fontsize=20)
plt.show()

"""
sst = ssr + sse
sst- sum of squares total:       measures the total variablity of the dataset
ssr- sum of squares regression:  measures the EXPLAINED variablity by your line
sse- sum of squares error:       measures the UNEXPLAINED variablity by the regression
"""

"""
OLS: ordinary least squares: method that aims min SSE
"""

"""
R-Squared: R^2: 
        measures how powerful the regression is. 
        its between 0 and 1. 
        the optimal interval of R^2 changes field to field. (e.g in history 0.2 is nice, in physics 0.7)
        found using the equation: R^2= SSR/SST
"""