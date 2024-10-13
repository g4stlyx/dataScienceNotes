"""
! dealing with categorical data

*e.g. we will be considering attendence in the GPA example. 
*attendance column in data reflects if a student attended more than 75% of the lectures (Yes and No) => 1 and 0

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set_theme()

raw_data = pd.read_csv("3Dummies.csv")
data = raw_data.copy()

#* to transform Attendance Yes and No's to 1s and 0s
data['Attendance'] = data['Attendance'].map({'Yes':1, 'No':0})
print(data.describe()) #* the fact that the mean < 0.5 shows that there are more 0s than 1s. 0.46 mean means 46% of the class joins the classes

# old model: GPA = 0.275 + 0.0017 * SAT
#! regression, new model: GPA = 0.6439 + 0.0014 * SAT + 0.2226 * Dummy Attendance

y = data['GPA']
x1 = data[['SAT','Attendance']]

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary()) #* adjusted r-squared increased. 


#* if att=0, model: GPA = 0.6439 + 0.0014 * SAT
#* if att=1, model: GPA = 0.8665 + 0.0014 * SAT. so,
plt.scatter(data['SAT'],y, c=data['Attendance'], cmap='RdYlGn_r')     #* c= variable to color, cmap= red if attended, green if not
yhat_no = 0.6439 + 0.0014 * data['SAT']
yhat_yes = 0.8665 + 0.0014 * data['SAT']
yhat = 0.275 + 0.0017 * data['SAT']
fig = plt.plot(data['SAT'],yhat_no, lw=2, c="#006837", label = "regression line not attended")
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c="#a50026", label = "regression line attended")
fig = plt.plot(data['SAT'],yhat, lw=3, c="#4C72B0", label = "regression line general")
plt.xlabel("SAT", fontsize=20)
plt.ylabel("GPA", fontsize=20)
plt.show()