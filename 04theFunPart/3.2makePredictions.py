"""


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set_theme()

raw_data = pd.read_csv("3Dummies.csv")
data = raw_data.copy()

new_data = pd.DataFrame({'const':1,'SAT':[1700,1670],'Attendance':[0,1]})
new_data = new_data[['const','SAT','Attendance']]
new_data.rename(index={0:'Bob',1:'Alice'})

data['Attendance'] = data['Attendance'].map({'Yes':1, 'No':0})
print(data.describe())

y = data['GPA']
x1 = data[['SAT','Attendance']]
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())

predictions = results.predict(new_data)

predictionsdf = pd.DataFrame({'Predictions':predictions})
joined = new_data.join(predictionsdf) #! gives the SAT score predictions
joined.rename(index={0:'Bob',1:'Alice'})

print(new_data)
print(predictions)