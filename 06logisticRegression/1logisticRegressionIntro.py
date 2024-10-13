#* NON-LINEAR, categorical: 0/1, yes/no, buy/dont buy etc.
#* predicts the probability of an event occuring. so the results are between 0 and 1.
# we did GPA predictions from STA scores using linear regression
# we will predict admissions(will they be accepted to the uni) from STA scores using logistic regression

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# apply a fix to the statsmodels library
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

raw_data = pd.read_csv("1.0admittance.csv")

data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1,'No':0}) # if user is admitted, make its admitted value 1. 0 if not admitted.

y = data['Admitted']
x1 = data['SAT']

#! plot of pure data without any regression

# plt.scatter(x1,y,color='C0')
# plt.xlabel('SAT',fontsize=20)
# plt.ylabel('Admitted', fontsize=20)
# plt.show()

#! plot with a logistic regression curve

x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()

def f(x,b0,b1):
    return np.array(np.exp(b0+x*b1) / (1 + np.exp(b0+x*b1)))

f_sorted = np.sort(f(x1,results_log.params[0],results_log.params[1]))
x_sorted = np.sort(np.array(x1))

plt.scatter(x1,y,color='C0')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('Admitted', fontsize = 20)
plt.plot(x_sorted,f_sorted,color='C8')
plt.show()