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

#! regression

x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()

print(results_log.summary())

#! summary info

"""
method: MLE, MLE= Max. Likelihood Estimation= tries to maximize the likelihood function = maximizing the probability that our model is correct.
log-likelihood: almost always negative. the bigger it is, the better.
ll-null (log likelihood-null): 
llr(log likelihood ratio): measures if our model is statistically different from ll-null.
pseudo r-squared: 

"""