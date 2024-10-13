import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# apply a fix to the statsmodels library
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

raw_data = pd.read_csv("3.0binaryPredictors.csv")

data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1,'No':0}) # if user is admitted, make its admitted value 1. 0 if not admitted.
data['Gender'] = data['Gender'].map({'Female':1,'Male':0})

y = data['Admitted']
x1 = data[['SAT','Gender']]

x= sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
print(results_log.summary())

# given the same SAT score, a female has 7 times higher odds to get admitted (for the university/degree having this dataset)