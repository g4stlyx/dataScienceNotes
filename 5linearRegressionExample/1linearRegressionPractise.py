import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

#! the aim is predicting the price of a used car depending on its specifications

raw_data = pd.read_csv('0realLifeExample.csv')

#! PREPROCESSING - cleaning the data (part 1)

#* exploring the descriptive statistics of the variables

print("*****************")
print(raw_data.describe(include="all"))

#* dropping unneeded variables(like registration, since almost all cars are registered the variable is useless)

data= raw_data.drop(['Model'], axis=1)
# data = raw_data.drop('registration', axis=1)

#* dealing with missing values
#* rule of thumb: if you are removing <5% of the observations, you are free to just remove all that have missing values

print("*****************")
print(data.isnull().sum()) # gives how many null entries is there for each variable 

data_no_mv = data.dropna(axis=0)

print("*****************")
print(data_no_mv.describe(include="all"))

#* exploring the PDFs (prob. dist. func.s)

# sns.histplot(data_no_mv["Price"], kde=True, stat="density", linewidth=0)
# sns.histplot(data_no_mv["Mileage"], kde=True, stat="density", linewidth=0)
# plt.show()

#* dealing with outliers, the extreme points on both ends

q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']< q]

q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']< q]

data_3 = data_2[data_2['EngineV']<6.5]

q= data_3['Year'].quantile(0.01)    # to remove vintage cars like from 1970 etc
data_4 = data_3[data_3['Year']> q]

data_cleaned = data_4.reset_index(drop=True) # reseting index so index of the removed observations be gone

print("*****************")
print(data_cleaned.describe(include="all"))

# sns.histplot(data_1["Price"], kde=True, stat="density", linewidth=0)
# sns.histplot(data_2["Mileage"], kde=True, stat="density", linewidth=0)
# sns.histplot(data_3["EngineV"], kde=True, stat="density", linewidth=0)
# plt.show()

#! Checking the OLS assumptions (part 2)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')
# plt.show()

# we spot some patterns, but not linear ones. so we should first transform one or more variables
# log transformation is the most common solution

log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price

# plot them with the log_price

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')
# plt.show()

data_cleaned = data_cleaned.drop(['Price'], axis=1) # since log_price works better for linearity, we dont need Price anymore

#* Multicollinearity
# mileage and year must be correlated, we will use VIF to check this assumption

variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns

# vif = 1: no multicollinearity
# 1 < vif < 5: perfectly okay
# 7 < vif: unacceptable         # some say above 5, some say above 6, some say above 10 is unacceptable
print(vif)

data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)
print(data_no_multicollinearity)

#! create dummy variables (part 3)









