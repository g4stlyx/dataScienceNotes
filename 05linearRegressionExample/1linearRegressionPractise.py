import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
"""
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')
"""
# plt.show()

# we spot some patterns, but not linear ones. so we should first transform one or more variables
# log transformation is the most common solution

log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price

# plot them with the log_price
"""
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')
"""
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
# if we have n categories for a feature, we have to create n-1 dummies

data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

print(data_with_dummies)

cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']

data_preprocessed = data_with_dummies[cols]
print(data_preprocessed.head())

#! linear regression model
#* inputs and targets
targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis=1)

#* standardization
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)    # we normally shouldnt standardize dummy variables

#* train test split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

#* creating the regression

reg= LinearRegression()
reg.fit(x_train, y_train)

y_hat = reg.predict(x_train)
# plt.scatter(y_train, y_hat)
# plt.xlabel('Targets (y_train)', size=18)
# plt.ylabel('Predictions (y_hat)', size=18)
# plt.xlim(6,13)
# plt.ylim(6,13)
# plt.show()

# sns.histplot(y_train - y_hat, kde=True, stat="density", linewidth=0)
# plt.title("Residuals PDF", size=18) # standard normal distribution with a long tail on left. mean = 0
# plt.show()

reg.score(x_train,y_train) # 0.74, meaning our model is explaining 74% of variablity of the data

reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
print("*****************")
print(reg_summary)

# positive weight shows that as a feature increases in value, so do the log_price and 'Price'. kısaca weight pozitif = o özellikle tahmin edilen özellik arasında doğru orantı var.
# negative weight = ters orantı

# for dummy variables, we had n-1 dummies. that -1 is caused by the BENCHMARK VALUE, in our case the brand 'Audi'
# so if the weight of a dummy variable is positive, its more expansive than 'Audi'.

#! Testing

y_hat_test = reg.predict(x_test)

plt.scatter(y_test, y_hat_test, alpha=0.2) # the more saturated the color, the higher the concentration
plt.xlabel('Targets (y_test)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions']) # data frame performance

y_test = y_test.reset_index(drop=True)
df_pf['Target'] = np.exp(y_test)
df_pf['Residuals'] = df_pf['Target'] - df_pf['Predictions']
df_pf['Difference%'] = np.absolute(df_pf['Residuals']/df_pf['Target']*100)

print("*********")
print(df_pf.describe())
print("**************")
print(df_pf.sort_values(by=['Difference%'])) # most of the predictions are pretty good.