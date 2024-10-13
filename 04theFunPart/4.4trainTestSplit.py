import numpy as np
from sklearn.model_selection import train_test_split

#! generating data to split

a = np.arange(1,101) # an array of numbers 1 to 100
b = np.arange(501,601)

#! splitting the data

# a_train, a_test = train_test_split(a, test_size=0.2, shuffle=False) # 20% of the data will be in a_test, and data wont be mixed: numbers 1 to 80 will be in a_train, 81 to 100 in a_test
a_train, a_test, b_train, b_test = train_test_split(a, test_size=0.2, random_state = 365) # giving a random_state provide us with the SAME RANDOM VALUES everytime program runs.