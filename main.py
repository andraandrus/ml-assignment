import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn import preprocessing as pre
from sklearn.neighbors import KNeighborsRegressor
import random
import matplotlib.pyplot as plt 


housing = pandas.read_csv('housing.csv')

print(housing) #displays in Jupyter notebook

# ocean_proximity can have the following values 'ISLAND' 'NEAR_OCEAN' 'INLAND' '<1H OCEAN' 'NEAR BAY'
print(housing['ocean_proximity'])

housing['1h_ocean'] = [1 if i=='<1H OCEAN' else 0 for i in housing.ocean_proximity.values]
housing['island'] = [1 if i=='ISLAND' else 0 for i in housing.ocean_proximity.values]
housing['inland'] = [1 if i=='INLAND' else 0 for i in housing.ocean_proximity.values]
housing['near_ocean'] = [1 if i=='NEAR OCEAN' else 0 for i in housing.ocean_proximity.values]
housing['near_bay'] = [1 if i=='NEAR BAY' else 0 for i in housing.ocean_proximity.values]
housing.drop(columns=['ocean_proximity'], inplace=True)

print (housing)

# total_bedrooms has missing values.
# Approaches:
# 1. replace with column average
not_missing = housing.total_bedrooms.notna()
missing = housing.total_bedrooms.isna()
count = sum(not_missing)
vals = sum(housing.total_bedrooms.values[not_missing])
avg = vals/count
print(avg)
print ('----')

print(len(housing.total_bedrooms))

for index in range(0, len(missing)):
    if missing[index]:
        housing.total_bedrooms[index] = avg


print ('----')
missing = housing.total_bedrooms.isna()
print(sum(missing))
# Regression using the above

# 2. replace with values from nearest neighbour
# kn?.predict(predicted_column.reshape(-1,1))

# 3. use regression with the values in the total_rooms column as prior knowledge