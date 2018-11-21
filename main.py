import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn import preprocessing as pre
from sklearn.neighbors import KNeighborsRegressor
import random
import matplotlib.pyplot as plt 


housing = pandas.read_csv('housing.csv')

# print(housing) #displays in Jupyter notebook

# ocean_proximity can have the following values 'ISLAND' 'NEAR_OCEAN' 'INLAND' '<1H OCEAN' 'NEAR BAY'
# print(housing['ocean_proximity'])

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
known_columns = np.array([['total_rooms'], ['population'], ['households'], ['median_income'], ['median_house_value']])
predicted_column = np.array(['total_bedrooms'])

knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(known_columns.reshape(1,5), predicted_column)

knn.predict(predicted_column.reshape(-1,1))

# 3. use regression with the values in the total_rooms column as prior knowledge