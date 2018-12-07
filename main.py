import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn import preprocessing as pre
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
import random
import matplotlib.pyplot as plt 


housing = pandas.read_csv('housing.csv')

# print(housing) #displays in Jupyter notebook

# ocean_proximity can have the following values 'ISLAND' 'NEAR_OCEAN' 'INLAND' '<1H OCEAN' 'NEAR BAY'
# print(housing['ocean_proximity'])

housing['1h_ocean'] = [1 if i=='<1H OCEAN' else 0 for i in housing.ocean_proximity.values]
housing['island'] = [1 if i=='ISLAND' else 0 for i in housing.ocean_proximity.values]
housing['inland'] = [1 if i=='INLAND' else 0 for i in housing.ocean_proximity.values]
housing['near_ocean'] = [1 if i=='NEAR OCEAN' else 0 for i in housing.ocean_proximity.values]
housing['near_bay'] = [1 if i=='NEAR BAY' else 0 for i in housing.ocean_proximity.values]
housing.drop(columns=['ocean_proximity'], inplace=True)

# print (housing)

# total_bedrooms has missing values.
# Approaches:
# 1. replace with column average
not_missing = housing.total_bedrooms.notna()
missing = housing.total_bedrooms.isna()
count = sum(not_missing)
vals = sum(housing.total_bedrooms.values[not_missing])
avg = vals/count
print(avg)
print('----')

print(len(housing.total_bedrooms))

for index in range(0, len(missing)):
    if missing[index]:
        housing.total_bedrooms[index] = avg

print('----')
missing = housing.total_bedrooms.isna()
print(sum(missing))

# Regression using the above
model = lm.LinearRegression()

# First, extract the data into arrays
y = housing.median_house_value.values.reshape(-1, 1)
X = housing.drop(columns=['median_house_value'], inplace=False).values
# print(X.shape)
# print(y.shape)

# Pull out values into a holdout set
holdout = random.sample(range(0, 10640), 1000)  # tried with different sizes of the holdout dataset.
                                                # Did not make much difference.
X_holdout = X[holdout]
y_holdout = y[holdout]

Xt = np.delete(X, holdout, 0)
yt = np.delete(y, holdout, 0)

print(Xt.shape)
print(yt.shape)

# Have to shuffle the data because it is grouped.
kf = KFold(n_splits=5, shuffle=True)

n = 0
for train_index, test_index in kf.split(Xt):
    print('\nFold #{n}'.format(n=n))
    X_train, X_test = Xt[train_index], Xt[test_index]
    y_train, y_test = yt[train_index], yt[test_index]

    print('----')
    print('Linear Regression Results: ')
    model.fit(X_train, y_train)
    print('Training error: ' + str(model.score(X_train, y_train)))
    print('Testing error: ' + str(model.score(X_test, y_test)))

    for deg in range(2, 5):
        poly = PolynomialFeatures(degree=deg)
        print('----')
        print('Polynomial Regression Results with degree = {deg}: '.format(deg=deg))
        X_train_, X_test_ = poly.fit_transform(X_train), poly.fit_transform(X_test)
        model.fit(X_train_, y_train)
        print('Training error: ' + str(model.score(X_train_, y_train)))
        print('Testing error: ' + str(model.score(X_test_, y_test)))

    n += 1

# Evaluation of method


# 2. replace with values from nearest neighbour
# kn?.predict(predicted_column.reshape(-1,1))

# 3. use regression with the values in the total_rooms column as prior knowledge