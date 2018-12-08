import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn import preprocessing as pre
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import KBinsDiscretizer
import random
import matplotlib.pyplot as plt 
import warnings


housing = pandas.read_csv('housing.csv')

# print(housing) #displays in Jupyter notebook

# ocean_proximity can have the following values 'ISLAND' 'NEAR_OCEAN' 'INLAND' '<1H OCEAN' 'NEAR BAY'
# print(housing['ocean_proximity'])

# housing['1h_ocean'] = [1 if i=='<1H OCEAN' else 0 for i in housing.ocean_proximity.values]
# housing['island'] = [1 if i=='ISLAND' else 0 for i in housing.ocean_proximity.values]
# housing['inland'] = [1 if i=='INLAND' else 0 for i in housing.ocean_proximity.values]
# housing['near_ocean'] = [1 if i=='NEAR OCEAN' else 0 for i in housing.ocean_proximity.values]
# housing['near_bay'] = [1 if i=='NEAR BAY' else 0 for i in housing.ocean_proximity.values]
# housing.drop(columns=['ocean_proximity'], inplace=True)

ocean_proximity_type = {
    'ISLAND': 1,
    'NEAR OCEAN': 2,
    'INLAND': 3,
    '<1H OCEAN': 4,
    'NEAR BAY': 5
}
housing['ocean_proximity_int'] = [ocean_proximity_type[x] for x in housing.ocean_proximity.values]
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

# flag = True
# n = 0
# while flag:
#     flag = False
#     kf = KFold(n_splits=5, shuffle=True)
#     ksplit = kf.split(Xt)
#     for train_index, test_index in kf.split(Xt):
#         X_train, X_test = Xt[train_index], Xt[test_index]
#         y_train, y_test = yt[train_index], yt[test_index]
#         for i in range(13):
#             abc1 = X_train[:, i]
#             abc2 = X_test[:, i]
#             if X_train[:, i].min() == X_train[:, i].max() or X_test[:, i].min() == X_test[:, i].max():
#                 flag = True
#     print(n)
#     n += 1

# Have to shuffle the data because it is grouped.
# kf = KFold(n_splits=5, shuffle=True)

n = 1
kf = KFold(n_splits=5, shuffle=True)

best_poly_model = [0, 0, 0]  # x index, y index, score
best_poly = None
best_binner_model = [0, 0, 0]  # x index, y index, score
best_binner = None

for train_index, test_index in kf.split(Xt):
    print('\nFold #{n}'.format(n=n))
    X_train, X_test = Xt[train_index], Xt[test_index]
    y_train, y_test = yt[train_index], yt[test_index]

    # print('----')
    # print('Linear Regression Results: ')
    # model.fit(X_train, y_train)

    # accuracy = model.score(X_train, y_train)
    # if accuracy > best_model[2]:
    #     best_model[2] = accuracy
    #     best_model[0] = X_train
    #     best_model[1] = y_train
    # print('Training accuracy: ' + str(accuracy * 100) + ' %')
    # print('Testing accuracy: ' + str(model.score(X_test, y_test) * 100) + ' %')

    for deg in range(1, 6):
        poly = PolynomialFeatures(degree=deg)
        # print('----')
        # print('Polynomial Regression Results with degree = {deg}: '.format(deg=deg))
        X_train_poly, X_test_poly = poly.fit_transform(X_train), poly.fit_transform(X_test)
        model.fit(X_train_poly, y_train)

        accuracy = model.score(X_test_poly, y_test)
        if accuracy > best_poly_model[2]:
            best_poly_model[2] = accuracy
            best_poly_model[0] = X_train_poly
            best_poly_model[1] = y_train
            best_poly = poly

        # print('Training error: ' + str(model.score(X_train_poly, y_train)))
        # print('Testing error: ' + str(model.score(X_test_poly, y_test)))


    for n_bins in range(10, 11):
        binner = KBinsDiscretizer(n_bins=n_bins, strategy='uniform')
        # print('----')
        # print('Binning Regression Results with no. of bins = {n_bins}: '.format(n_bins=n_bins))
        X_train_binned = binner.fit_transform(X_train)
        X_test_binned = binner.fit_transform(X_test)
        model.fit(X_train_binned, y_train)

        accuracy = model.score(X_test_binned, y_test)
        if accuracy > best_binner_model[2]:
            best_binner_model[2] = accuracy
            best_binner_model[0] = X_train_binned
            best_binner_model[1] = y_train
            best_binner = binner

        # print('Training error: ' + str(model.score(X_train_binned, y_train)))
        # print('Testing error: ' + str(model.score(X_test_binned, y_test)))


    n += 1


# Evaluation of method
optimal_model = model.fit(best_poly_model[0], best_poly_model[1])
print('Accuracy on unseen data POLY:' + str(optimal_model.score(best_poly.fit_transform(X_holdout), y_holdout) * 100))
print('Polynomial: ' + str(best_poly.degree))
optimal_model = model.fit(best_binner_model[0], best_binner_model[1])
print('Accuracy on unseen data BINN:' + str(optimal_model.score(best_binner.fit_transform(X_holdout), y_holdout) * 100))
print('No of bins: ' + str(binner.n_bins))

# 2. replace with values from nearest neighbour
# kn?.predict(predicted_column.reshape(-1,1))

# 3. use regression with the values in the total_rooms column as prior knowledge