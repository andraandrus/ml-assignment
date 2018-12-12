import pandas
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import preprocessing

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
avg = vals / count
print(avg)
print('----')

print(len(housing.total_bedrooms))

for index in range(0, len(missing)):
    if missing[index]:
        housing.total_bedrooms[index] = avg

print('----')
missing = housing.total_bedrooms.isna()
print(sum(missing))

robust_scaler = preprocessing.RobustScaler()
robust_housing = robust_scaler.fit_transform(housing)
robust_housing = pandas.DataFrame(robust_housing, columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                                                           'total_bedrooms',
                                                           'population', 'households', 'median_income',
                                                           'median_house_value', 'ocean_proximity_int'])

# Plot correlation matrix between variables using standard normalisation
plt.figure(figsize=(12, 10))
sns.heatmap(cbar=False, annot=True, data=robust_housing.corr() * 100, cmap='coolwarm')
plt.title('Standard Housing Correlation Matrix')
plt.show()

# Feature removal
# housing = housing.drop(columns=['latitude'])
# housing = housing.drop(columns=['longitude'])
# robust_housing = robust_housing.drop(columns=['latitude'])
# robust_housing = robust_housing.drop(columns=['longitude'])

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

robust_y = robust_housing.median_house_value.values.reshape(-1, 1)
robust_X = robust_housing.drop(columns=['median_house_value'], inplace=False).values
robust_X_holdout = robust_X[holdout]
robust_y_holdout = robust_y[holdout]
robust_Xt = np.delete(robust_X, holdout, 0)
robust_yt = np.delete(robust_y, holdout, 0)

# Have to shuffle the data because it is grouped.
kf = KFold(n_splits=5, shuffle=True)

best_linear_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
best_linear = None
best_poly_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
best_poly = None
best_binner_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
best_binner = None

robust_best_linear_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
robust_best_linear = None
robust_best_poly_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
robust_best_poly = None
robust_best_binner_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
robust_best_binner = None

n = 1  # Fold counter
splits = kf.split(Xt)

for train_index, test_index in splits:
    print('\nFold #{n}'.format(n=n))
    X_train, X_test = Xt[train_index], Xt[test_index]
    y_train, y_test = yt[train_index], yt[test_index]
    robust_X_train, robust_X_test = robust_Xt[train_index], robust_Xt[test_index]
    robust_y_train, robust_y_test = robust_yt[train_index], robust_yt[test_index]

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    model.fit(robust_X_train, robust_y_train)
    robust_accuracy = model.score(robust_X_test, robust_y_test)

    if accuracy > best_linear_model[2]:
        print("NEW LINEAR BEST: FOLD: " + n.__str__() + " ACCURACY: " + accuracy.__str__())
        best_linear_model[0] = X_train
        best_linear_model[1] = y_train
        best_linear_model[2] = accuracy
        best_linear_model[3] = n

    if robust_accuracy > robust_best_linear_model[2]:
        print("NEW NORMALIZED LINEAR BEST: FOLD: " + n.__str__() + " ACCURACY: " + robust_accuracy.__str__())
        robust_best_linear_model[0] = robust_X_train
        robust_best_linear_model[1] = robust_y_train
        robust_best_linear_model[2] = robust_accuracy
        robust_best_linear_model[3] = n

    for deg in range(1, 5):
        poly = PolynomialFeatures(degree=deg)
        X_train_poly, X_test_poly = poly.fit_transform(X_train), poly.fit_transform(X_test)
        robust_X_train_poly, robust_X_test_poly = poly.fit_transform(robust_X_train), poly.fit_transform(robust_X_test)

        model.fit(X_train_poly, y_train)
        accuracy = model.score(X_test_poly, y_test)

        model.fit(robust_X_train_poly, robust_y_train)
        robust_accuracy = model.score(robust_X_test_poly, robust_y_test)

        if accuracy > best_poly_model[2]:
            print(
                "NEW POLYNOMIAL BEST: FOLD: " + n.__str__() + " DEGREE: " + deg.__str__() + " ACCURACY: " + accuracy.__str__())
            best_poly_model[0] = X_train_poly
            best_poly_model[1] = y_train
            best_poly_model[2] = accuracy
            best_poly_model[3] = n
            best_poly = poly

        if robust_accuracy > robust_best_poly_model[2]:
            print(
                "NEW NORMALIZED POLYNOMIAL BEST: FOLD: " + n.__str__() + " DEGREE: " + deg.__str__() + " ACCURACY: " + robust_accuracy.__str__())
            robust_best_poly_model[0] = robust_X_train_poly
            robust_best_poly_model[1] = robust_y_train
            robust_best_poly_model[2] = robust_accuracy
            robust_best_poly_model[3] = n
            robust_best_poly = poly

    for n_bins in range(10, 40):
        binner = KBinsDiscretizer(n_bins=n_bins, strategy='uniform')
        X_train_binned, X_test_binned = binner.fit_transform(X_train), binner.fit_transform(X_test)
        robust_X_train_binned, robust_X_test_binned = binner.fit_transform(robust_X_train), binner.fit_transform(
            robust_X_test)

        model.fit(X_train_binned, y_train)
        accuracy = model.score(X_test_binned, y_test)

        model.fit(robust_X_train_binned, robust_y_train)
        robust_accuracy = model.score(robust_X_test_binned, robust_y_test)

        if accuracy > best_binner_model[2]:
            print(
                "NEW BINNING BEST: FOLD: " + n.__str__() + " BIN NO.: " + n_bins.__str__() + " ACCURACY: " + accuracy.__str__())
            best_binner_model[0] = X_train_binned
            best_binner_model[1] = y_train
            best_binner_model[2] = accuracy
            best_binner_model[3] = n
            best_binner = binner

        if robust_accuracy > robust_best_binner_model[2]:
            print(
                "NEW NORMALIZED BINNING BEST: FOLD: " + n.__str__() + " BIN NO.: " + n_bins.__str__() + " ACCURACY: " + robust_accuracy.__str__())
            robust_best_binner_model[0] = robust_X_train_binned
            robust_best_binner_model[1] = robust_y_train
            robust_best_binner_model[2] = robust_accuracy
            robust_best_binner_model[3] = n
            robust_best_binner = binner

    # svr_rbf = SVR(kernel='rbf', gamma=0.00001, C=1e6)
    # svr_rbf.fit(X_train, y_train)

    # print('----')
    # print('SVR results for RBF: ')
    # print('Training error: ' + str(svr_rbf.score(X_train, y_train)))
    # print('Testing error: ' + str(svr_rbf.score(X_test, y_test)))

    n += 1

# Evaluation of method
optimal_model = model.fit(best_linear_model[0], best_linear_model[1])
print('Accuracy on seen data LIN: ' + str(best_linear_model[2] * 100) + " on fold: " + str(best_linear_model[3]))
print('Accuracy on unseen data LIN: ' + str(optimal_model.score(X_holdout, y_holdout) * 100))
optimal_model = model.fit(robust_best_linear_model[0], robust_best_linear_model[1])
print('Accuracy on seen NORMALIZED data LIN: ' + str(robust_best_linear_model[2] * 100) + " on fold: " + str(
    robust_best_linear_model[3]))
print('Accuracy on unseen NORMALIZED data LIN: ' + str(optimal_model.score(robust_X_holdout, robust_y_holdout) * 100))
optimal_model = model.fit(best_poly_model[0], best_poly_model[1])
print('Accuracy on seen data POLY: ' + str(best_poly_model[2] * 100) + " on fold: " + str(best_poly_model[3]))
print('Accuracy on unseen data POLY: ' + str(optimal_model.score(best_poly.fit_transform(X_holdout), y_holdout) * 100))
print('Polynomial: ' + str(best_poly.degree))
optimal_model = model.fit(robust_best_poly_model[0], robust_best_poly_model[1])
print('Accuracy on seen NORMALIZED data POLY: ' + str(robust_best_poly_model[2] * 100) + " on fold: " + str(
    robust_best_poly_model[3]))
print('Accuracy on unseen NORMALIZED data POLY: ' + str(optimal_model.score(robust_best_poly.fit_transform(robust_X_holdout), robust_y_holdout) * 100))
print('NORMALIZED Polynomial: ' + str(robust_best_poly.degree))
optimal_model = model.fit(best_binner_model[0], best_binner_model[1])
print('Accuracy on seen data BIN: ' + str(best_binner_model[2] * 100) + " on fold: " + str(best_binner_model[3]))
print('Accuracy on unseen data BIN: ' + str(optimal_model.score(best_binner.fit_transform(X_holdout), y_holdout) * 100))
print('No of bins: ' + str(best_binner.n_bins))
optimal_model = model.fit(robust_best_binner_model[0], robust_best_binner_model[1])
print('Accuracy on NORMALIZED seen data BIN: ' + str(robust_best_binner_model[2] * 100) + " on fold: " + str(
    robust_best_binner_model[3]))
print('Accuracy on NORMALIZED unseen data BIN: ' + str(
    optimal_model.score(robust_best_binner.fit_transform(robust_X_holdout), robust_y_holdout) * 100))
print('NORMALIZED No of bins: ' + str(robust_best_binner.n_bins))

best_model = None
best_normalized_model = None

if np.max([best_binner_model[2], best_poly_model[2]]) == best_poly_model[2]:
    best_model = best_poly
    print("Best performing model is Polynomial.")
else:
    best_model = best_binner
    print("Best performing model is Binner.")
if np.max([robust_best_poly_model[2], robust_best_binner_model[2]]) == robust_best_poly_model[2]:
    best_model = robust_best_poly
    print("Best NORMALIZED performing model is Polynomial.")
else:
    best_model = robust_best_binner
    print("Best NORMALIZED performing model is Binner.")
