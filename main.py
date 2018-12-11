import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn import preprocessing as pre
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import KBinsDiscretizer
import random


housing = pandas.read_csv('housing.csv')
normalization = True

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

if normalization:
    scaler = pre.RobustScaler()
    normalized_housing = scaler.fit_transform(housing)
    normalized_housing = pandas.DataFrame(normalized_housing, columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity_int'])

print('----')
missing = housing.total_bedrooms.isna()
print(sum(missing))

# Regression using the above
model = lm.LinearRegression()

if normalization:
    normalized_model = lm.LinearRegression()

# First, extract the data into arrays
y = housing.median_house_value.values.reshape(-1, 1)
X = housing.drop(columns=['median_house_value'], inplace=False).values
if normalization:
    normalized_y = normalized_housing.median_house_value.values.reshape(-1, 1)
    normalized_X = normalized_housing.drop(columns=['median_house_value'], inplace=False).values
# print(X.shape)
# print(y.shape)

# Pull out values into a holdout set
holdout = random.sample(range(0, 10640), 1000)  # tried with different sizes of the holdout dataset.
                                                # Did not make much difference.
X_holdout = X[holdout]
y_holdout = y[holdout]

Xt = np.delete(X, holdout, 0)
yt = np.delete(y, holdout, 0)

if normalization:
    normalized_X_holdout = normalized_X[holdout]
    normalized_y_holdout = normalized_y[holdout]

    normalized_Xt = np.delete(normalized_X, holdout, 0)
    normalized_yt = np.delete(normalized_y, holdout, 0)

print(Xt.shape)
print(yt.shape)

# Have to shuffle the data because it is grouped.
kf = KFold(n_splits=5, shuffle=True)

best_linear_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
best_linear = None
best_poly_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
best_poly = None
best_binner_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
best_binner = None

if normalization:
    normalized_best_linear_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
    normalized_best_linear = None
    normalized_best_poly_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
    normalized_best_poly = None
    normalized_best_binner_model = [0, 0, 0, 0]  # x index, y index, score, fold no.
    normalized_best_binner = None

n = 1  # Fold counter

for train_index, test_index in kf.split(Xt):
    print('\nFold #{n}'.format(n=n))
    X_train, X_test = Xt[train_index], Xt[test_index]
    y_train, y_test = yt[train_index], yt[test_index]

    model.fit(X_train, y_train)

    accuracy = model.score(X_train, y_train)

    if normalization:
        normalized_X_train, normalized_X_test = normalized_Xt[train_index], normalized_Xt[test_index]
        normalized_y_train, normalized_y_test = normalized_yt[train_index], normalized_yt[test_index]

        normalized_model.fit(normalized_X_train, normalized_y_train)

        normalized_accuracy = normalized_model.score(normalized_X_train, normalized_y_train)

    if accuracy > best_linear_model[2]:
        print("NEW LINEAR BEST: FOLD: " + n.__str__() + " ACCURACY: " + accuracy.__str__())
        best_linear_model[0] = X_train
        best_linear_model[1] = y_train
        best_linear_model[2] = accuracy

        best_linear_model[3] = n

    if normalization:
        if normalized_accuracy > normalized_best_linear_model[2]:
            print("NEW NORMALIZED LINEAR BEST: FOLD: " + n.__str__() + " ACCURACY: " + normalized_accuracy.__str__())
            normalized_best_linear_model[0] = normalized_X_train
            normalized_best_linear_model[1] = normalized_y_train
            normalized_best_linear_model[2] = normalized_accuracy

            normalized_best_linear_model[3] = n

    for deg in range(1, 4):
        poly = PolynomialFeatures(degree=deg)
        X_train_poly, X_test_poly = poly.fit_transform(X_train), poly.fit_transform(X_test)
        model.fit(X_train_poly, y_train)

        accuracy = model.score(X_test_poly, y_test)

        if normalization:
            normalized_X_train_poly, normalized_X_test_poly = poly.fit_transform(normalized_X_train), poly.fit_transform(normalized_X_test)
            normalized_model.fit(normalized_X_train_poly, normalized_y_train)

            normalized_accuracy = normalized_model.score(normalized_X_test_poly, normalized_y_test)

        if accuracy > best_poly_model[2]:
            print("NEW POLYNOMIAL BEST: FOLD: " + n.__str__() + " DEGREE: " + deg.__str__() + " ACCURACY: " + accuracy.__str__())
            best_poly_model[0] = X_train_poly
            best_poly_model[1] = y_train
            best_poly_model[2] = accuracy
            best_poly_model[3] = n
            best_poly = poly

        if normalization:
            if normalized_accuracy > normalized_best_poly_model[2]:
                print("NEW NORMALIZED POLYNOMIAL BEST: FOLD: " + n.__str__() + " DEGREE: " + deg.__str__() + " ACCURACY: " + normalized_accuracy.__str__())
                normalized_best_poly_model[0] = normalized_X_train_poly
                normalized_best_poly_model[1] = normalized_y_train
                normalized_best_poly_model[2] = normalized_accuracy
                normalized_best_poly_model[3] = n
                normalized_best_poly = poly

    for n_bins in range(10, 70):
        binner = KBinsDiscretizer(n_bins=n_bins, strategy='uniform')
        X_train_binned, X_test_binned = binner.fit_transform(X_train), binner.fit_transform(X_test)
        model.fit(X_train_binned, y_train)

        accuracy = model.score(X_test_binned, y_test)

        if normalization:
            normalized_X_train_binned, normalized_X_test_binned = binner.fit_transform(normalized_X_train), binner.fit_transform(normalized_X_test)
            normalized_model.fit(normalized_X_train_binned, normalized_y_train)

            normalized_accuracy = normalized_model.score(normalized_X_test_binned, normalized_y_test)

        if accuracy > best_binner_model[2]:
            print("NEW BINNING BEST: FOLD: " + n.__str__() + " BIN NO.: " + n_bins.__str__() + " ACCURACY: " + accuracy.__str__())
            best_binner_model[0] = X_train_binned
            best_binner_model[1] = y_train
            best_binner_model[2] = accuracy
            best_binner_model[3] = n
            best_binner = binner

        if normalization:
            if normalized_accuracy > normalized_best_binner_model[2]:
                print("NEW NORMALIZED BINNING BEST: FOLD: " + n.__str__() + " BIN NO.: " + n_bins.__str__() + " ACCURACY: " + normalized_accuracy.__str__())
                normalized_best_binner_model[0] = normalized_X_train_binned
                normalized_best_binner_model[1] = normalized_y_train
                normalized_best_binner_model[2] = normalized_accuracy
                normalized_best_binner_model[3] = n
                normalized_best_binner = binner

    n += 1

# Evaluation of method
optimal_model = model.fit(best_linear_model[0], best_linear_model[1])
print('Accuracy on seen data LIN: ' + str(best_linear_model[2] * 100) + " on fold: " + str(best_linear_model[3]))
print('Accuracy on unseen data LIN: ' + str(optimal_model.score(X_holdout, y_holdout) * 100))
normalized_optimal_model = normalized_model.fit(normalized_best_linear_model[0], normalized_best_linear_model[1])
print('Accuracy on seen data NORMALIZED LIN: ' + str(normalized_best_linear_model[2] * 100) + " on fold: " + str(normalized_best_linear_model[3]))
print('Accuracy on unseen data NORMALIZED LIN: ' + str(normalized_optimal_model.score(normalized_X_holdout, normalized_y_holdout) * 100))

optimal_model = model.fit(best_poly_model[0], best_poly_model[1])
print('Accuracy on seen data POLY: ' + str(best_poly_model[2] * 100) + " on fold: " + str(best_poly_model[3]))
print('Accuracy on unseen data POLY: ' + str(optimal_model.score(best_poly.fit_transform(X_holdout), y_holdout) * 100))
print('Polynomial degree: ' + str(best_poly.degree))
normalized_optimal_model = normalized_model.fit(normalized_best_poly_model[0], normalized_best_poly_model[1])
print('Accuracy on seen data NORMALIZED POLY: ' + str(normalized_best_poly_model[2] * 100) + " on fold: " + str(normalized_best_poly_model[3]))
print('Accuracy on unseen data NORMALIZED POLY: ' + str(normalized_optimal_model.score(normalized_best_poly.fit_transform(normalized_X_holdout), normalized_y_holdout) * 100))
print('Polynomial degree, NORMALIZED: ' + str(normalized_best_poly.degree))

optimal_model = model.fit(best_binner_model[0], best_binner_model[1])
print('Accuracy on seen data BIN: ' + str(best_binner_model[2] * 100) + " on fold: " + str(best_binner_model[3]))
print('Accuracy on unseen data BIN: ' + str(optimal_model.score(best_binner.fit_transform(X_holdout), y_holdout) * 100))
print('No of bins: ' + str(best_binner.n_bins))
normalized_optimal_model = normalized_model.fit(normalized_best_binner_model[0], normalized_best_binner_model[1])
print('Accuracy on seen data NORMALIZED BIN: ' + str(normalized_best_binner_model[2] * 100) + " on fold: " + str(normalized_best_binner_model[3]))
print('Accuracy on unseen data NORMALIZED BIN: ' + str(normalized_optimal_model.score(normalized_best_binner.fit_transform(normalized_X_holdout), normalized_y_holdout) * 100))
print('No of bins, NOMRALIZED: ' + str(normalized_best_binner.n_bins))

# 2. replace with values from nearest neighbour
# kn?.predict(predicted_column.reshape(-1,1))

# 3. use regression with the values in the total_rooms column as prior knowledge
