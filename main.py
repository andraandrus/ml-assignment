import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn import preprocessing as pre
from sklearn.neighbors import KNeighborsRegressor
import random
import matplotlib.pyplot as plt 
import matplotlib as mpl
import math
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics

def regression(housing):
    model = lm.LinearRegression()

    # First, extract the data into arrays
    y = housing.median_house_value.values.reshape(-1, 1)
    X = housing.drop(columns=['median_house_value'], inplace=False).values

    # Pull out values into a holdout set of unseen data
    holdout = random.sample(range(0, 20639), 5000) 
   
    X_unseen = X[holdout]
    y_unseen = y[holdout]
    
    Xt = np.delete(X, holdout, 0)
    yt = np.delete(y, holdout, 0)

    # Have to shuffle the data because it is grouped.
    kf = KFold(n_splits=5, shuffle=True)

    best_model = [0, 0, 0]  # x index, y index, score   

    for train_index, test_index in kf.split(Xt):
        X_train, X_test = Xt[train_index], Xt[test_index]
        y_train, y_test = yt[train_index], yt[test_index]
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        if accuracy > best_model[2]:
                best_model[2] = accuracy
                best_model[0] = X_train
                best_model[1] = y_train
        print('Training accuracy: ' + str(accuracy*100) + ' %')

    optimal_model = model.fit(best_model[0], best_model[1])
    predicted_values = optimal_model.predict(X_unseen)
    score = metrics.r2_score(y_unseen, predicted_values)

    print('R2 score on unseen data:' + str(score*100))
    return optimal_model

# Import housing data to Pandas DataFrame
housing = pandas.read_csv('housing.csv')

# ocean_proximity can have the following values 'ISLAND' 'NEAR_OCEAN' 'INLAND' '<1H OCEAN' 'NEAR BAY'

housing['1h_ocean'] = [1 if i=='<1H OCEAN' else 0 for i in housing.ocean_proximity.values]
housing['island'] = [1 if i=='ISLAND' else 0 for i in housing.ocean_proximity.values]
housing['inland'] = [1 if i=='INLAND' else 0 for i in housing.ocean_proximity.values]
housing['near_ocean'] = [1 if i=='NEAR OCEAN' else 0 for i in housing.ocean_proximity.values]
housing['near_bay'] = [1 if i=='NEAR BAY' else 0 for i in housing.ocean_proximity.values]
housing.drop(columns=['ocean_proximity'], inplace=True)

initialHousing = housing.copy()

# total_bedrooms has missing values.
# Approaches:
# 1. replace with column average

mean = housing['total_bedrooms'].mean() 
housing['total_bedrooms'].fillna(mean, inplace =True)

print("1. replace with column average")
print("------------------------------")
regression(housing)

# 2. replace with values from nearest neighbour
housing = initialHousing.copy()

housing_missing = housing[housing.total_bedrooms.isna()] # these we want to predict
housing_missing = housing_missing['total_rooms']

housing_not_missing = housing[housing.total_bedrooms.notna()]
X_train = housing_not_missing['total_rooms']                   # data  (X) -> longitude, latitude, total_rooms, population, households, median_house_value
y_train = housing_not_missing[['total_bedrooms']]              # label (y) -> total_bedrooms (column we want to predict)

X_shaped = []
for index in range(0, len(X_train)):
        X_shaped.append([X_train.index[index]])

regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(X_shaped, y_train)

predicted_values = []
for index in range(0, len(housing_missing)):
        values = housing_missing.iloc[index].tolist()
        y_pred = regressor.predict([[values]])
        predicted_values.append(y_pred[0][0])

housing.loc[housing.total_bedrooms.isna(), 'total_bedrooms'] = predicted_values

print("\n2. replace with values from nearest neighbour")
print("--------------------------------")
regression(housing)

# 3. use regression with the values in the total_rooms column as prior knowledge
housing = initialHousing.copy()

notna = housing.total_bedrooms.notna()
isna = housing.total_bedrooms.isna()

model = lm.LinearRegression()
model.fit(housing.total_rooms.values[notna].reshape(-1,1), housing.total_bedrooms.values[notna].reshape(-1,1))
model.score(housing.total_rooms.values[notna].reshape(-1,1), housing.total_bedrooms.values[notna].reshape(-1,1))

missing_bedrooms = model.predict(housing.total_rooms.values[isna].reshape(-1,1))
housing.total_bedrooms.loc[isna] = np.squeeze(missing_bedrooms)

print("\n3. use prior knowledge")
print("--------------------------------")
price_model = regression(housing)

# Drop logically irrelevant columns - neet to normalise beforehand:
# use regression_model.coef to figure out unimportant variables

housing = housing.drop(columns=['median_house_value'])

# for index, col_name in enumerate(housing.columns):
# 	print('column: ' + col_name + ', coef:' + str(price_model.coef_[0][index-1]))
        