import pandas
import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn import preprocessing as pre
from sklearn.neighbors import KNeighborsRegressor
import random
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib as mpl
import math
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from scipy import stats 
from sklearn.impute import SimpleImputer

# Evaluation methods:

# Train and Test Sets. (Need to pass whole dataset)
def train_and_test(model, X, y):
	seed = 3
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=5000, random_state=seed)
	model.fit(X_train, Y_train)
	result = model.score(X_test, Y_test)
	print("Accuracy: %.3f%%" % (result*100.0))
	return result

# Leave One Out Cross Validation.
# def loocv(model, X, y):
# 	loocv = model_selection.LeaveOneOut()
# 	results = model_selection.cross_val_score(model, X, y, cv=loocv, scoring='r2')
# 	mean = results.mean()*100.0
# 	print("Accuracy: %.3f%% (%.3f%%)" % (mean, results.std()*100.0))
# 	return mean

# Repeated Random Test-Train Splits. (Need to pass whole dataset)
def rrtts(model, X, y):
	test_size = 0.33
	seed = 3
	kfold = model_selection.ShuffleSplit(n_splits=5, test_size=test_size, random_state=seed)
	results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='r2')
	print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
	return results.mean()*100.0

# K-fold Cross Validation.
def cross_validation(model, Xt, yt):
	seed = 3
	kfold = model_selection.KFold(n_splits=5, random_state=seed, shuffle=True)
	results = model_selection.cross_val_score(model, Xt, yt, cv=kfold, scoring='r2')
	r2 = results.mean()*100.0
	print("Cross Validation results:")
	print("R2 mean: %.3f%%, Standard deviation: %.3f%%" % (r2, results.std()*100.0))
	return r2

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

	# Perform cross validation for model
	r2 = cross_validation(model, Xt, yt)

	model_and_score = {'model': model, 'score': r2}
	return model_and_score

# Import housing data to Pandas DataFrame
housing = pandas.read_csv('housing.csv')

# Replace ocean_proximity categorical variable with one-hot vectors
housing['1h_ocean'] = [1 if i=='<1H OCEAN' else 0 for i in housing.ocean_proximity.values]
housing['island'] = [1 if i=='ISLAND' else 0 for i in housing.ocean_proximity.values]
housing['inland'] = [1 if i=='INLAND' else 0 for i in housing.ocean_proximity.values]
housing['near_ocean'] = [1 if i=='NEAR OCEAN' else 0 for i in housing.ocean_proximity.values]
housing['near_bay'] = [1 if i=='NEAR BAY' else 0 for i in housing.ocean_proximity.values]
housing.drop(columns=['ocean_proximity'], inplace=True)

initialHousing = housing.copy()

# total_bedrooms has missing values.
# Approaches:

# # 1. replace with column average
mean = housing['total_bedrooms'].mean() 
housing['total_bedrooms'].fillna(mean, inplace =True)

print("1. replace with column average")
print("------------------------------")

col_avg_model = regression(housing)
col_avg_model['name'] = 'replaced missing values with column average'

# 2. replace with most frequent

housing = initialHousing.copy()

imp = SimpleImputer(strategy="most_frequent")
housing_array = imp.fit_transform(housing)
columns = ['longitude', 'latitude',	'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value',
			'1h_ocean', 'island', 'inland', 'near_ocean', 'near_bay']

housing = pandas.DataFrame(data=housing_array[1:, 0:], columns=columns)  

print("\n2. replace with most frequent")
print("------------------------------")
most_frequent_model = regression(housing)
most_frequent_model['name'] = 'replaced missing values with most frequent value'

# 3. replace with values from nearest neighbour
housing = initialHousing.copy()

housing_missing = housing[housing.total_bedrooms.isna()] # these we want to predict
housing_missing = housing_missing[['longitude', 'latitude',	'housing_median_age', 'total_rooms',
		 	'population', 'households', 'median_income', 'median_house_value',
			'1h_ocean', 'island', 'inland', 'near_ocean', 'near_bay']]

housing_not_missing = housing[housing.total_bedrooms.notna()]

X_train = housing_not_missing[['longitude', 'latitude',	'housing_median_age', 'total_rooms',
		 	'population', 'households', 'median_income', 'median_house_value',
			'1h_ocean', 'island', 'inland', 'near_ocean', 'near_bay']]         # data  (X) -> longitude, latitude, total_rooms, population, households, median_house_value

y_train = housing_not_missing[['total_bedrooms']]            				   # label (y) -> total_bedrooms (column we want to predict)

regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(X_train, y_train)

predicted_values = []
for index in range(0, len(housing_missing)):
        values = housing_missing.iloc[index].tolist()
        y_pred = regressor.predict([values])
        predicted_values.append(y_pred[0][0])

housing.loc[housing.total_bedrooms.isna(), 'total_bedrooms'] = predicted_values

print("\n3. replace with values from nearest neighbour")
print("--------------------------------")
knn_model = regression(housing)
knn_model['name'] = 'replaced missing values with KNN'

# 4. use regression with the values in the total_rooms column as prior knowledge
housing = initialHousing.copy()

notna = housing.total_bedrooms.notna()
isna = housing.total_bedrooms.isna()

model = lm.LinearRegression()
model.fit(housing.total_rooms.values[notna].reshape(-1,1), housing.total_bedrooms.values[notna].reshape(-1,1))
model.score(housing.total_rooms.values[notna].reshape(-1,1), housing.total_bedrooms.values[notna].reshape(-1,1))

missing_bedrooms = model.predict(housing.total_rooms.values[isna].reshape(-1,1))
housing.total_bedrooms.loc[isna] = np.squeeze(missing_bedrooms)

print("\n4. use prior knowledge")
print("--------------------------------")
pk_model = regression(housing)
pk_model['name'] = 'replaced missing values using prior knowledge'

# Need to decide on the best of the three models => baseline
model_list = [col_avg_model, knn_model, pk_model, most_frequent_model]
best_score = max(col_avg_model['score'], knn_model['score'], pk_model['score'])
baseline_model = None
for model in model_list:
	if best_score == model['score']:
		baseline_model = model

# Remove outliers from data
housing = housing[(np.abs(stats.zscore(housing[['longitude', 'latitude',	'housing_median_age', 'total_rooms',
		 	'population', 'households', 'median_income', 'median_house_value']])) < 3).all(axis=1)]

# Test baseline model on unseen data
# Extract the data into arrays
y = housing.median_house_value.values.reshape(-1, 1)
X = housing.drop(columns=['median_house_value'], inplace=False).values

# Pull out values into a holdout set of unseen data
holdout = random.sample(range(0, len(housing)), 5000) 
X_unseen = X[holdout]
y_unseen = y[holdout]

Xt = np.delete(X, holdout, 0)
yt = np.delete(y, holdout, 0)

baseline_model['model'].fit(Xt, yt)
predicted_values = baseline_model['model'].predict(X_unseen)
score = metrics.r2_score(y_unseen, predicted_values)
print("R2 score on unseen data when " + baseline_model['name'] + ': ' + str(score*100))

# Normalise data

# Plot correlation matrix between variables
plt.figure(figsize=(12,10))
sns.heatmap(cbar=False,annot=True,data=housing.corr()*100,cmap='coolwarm')
plt.title('Corelation Matrix')
plt.show()

# Drop logically irrelevant columns
# use correlation matrix & regression_model.coef to figure out unimportant variables 
# for index, col_name in enumerate(housing.columns):
# 	print('column: ' + col_name + ', coef:' + str(price_model.coef_[0][index-1]))
        