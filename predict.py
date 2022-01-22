# Import libraries
import numpy as np
import pandas as pd
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Import scikit-learn packages
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Upload dataset
from google.colab import files
data = files.upload()

#Create Index as Day
dataset['Day'] = range(1, len(dataset) + 1)

#Plot data
dataset.plot("Day", ["Volume", "Close"], subplots=True)

#Define variables X and Y
X = dataset['Day'].values.reshape(-1,1)
y = dataset['Close'].values.reshape(-1,1)

#Splitting dataset
train, test = train_test_split(dataset, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Simple Linear Regression
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

#Create regressor and fit data train
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting
y_pred = regressor.predict(X)

# Visualizing the Linear Regression results
plt.plot(X,y, color='red')
plt.plot(X,y_pred, color='blue')
plt.title('BBCA Stock Prediction (Linear Regression)')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()

#Accuracy
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))

# Calculate model score for data train
train_score =  regressor.score(X_train, y_train)
print('data train - coefficient of determination:', train_score)

# Calculate model score for data test
test_score =  regressor.score(X_test, y_test)
print('data test - coefficient of determination:', test_score)

# Create polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)
poly_reg.fit(X)
X_poly = poly_reg.transform(X);

#Splitting dataset
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)

pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

#Predicting
y_poly_pred=pol_reg.predict(X_poly)

# Visualizing the Polymonial Regression results
plt.plot(X, y, color='red')
plt.plot(X, y_poly_pred, color='blue')
plt.title('BBCA Stock Prediction (Linear Regression)')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()

#Accuracy
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_poly_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y, y_poly_pred)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_poly_pred)))

# Calculate model score for data train
train_score =  pol_reg.score(X_train_poly, y_train)
print('data train - coefficient of determination:', train_score)

# Calculate model score for data test
test_score =  pol_reg.score(X_test_poly, y_test)
print('data test - coefficient of determination:', test_score)
