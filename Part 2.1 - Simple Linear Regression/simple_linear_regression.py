# Simple Linear Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values # the 2nd column

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling (do not need this for simple linear regression, the library will take care of it)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set (learn the correlation between experience and salary)
from sklearn.linear_model import LinearRegression # linear regression class
regressor = LinearRegression() # regressor object
regressor.fit(X_train, y_train) # fit method used to fit the regressor object to the training set

# Predicting the Test set results (test how well this model predicts new observations)
y_pred = regressor.predict(X_test) # predict method used to get a vector of prediction of DV i.e. predicted salary for all the observations in the test set

# Visualising the Training set results (the model is tested using the same dataset)
plt.scatter(X_train, y_train, color = 'red') # scatter plot of the training set (actual values)
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # simple linear regression model trained using the training set (predicted values, pattern learned)
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience') 
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results (the model is tested using a different dataset)
plt.scatter(X_test, y_test, color = 'red') # scatter plot of the test set (actual values)
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # simple linear regression model trained using the training set (predicted values, pattern learned)
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

