# Data Preprocessing


# 1. Importing the libraries
import numpy as np # mathematical tools
import matplotlib.pyplot as plt # plot chats 
import pandas as pd # import and manage datasets

# 2. Importing the dataset
dataset = pd.read_csv('Data.csv') # import dataset
X = dataset.iloc[:, :-1].values # take values from all the rows for all columns except the last column
y = dataset.iloc[:, 3].values # take values from all the rows for the last column

# 3. Taking care of missing data
from sklearn.preprocessing import Imputer # handle missing values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # replace missing value with the mean of the column
imputer = imputer.fit(X[:, 1:3]) # take values from all rows for the 2nd and 3rd column
X[:, 1:3] = imputer.transform(X[:, 1:3]) # change the missing value

# 4. Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # handle categorical values using dummy variables
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # encode values from all rows for the 1st column
onehotencoder = OneHotEncoder(categorical_features = [0]) # transform labels into dummys to avoid numbering signals ranking
X = onehotencoder.fit_transform(X).toarray()  
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y)

# 5. Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # 20% of the dataset is used as the test set; random_state = 0 to ensure the same output as the instructor

# 6. Feature Scaling (transform variables into the same scale e.g. standardisation or normalisation)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
