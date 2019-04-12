# Data Preprocessing


# 1. Importing the dataset
dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3] # use a subset of the dataset

# 2. Taking care of missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

# 3. Encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3)) # not ranked
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))

# 4. Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools) # split data into train set and test set
set.seed(123) # ensure the same output as the instructor
split = sample.split(dataset$Purchased, SplitRatio = 0.8) # 80% of the dataset is used as the train set
training_set = subset(dataset, split == TRUE) # get the training set
test_set = subset(dataset, split == FALSE) # get the test set

# 5. Feature Scaling
# take the numeric values from the 2nd and 3rd column
training_set[, 2:3] = scale(training_set[, 2:3]) 
test_set[, 2:3] = scale(test_set[, 2:3]) 

