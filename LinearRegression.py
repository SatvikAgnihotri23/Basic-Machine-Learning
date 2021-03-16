'''
This will be used to predict the final grades of students based on various variables in the dataset
Use linear regression when two variables are correlated.
It does not work well with randomized data
LinReg finds the best fit line in multi-dimensional space. 
'''
# importing

import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# declaring what separates data values - 'reference student-mat.csv' to see "csv" = comma separated values
# but this data set is weird
data = pd.read_csv("student-mat.csv", sep=";")
# sep = ";" means the seperator of data points is a semi colon.

# prints first 5 pieces of data and their values
# print(data.head())

# We don't want all of the attributes --> seperate data by:
data = data[["G1", "G2", "G3", "studytime", "failures", "internet"]]
# Note: try to chose attributes that use integer values. --> if not ->  you have to reformat data

# print data.head of newly selected attributes
# print(data.head())

predict = "G3"
# setting the goal
# G3 = output value (label) --> will remove predictor variable so there is no overlap

# returns new data frame that doesn't have "G3" in it
# this is declaring training data --> features + attributes
x = np.array(data.drop([predict], 1))
# y is declaring features of training data
y = np.array(data[predict])

# redefine here so that when the loop is closed,
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
'''
- taking all of attributes/labels/what we are trying to predict --> split them into four different arrays
- first array is x train, second is y train
- x train = a section of the 'x' array - line 35 // y train = a section of the 'y' array - line 37
- test data = testing section of the array
- order here matter a lot.
'''
'''
# what is the best score so far
best = 0
# run the model total of 30 times and chose best possible score
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # Develop Pickle file to represent data --> once its
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    # only save model if it is better than any previous one we have seen (67-68)
    if acc > best:
        best = acc
        # "pickle it" - creating a file for the model
        with open("studentmodel.pickle", "wb") as f:
            # save a pickle file in the directory so we can open and look
            # dump model linear into file f
            pickle.dump(linear, f)
'''

# read in the pickle file - open 'studentmodel.pickle' in 'rb' mode
pickle_in = open("studentmodel.pickle", "rb")
# load pickle into linear model --> load pickle into variable linear
linear = pickle.load(pickle_in)

# printing the various parts of y=mx+b equation --> coefficents and intercept
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

'''
Actually predicting students final grades
'''
# print out all predictions --> then find the input data for those predictions
predictions = linear.predict(x_test)

# Note: len = length
for x in range(len(predictions)):
    # print out predictions, then what the input was, then what the actual answer was
    print(predictions[x], x_test[x], y_test[x])

# can redefine p as any variable to see what has the strongest correlation in a regular graph


p = "studytime"
style.use("ggplot")

# create a scatter plot- data 1 = p = x axis -- data 2= G3 = y axis
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()


