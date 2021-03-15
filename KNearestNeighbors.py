import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# preprocessing allows you to convert non-numerical data into numerical data
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

# takes labels in Label Encoderand coverts them into apropriate integer values
# needs a list - currently we have a pandas data frame
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
cls = le.fit_transform(list(data["class"]))
safety = le.fit_transform(list(data["safety"]))

# below shows values asintegers after data frame has been converted to integers through code above
# print(buying)

predict = "class"

# zip creates
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# test size - the larger the more data. After 0.2, you start sacrificing preformance with less data to train on
# test size = percentage of data used as testing data represeted as a decimal
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.1)
'''
Note:
k must be odd to make sure there are no ties  i.e. if k=4 with 3 clusters, and 2/4 of the data points belong to one
cluster, and the other 2/4 belong to a second cluster, it's a tie
'''

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

# actual predicting
predicted = model.predict(x_test)
# for loop + loop through the test data + print out test data + prediction + what the actual value is
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    #printing distance between points

    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)
    
