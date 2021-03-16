import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data  #[:100] --> means data up to 100 is training anything past it is test
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)


#print(x_train, y_train)
# index data such that it will output malignant or benign not 1 or 0
classes = ['malignant', 'benign']

# needs kernals to divide data on z axis ex.x1^2 + x2^2 --> x3
# soft margin - allow you to factor out a given number of outliers that violate the initially programmed hyperplane

'''
Kernal options
Linear - relatively quick. Can make alterations to C in '(kernel="linear", C=2 )' to add to a soft margin. C=0 is hard
margin, C=1 is slightly softer, C=2 is soft.
Note: C must be capital

Poly- takes a long time --> you can use degree in '(kernel="poly", degree=2)' to change the exopnent to something 
smaller ex.2 such that it takes less time
'''
clf = svm.SVC(kernel="linear", C=1 )

# compare accuracy with
#clf = KNeighborsClassifier(n_neighbors=5)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
