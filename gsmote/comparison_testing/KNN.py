
# Importing the libraries
import sys
import numpy as np
from gsmote import GSMOTE
from gsmote import preprocessing as pp
from gsmote.comparison_testing import Evaluator
sys.path.append('../../')

data_filename = "../../data/adultmini.csv".replace('\\', '/')
X,y = pp.preProcess(data_filename)

from sklearn.model_selection import train_test_split
X_t, X_test, y_t, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

X_train,y_train = GSMOTE.OverSample(X_t,y_t)
# X_train,y_train = X_t,y_t
# Fitting Simple Linear Regression to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=2, metric='minkowski',p=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test).astype(int)
print(y_pred)

#find confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.astype(int), y_pred)
print(cm)

Evaluator.evaluate(y_test, y_pred)
