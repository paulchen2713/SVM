"""
Created on Thur May 20 18:20:52 2021

@author: me
@file: svm_sklearn.py
"""

import numpy as np
from sklearn import  preprocessing, neighbors, svm
# cross_validation is deprecated since version 0.18. 
# from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import pandas as pd

# the 11 labels should only be splited by "," do not add any "space" between them,
# otherwise we would have a KeyError: "['class'] not found in axis"
df = pd.read_csv('D:/BeginnerPythonProjects/Support_Vector_Machine_Python/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True) # replace missing data '?' with -99999, simply make them a outlier
df.drop(['id'], 1, inplace=True)      # 

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# originally we use cross_validatation.train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[1,0,0,0,1,1,0,1,0]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)

# [1,0,0,0,1,1,0,1,0]
# 0.6571428571428571
# [2]

