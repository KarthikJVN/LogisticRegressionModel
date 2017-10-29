from sklearn import preprocessing, svm, neighbors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm, utils
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


AngerScore = [80,77,70,68,64,60,50,46,40,35,30,25]
SecondHeartAttack = [1,1,0,1,0,1,1,0,1,0,0,1]
# ones = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

new = [[1,80],[1,77],[1,70],[1,68],[1,64],[1,60],[1,50],[1,46],[1,40],[1,35],[1,30],[1,25]]

features = np.reshape(new,(12,2))
Class = np.reshape(SecondHeartAttack,(12,1))

# print(features)

#features are X and Labels are Y
x = np.array(features)

y = np.array(Class)

print(x.shape,y.shape)


clf = LogisticRegression()
clf.fit(x, y)
print(clf.coef_)
print(clf.intercept_)

beta1 = clf.coef_[0][1]
beta0 = clf.intercept_[0]

def model(x):
    return 1 / (1 + np.exp(-(beta0+(beta1*x))))

print(model(55))

print(clf.predict_proba([1,55]))
