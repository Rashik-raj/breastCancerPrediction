### Import Section ###
import pandas as pd
import numpy as np
# for graph
import matplotlib.pyplot as plt 
import seaborn as sns
# for models and trainings
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

### Data Loading ###
cancer = datasets.load_breast_cancer()
# 0 => malignant , 1 => benign
# ['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']
x = cancer.data
y = cancer.target

### Standadizing the datasets ###
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

### Linear Kernel ###
print('\n##### Linear Kernel #####\n')
### Without Standardization of datasets
print('*** Without Standardization of datasets ***\n')
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1, stratify = y)
clf = svm.SVC(kernel='linear')
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
print('accuracy: ', metrics.accuracy_score(y_test, y_predict))
print('precision: ', metrics.precision_score(y_test,y_predict))
print('recall: ', metrics.recall_score(y_test, y_predict))
### With Standardization of datasets
print('\n*** With Standardization of datasets ***\n')
x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,test_size = 0.2, random_state = 1, stratify = y)
clf = svm.SVC(kernel='linear')
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
print('accuracy: ', metrics.accuracy_score(y_test, y_predict))
print('precision: ', metrics.precision_score(y_test,y_predict))
print('recall: ', metrics.recall_score(y_test, y_predict))
print('confusion matrix')
mat = metrics.confusion_matrix(y_test, y_predict)
sns.heatmap(mat, square = True, annot = True, fmt = 'd', cbar = False,
           xticklabels = cancer.target_names,
           yticklabels = cancer.target_names)
plt.xlabel('predicted')
plt.ylabel('true label')
plt.show()

### Polynomial Kernel ###
print('\n##### Polynomial Kernel #####\n')
clf = svm.SVC(kernel='poly', degree = 1, gamma = 100)
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
print('accuracy: ', metrics.accuracy_score(y_test, y_predict))
print('precision: ', metrics.precision_score(y_test,y_predict))
print('recall: ', metrics.recall_score(y_test, y_predict))
print('confusion matrix')
mat = metrics.confusion_matrix(y_test, y_predict)
sns.heatmap(mat, square = True, annot = True, fmt = 'd', cbar = False,
           xticklabels = cancer.target_names,
           yticklabels = cancer.target_names)
plt.xlabel('predicted')
plt.ylabel('true label')
plt.show()

### Sigmoid Kernel ###
print('\n##### Sigmoid Kernel #####\n')
clf = svm.SVC(kernel='sigmoid', gamma = 200, C = 0.1)
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
print('accuracy: ', metrics.accuracy_score(y_test, y_predict))
print('precision: ', metrics.precision_score(y_test,y_predict))
print('recall: ', metrics.recall_score(y_test, y_predict))
print('confusion matrix')
mat = metrics.confusion_matrix(y_test, y_predict)
sns.heatmap(mat, square = True, annot = True, fmt = 'd', cbar = False,
           xticklabels = cancer.target_names,
           yticklabels = cancer.target_names)
plt.xlabel('predicted')
plt.ylabel('true label')
plt.show()