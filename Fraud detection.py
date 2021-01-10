# IEEE-CIS Fraud Detection Kaggle competition
# September 2019

import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import time 

#%% Preprocessing
os.chdir('directory')

tic = time.time()

train_transaction = pd.read_csv('train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('sample_submission.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)

y_train = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity

# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()

del train, test

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))   

toc = time.time()
print("Time elapsed in preprocesing:", (toc-tic)/60, "minutes")

#%% Model- xgboost 
tic = time.time()

import xgboost as xgb
clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.01,
    subsample=0.9,
    colsample_bynode=0.9,
    missing=-999,
    random_state=2019,
    tree_method='hist'  
)        

clf.fit(X_train, y_train)

toc = time.time()
print("Time elapsed in fitting the model:", (toc-tic)/60, "minutes")


#%% Accuracy - Work in progress - I need to split train/test sets first
yhat = clf.predict(X_test)

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_hat))


#%% Submission
sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('ACCxgboost.csv')
