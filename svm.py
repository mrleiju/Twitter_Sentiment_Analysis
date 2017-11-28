#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 01:37:26 2017

@author: dakotashen
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier

from sklearn import svm
import pandas as pd

dataset = pd.read_csv('final_data.csv', error_bad_lines=False)
#print(dataset.head())

# prepare training and testing dataset
print len(dataset)
print("Number of Training Samples: 1048000")
num_test=len(dataset)-1048000
print "Number of Testing Samples: ",num_test
train_data, test_data = dataset[:1048000], dataset[1048000:]
X_train = train_data['Text'].values.astype('U')
y_train = train_data['Sentiment'].values.astype('U')
X_test = test_data['Text'].values.astype('U')
y_test = test_data['Sentiment'].values.astype('U')

def build_pipeline():
    text_clf = Pipeline([('vect', CountVectorizer(min_df=1, stop_words='english', binary=True)),
                         ('tfidf', TfidfTransformer()),
                         ('clf' ,SGDClassifier(l1_ratio=0, n_jobs=-1)),
                         ])
    return text_clf

text_clf = build_pipeline()
text_clf = text_clf.fit(X_train, y_train)

y_pred = text_clf.predict(X_test)
print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))