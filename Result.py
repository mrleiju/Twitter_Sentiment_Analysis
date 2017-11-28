#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:57:20 2017

@author: chloezeng
"""

from sklearn.metrics import classification_report, accuracy_score

class Result:
    def __init__(self, train, val, clf):
        self.trainFeatures = train
        self.valFeatures = val
        
        #initiates global_label and global_result
        self.global_label = []
        self.global_result = []
        
        #puts correctly labeled sentences in trueSets and the predictively labeled version in predSets
        for i, (features, label) in enumerate(self.valFeatures):
            self.global_label.append(label)
            self.global_result.append(clf.classify(features))

        
    def print_result(self):
        print('Accuracy: {}'.format(accuracy_score(self.global_label, self.global_result)))
        print classification_report(self.global_label, self.global_result)
        

        
        