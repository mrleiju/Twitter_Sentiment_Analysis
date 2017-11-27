#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:57:20 2017

@author: chloezeng
"""

from nltk.classify.util import accuracy
from nltk.metrics.scores import precision as precision
from nltk.metrics.scores import recall as recall
import collections

class Result:
    def __init__(self, train, val, clf):
        self.trainFeatures = train
        self.valFeatures = val
        
        #initiates trueSets and predSets
        trueSets = collections.defaultdict(set)
        predSets = collections.defaultdict(set)
        
        #puts correctly labeled sentences in trueSets and the predictively labeled version in predSets
        for i, (features, label) in enumerate(self.valFeatures):
            trueSets[label].add(i)
            predicted = clf.classify(features)
            predSets[predicted].add(i)
            
        self.accuracy = accuracy(clf, self.valFeatures)
        self.pos_precision = precision(trueSets['pos'], predSets['pos'])
        self.pos_recall = recall(trueSets['pos'], predSets['pos'])
        self.neg_precision = precision(trueSets['neg'], predSets['neg'])
        self.neg_recall = recall(trueSets['neg'], predSets['neg'])
        
    def print_result(self):
        print 'train on %d instances, test on %d instances' % (len(self.trainFeatures), len(self.valFeatures))
        print 'accuracy:', self.accuracy
        print 'pos precision:', self.pos_precision
        print 'pos recall:', self.pos_recall
        print 'neg precision:', self.neg_precision
        print 'neg recall:', self.neg_recall
        
        