#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: chloezeng
"""

from tqdm import tqdm_notebook as tqdm
import itertools
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures

class FeatureSelection:
    def __init__(self, pos, neg):
                
        self.posFeatures = list(itertools.chain(*pos))
        self.negFeatures = list(itertools.chain(*neg))
        
        #build frequency distibution of all words and then frequency distributions of words within positive and negative labels
        word_fd = FreqDist()
        cond_word_fd = ConditionalFreqDist()
        for word in tqdm(self.posFeatures):
                word_fd[word] += 1
                cond_word_fd['pos'][word] += 1
        for word in tqdm(self.negFeatures):
                word_fd[word] += 1
                cond_word_fd['neg'][word] += 1
            
        #finds the number of positive and negative words, as well as the total number of words
        pos_word_count = cond_word_fd['pos'].N()
        neg_word_count = cond_word_fd['neg'].N()
        total_word_count = pos_word_count + neg_word_count
        self.word_scores = {}
        for word, freq in word_fd.iteritems():
            pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
            neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
            self.word_scores[word] = pos_score + neg_score
            
    def feature_selection(self, number):
        best_vals = sorted(self.word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
        best_words = set([w for w, s in best_vals])
        
        return best_words
        
