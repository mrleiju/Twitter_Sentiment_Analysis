from Preprocessor import PreProcessor
import csv
import re
from tqdm import tqdm
from sklearn.model_selection import KFold
from itertools import islice
import numpy as np
import math


class DataLoader:
    def __init__(self, k_value = 1):
        self.posFeatures = []
        self.negFeatures = []
        self.pp = PreProcessor()
        self.k = k_value
        if self.k >= 2:
            self.pos_kf = KFold(n_splits=k_value)
            self.neg_kf = KFold(n_splits=k_value)

        with open('Sentiment Analysis Dataset.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            print "start importing text data..."
            for row in tqdm(reader):
                if row['Sentiment'] == '1':
                    posWords = re.findall(r"[a-zA-Z_']+", row['SentimentText'].rstrip())
                    self.posFeatures.append([self.pp.process(posWords), 'pos'])
                elif row['Sentiment'] == '0':
                    negWords = re.findall(r"[a-zA-Z_']+", row['SentimentText'].rstrip())
                    self.negFeatures.append([self.pp.process(negWords), 'neg'])
            print "posFeatures size: " + str(len(self.posFeatures))
            print "negFeatures size: " + str(len(self.negFeatures))

        if self.k >= 2:
            self.pos_kf.get_n_splits(self.posFeatures)
            self.neg_kf.get_n_splits(self.negFeatures)

    def get_ratio_data(self, r):
        posCutoff = int(math.floor(len(self.posFeatures) * 3 / 4))
        negCutoff = int(math.floor(len(self.negFeatures) * 3 / 4))
        trainFeatures = self.posFeatures[:posCutoff] + self.negFeatures[:negCutoff]
        testFeatures = self.posFeatures[posCutoff:] + self.negFeatures[negCutoff:]
        return trainFeatures, testFeatures

    def get_k_data(self, i):
        if self.k == 1:
            print 'didn\'t assign valid k'
            return
        # input should start from 0
        if i >= self.k:
            print 'input value larger than split'
            return
        train_pos, val_pos = next(islice(self.pos_kf.split(self.posFeatures), i, i + 1))
        train_neg, val_neg = next(islice(self.neg_kf.split(self.negFeatures), i, i + 1))

        pos_train_data, pos_val_data = np.array(self.posFeatures)[train_pos], np.array(self.posFeatures)[val_pos]
        neg_train_data, neg_val_data = np.array(self.negFeatures)[train_neg], np.array(self.negFeatures)[val_neg]

        # print pos_train_data

        return pos_train_data.tolist() + neg_train_data.tolist(), \
               pos_val_data.tolist() + neg_val_data.tolist()
