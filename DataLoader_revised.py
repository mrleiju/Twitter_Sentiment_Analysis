import csv
import re
import nltk
import string
import math
from tqdm import tqdm_notebook as tqdm

from Preprocessor import PreProcessor

# Tokenize words and keep emoticons
class DataLoader(self):
    def __init__(self):
        self.posFeatures = []
        self.negFeatures = []
        self.pp = PreProcessor()
        with open('Sentiment Analysis Dataset.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader):
                if row['Sentiment'] == '1':
                    posWord = re.findall(r"[http]+://[^\s]*|@[^\s]*|#[^\s]*|[\w']+|[:)-;=)(>o3Dx^\/*w8~_T|]+", row["SentimentText"].rstrip())
                    self.posFeatures.append([self.pp.process(posWord), 'pos'])
                elif row['Sentiment'] == '0':
                    negWord = re.findall(r"[http]+://[^\s]*|@[^\s]*|#[^\s]*|[\w']+|[:)-;=)(>o3Dx^\/*w8~_T|]+", row["SentimentText"].rstrip())
                    self.negFeatures.append([self.pp.process(negWord), 'neg'])

    def get_data(self):
        return self.posFeatures, self.negFeatures

    # POS tagging
    def get_posdata(self):
        posFeatures_pos = []
        negFeatures_pos = []
        for words in tqdm(self.posFeatures):
            posFeatures_pos.append([nltk.pos_tag(words[0]), 'pos'])
    
        for words in tqdm(self.negFeatures):
            negFeatures_pos.append([nltk.neg_tag(words[0]), 'neg'])

        return posFeatures_pos, negFeatures_pos

    def get_ratio_data(posdata, negdata, r):
        posCutoff = int(math.floor(len(posdata) * r))
        negCutoff = int(math.floor(len(negdata) * r))
        trainFeatures = posdata[:posCutoff] + negdata[:negCutoff]
        testFeatures = posdata[posCutoff:] + negdata[negCutoff:]
        return trainFeatures, testFeatures

