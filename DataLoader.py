import csv
import re
import nltk
import string
import math
from tqdm import tqdm_notebook as tqdm
from random import shuffle
from Preprocessor import PreProcessor


class DataLoader:
    def __init__(self):
        # Tokenize words and keep emoticons
        self.posFeatures = []
        self.negFeatures = []
        self.posWords = []
        self.negWords = []
        self.pp = PreProcessor()
        
        # Import tweet data
        with open('Sentiment Analysis Dataset.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader):
                if row['Sentiment'] == '1':
                    # Divide http_@_# and keep emoticons
                    posWord = re.findall(r"[http]+://[^\s]*|@[^\s]*|#[^\s]*|[\w']+|[:)-;=)(>o3Dx^\/*w8~_T|]+", row["SentimentText"].rstrip())
                    # Preprocess
                    posWord = self.pp.process(posWord)
                    self.posFeatures.append(posWord)
                    self.posWords.append(' '.join(posWord))
                    
                elif row['Sentiment'] == '0':
                    negWord = re.findall(r"[http]+://[^\s]*|@[^\s]*|#[^\s]*|[\w']+|[:)-;=)(>o3Dx^\/*w8~_T|]+", row["SentimentText"].rstrip())
                    negWord = self.pp.process(negWord)
                    self.negFeatures.append(negWord)
                    self.negWords.append(' '.join(negWord))
                    
        shuffle(self.posFeatures)
        shuffle(self.negFeatures)
        shuffle(self.posWords)
        shuffle(self.negWords)

    # Return a whole sentence
    def get_data(self):
        return self.posWords, self.negWords
    
    # Return a unigram feature
    def get_uni_data(self):
        posFeatures = list(self.posFeatures)
        negFeatures = list(self.negFeatures)
        
        for i, words in tqdm(enumerate(self.posFeatures)):
            posFeatures[i] = words
        for i, words in tqdm(enumerate(self.negFeatures)):
            negFeatures[i] = words
        # Without label 'pos'&'neg'
        return posFeatures, negFeatures
    
    # Return a unigram&bigram feature
    def get_unibi_data(self):
        posFeatures = list(self.posFeatures)
        negFeatures = list(self.negFeatures)
        
        for i, word in tqdm(enumerate(self.posFeatures)):
            word_new = list(word)
            word_new.extend(self.bigram(word))
            posFeatures[i] = word_new
        for i, word in tqdm(enumerate(self.negFeatures)):
            word_new = list(word)
            word_new.extend(self.bigram(word))
            negFeatures[i] = word_new
        
        # Without label 'pos'&'neg'
        return posFeatures, negFeatures

    # POS tagging
    def get_posdata(self):
        posFeatures_pos = []
        negFeatures_pos = []
        for words in tqdm(self.posFeatures):
            posFeatures_pos.append([nltk.pos_tag(words[0]), 'pos'])
    
        for words in tqdm(self.negFeatures):
            negFeatures_pos.append([nltk.neg_tag(words[0]), 'neg'])

        return posFeatures_pos, negFeatures_pos

    # Get train&val data
    # Parameters posdata&negdata should be labeled
    def get_ratio_data(posdata, negdata, r):
        posCutoff = int(math.floor(len(posdata) * r))
        negCutoff = int(math.floor(len(negdata) * r))
        trainFeatures = posdata[:posCutoff] + negdata[:negCutoff]
        valFeatures = posdata[posCutoff:] + negdata[negCutoff:]
        
        return trainFeatures, valFeatures
    
    def bigram(self, words):
        return list(nltk.bigrams(words))

