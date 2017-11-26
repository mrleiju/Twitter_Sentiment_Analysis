import nltk
import string
import re

class PreProcessor:
    def __init__(self):
        self.slangs = dict()
        
        f = open("internetSlangs.txt", 'r')
        lines = f.readlines()
        for line in lines:
            sen= line.split(",")
            if not self.slangs.has_key(sen[0]):
                self.slangs[sen[0]] = sen[-1]
    

    def process(self, words):
            #http & @ & #
            for word in words:    
                if re.search(r"[http]+://[^\s]*|@[^\s]*|#[^\s]*", word)!= None:
                    words.remove(word)
                    
            #slang words
            for i, word in enumerate(words):
                if self.slangs.has_key(word):
                    words.insert(i, self.slangs[word])
                    words.pop(i+1)
            words = re.findall(r"[\w]+|[:)-;=)(>o3Dx^\/*w8~_T|]+", ' '.join(words))
            
            #digit & punctuation
            for word in words: 
                if re.search(r"[0-9_-]+", word) != None or word in ['.',',',':',';','*','-',')','(','\\','/']:
                    words.remove(word)
    
            #lowercase
            return map(string.lower, words)


    