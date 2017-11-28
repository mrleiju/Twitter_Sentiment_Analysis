from nltk.classify import NaiveBayesClassifier

from DataLoader import DataLoader
from FeatureSelection import FeatureSelection
from Result import Result

from tqdm import tqdm_notebook as tqdm

# Import preprocessed data
ds = DataLoader()

# Unigram
# Without label 'pos'&'neg'
posFeatures_uni, negFeatures_uni = ds.get_uni_data()
 
# Unigram&Bigram
# Without label 'pos'&'neg'
posFeatures_unibi, negFeatures_unibi = ds.get_unibi_data()

def model(model, posFeatures, negFeatures, feature_nums):   
    # Initiate feature selection
    fs = FeatureSelection(posFeatures, negFeatures)
    
    posFeatures_label = list(posFeatures)
    negFeatures_label = list(negFeatures)

    # Add labels
    for i, word in tqdm(enumerate(posFeatures)):
        posFeatures_label[i] = [word, 'pos']
    # Add labels
    for i, word in tqdm(enumerate(negFeatures)):
        negFeatures_label[i] = [word, 'neg']
    
    trainFeatures, valFeatures = ds.get_ratio_data(posFeatures_label, negFeatures_label, 3/4.0)
    
    if model == 'uni':
        # No feature selection
        print "No feature selection"

        trainFeatures_new = list(trainFeatures)
        valFeatures_new = list(valFeatures)
        # Assign each word of posFeatures the value True
        for i, sentence in tqdm(enumerate(trainFeatures_new)):
            sentence[0] = dict([word, True] for word in sentence[0]) 
        # Assign each word of negFeatures the value True 
        for i, sentence in tqdm(enumerate(valFeatures_new)):
            sentence[0] = dict([word, True] for word in sentence[0])
        
        print "start Naive Bayes Classifier..."
        # Train a Naive Bayes Classifier
        classifier = NaiveBayesClassifier.train(trainFeatures_new)
    
        result = Result(trainFeatures_new, valFeatures_new, classifier)
        result.print_result()
    
    # Feature selection
    for num in feature_nums:
        print "Feature num:" + str(num)
    
        best_words = fs.feature_selection(num)
        trainFeatures_new = list(trainFeatures)
        valFeatures_new = list(valFeatures)
        # Assign each word of posFeatures the value True after feature selection
        for i, sentence in tqdm(enumerate(trainFeatures_new)):
            sentence[0] = dict([word, True] for word in sentence[0] if word in best_words) 
        # Assign each word of negFeatures the value True after feature selection
        for i, sentence in tqdm(enumerate(valFeatures_new)):
            sentence[0] = dict([word, True] for word in sentence[0] if word in best_words)
        
        print "start Naive Bayes Classifier..."
        # Train a Naive Bayes Classifier
        classifier = NaiveBayesClassifier.train(trainFeatures_new)
    
        result = Result(trainFeatures_new, valFeatures_new, classifier)
        result.print_result()
    
# Unigram model
print "Unigram model"
model('uni', posFeatures_uni, negFeatures_uni, [100000, 50000, 47500, 45000, 42500, 40000])

# Uni&Bigram model
print "Uni&Bigram model"
model('unibi', posFeatures_unibi, negFeatures_unibi, [1500000, 1250000, 1000000, 750000, 500000, 250000, 100000])