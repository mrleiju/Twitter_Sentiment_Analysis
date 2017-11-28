import re
import nltk, nltk.classify.util


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
import csv
from Preprocessor import PreProcessor
from FeatureSelection import FeatureSelection
from random import shuffle

# helper function for testing
def purify(s):
    s = re.sub(r'[^\w!? \t\n]+', '', s)
    s = re.sub(r'[!]+', ' ! ', s)
    s = re.sub(r'[?]+', ' ? ', s)
    return s

# data loader, adapted for KNN specifically
class DataLoader:
    def __init__(self):
        # Tokenize words and keep emoticons
        self.posFeatures = []
        self.negFeatures = []
        self.pp = PreProcessor()
        with open('Sentiment Analysis Dataset.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Sentiment'] == '1':
                    posWord = re.findall(r"[http]+://[^\s]*|@[^\s]*|#[^\s]*|[\w']+|[:)-;=)(>o3Dx^\/*w8~_T|]+", row["SentimentText"].rstrip())
                    self.posFeatures.append(self.pp.process(posWord))
                elif row['Sentiment'] == '0':
                    negWord = re.findall(r"[http]+://[^\s]*|@[^\s]*|#[^\s]*|[\w']+|[:)-;=)(>o3Dx^\/*w8~_T|]+", row["SentimentText"].rstrip())
                    self.negFeatures.append(self.pp.process(negWord))
        shuffle(self.posFeatures)
        shuffle(self.negFeatures)

    def get_data(self):
        return self.posFeatures, self.negFeatures

    # POS tagging
    def select(self, feature_select):
        posFeatures_selected = []
        negFeatures_selected = []
        for words in self.posFeatures:
            posFeatures_selected.append([feature_select(words), 'pos'])
        for words in self.negFeatures:
            negFeatures_selected.append([feature_select(words), 'neg'])
        return posFeatures_selected, negFeatures_selected

    # Get train&test data
    def split(self, posdata, negdata, train_num, test_num):
        train_num /= 2
        test_num /= 2
        trainFeatures = posdata[:train_num] + negdata[:train_num]
        testFeatures = posdata[-test_num:] + negdata[-test_num:]
        return trainFeatures, testFeatures


# Load the dat aand construct the scoring dictionary
data = DataLoader()
feature_selector = FeatureSelection(data.posFeatures, data.negFeatures)
best_words = feature_selector.feature_selection(200)

# feature selectors
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])
def make_full_dict(words):
	return dict([(word, True) for word in words])

# select the features and split the train, test sets
posFeatures_selected, negFeatures_selected = data.select(best_word_features)


# selecting the K:
def selectK():
    for K in range(1, 28, 2):
        classifier = SklearnClassifier(knn(n_neighbors=K))
        classifier.train(train)
        print 'K = %2d: accuracy: %.3f' % (K, nltk.classify.util.accuracy(classifier, test))




#%% split the training and testing set, then select the best K
train, test = data.split(posFeatures_selected, negFeatures_selected, 700000, 500)

selectK()

#%% Now we can see how we are doing via the various metrics

classifier = SklearnClassifier(knn(n_neighbors=17))
classifier.train(train)


referenceSets = {}
referenceSets['pos'] = set()
referenceSets['neg'] = set()
testSets = {}
testSets['pos'] = set()
testSets['neg'] = set()

shuffle(test)
for i, (features, label) in enumerate(test):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)
print 'After training on %d samples, start to test on %d instances:' % (len(train), len(test))
print 'accuracy:        %.2f' % nltk.classify.util.accuracy(classifier, test)
print 'pos precision:   %.2f' % nltk.precision(referenceSets['pos'], testSets['pos'])
print 'neg precision:   %.2f' %  nltk.precision(referenceSets['neg'], testSets['neg'])
print 'pos recall:      %.2f' % nltk.recall(referenceSets['pos'], testSets['pos'])
print 'neg recall:      %.2f' %  nltk.recall(referenceSets['neg'], testSets['neg'])


testSets['neg'].clear()
testSets['pos'].clear()
referenceSets['neg'].clear()
referenceSets['pos'].clear()

for j in range(5):
    test_subset = test[j*100:(j+1)*100]
    for i, (features, label) in enumerate(test_subset):
            referenceSets[label].add(i)
            predicted = classifier.classify(features)
            testSets[predicted].add(i)
    
    print 'After training on %d samples, start to test on %d instances:'  % (len(train), len(test_subset))
    print 'accuracy:        %.2f' % nltk.classify.util.accuracy(classifier, test_subset)
    print 'pos precision:   %.2f' % nltk.precision(referenceSets['pos'], testSets['pos'])
    print 'neg precision:   %.2f' %  nltk.precision(referenceSets['neg'], testSets['neg'])
    print 'pos recall:      %.2f' % nltk.recall(referenceSets['pos'], testSets['pos'])
    print 'neg recall:      %.2f' %  nltk.recall(referenceSets['neg'], testSets['neg'])
    
