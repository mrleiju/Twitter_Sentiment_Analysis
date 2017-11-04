import nltk
import string


class PreProcessor:
    def __init__(self):
        text = nltk.corpus.words.words('en')
        self.text_dict = dict()
        for word in text:
            if not self.text_dict.has_key(word):
                self.text_dict[word] = True

    def process(self, words):
        for word in words:
            if (not self.text_dict.has_key(word)) or word.isdigit() or (word in string.punctuation):
                words.remove(word)
        return map(string.lower, words)
