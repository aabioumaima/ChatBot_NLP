import nltk 
from nltk.stem.porter import PorterStemmer
import numpy as np
#nltk.download('punkt')


stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
    pass

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_word(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype = np.float32) # a bag of zeros, with the same length of all_words
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bog = bag_of_word(sentence, words)
print(bog)