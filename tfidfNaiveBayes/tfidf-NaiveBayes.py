import os
import re
from operator import add
from math import sqrt
from math import log
from numpy import power
from numpy import divide
from numpy import array
from numpy import unique

import nltk.corpus
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

directoryPos = "../aclImdb/train/pos"
directoryNeg = "../aclImdb/train/neg"
testPosDirectory = "../aclImdb/test/pos"
testNegDirectory = "../aclImdb/test/neg"

TrP = "./trainPos"
TrN = "./trainNeg"
TeP = "./testPos"
TeN = "./testNeg"

corpus = []
review_actual_val = []

# stop list from nltk
stopwords = nltk.corpus.stopwords.words("english")

pStemmer = PorterStemmer()


# Get reviews from a folder
# store the text in the corpus and store vals in a array also
def getReviews(path, firstIndex, lastIndex):
    print("[getReviews() function called]")
    corpus = []  # List of text
    review_actual_val = []  # List of values

    # Traverse folder for text files names
    reviewNames = os.listdir(path)[firstIndex:lastIndex]
    for x in reviewNames:
        if x.endswith(".txt"):
            # get the rating from file name
            rating = re.search(r"(?<=_)\d*", x).group()
            review_actual_val.append(rating)

            # Open the text file
            filePath = path + "/" + x
            file = open(filePath, "r", encoding="utf8")

            # add review to corpus
            text_review = file.read()

            # Close file
            file.close()

            corpus.append(text_review)

    return corpus, review_actual_val


def filterData(corpus, stopwords):
    print("filterData() function called")
    tempCorpus = []
    for text_review in corpus:
        # turn every character to lower case
        text_review = text_review.lower()

        # specific
        text_review = re.sub(r"won\'t", "will not", text_review)
        text_review = re.sub(r"can\'t", "can not", text_review)

        # general
        text_review = re.sub(r"n\'t", " not", text_review)
        text_review = re.sub(r"\'re", " are", text_review)
        text_review = re.sub(r"\'s", " is", text_review)
        text_review = re.sub(r"\'d", " would", text_review)
        text_review = re.sub(r"\'ll", " will", text_review)
        text_review = re.sub(r"\'t", " not", text_review)
        text_review = re.sub(r"\'ve", " have", text_review)
        text_review = re.sub(r"\'m", " am", text_review)

        # remove special characters leaving only word
        text_review = re.sub("[^A-Za-z]+", ' ', text_review)

        text_token = text_review.split()

        # filter out stop words and stem
        tokens_without_sw = [pStemmer.stem(word) for word in text_token if not word in stopwords]
        tempStr = ' '.join(tokens_without_sw)

        tempCorpus.append(tempStr)

    return tempCorpus


class Tfidf_Vectorizer():
    def __init__(self, n_gram=1):
        self.words_list = {}
        self.n_gram = n_gram

    def __str__(self):
        return "Test corpus: %s\nTest wordsList: %s" % (self.corpus, self.words_list)

    # get every unique word or n gram phrase from corpus
    def fit(self, corpus):
        print("Fitting corpus to vectors")
        unique_words = set()

        # loop through the reviews documents
        for i in range(len(corpus)):

            # split up words in review
            temp_words = corpus[i].split()

            for j in range(len(temp_words) - self.n_gram + 1):
                # create the ngram phrase
                n_gram_word = " ".join(temp_words[j:j + self.n_gram])  # Test this

                # add to wordsList if it is not in it already
                unique_words.add(n_gram_word)

        unique_words = list(set(unique_words))
        self.words_list = {unique_words[i]: i for i in range(len(unique_words))}

        return

    def transform(self, corpus):
        print("Transforming vectors...")
        # tfidf, total documents, total terms in document to normalize, number of document with term

        tfidf = [[0 for j in range(len(self.words_list))] for i in range(len(corpus))]
        total_doc = len(corpus)
        total_terms = [0 for i in range(len(corpus))]
        doc_with_term = [0 for i in range(len(self.words_list))]

        # loop through the review documents
        for i in range(0, len(corpus), 1):

            # split up words in review
            temp_words = corpus[i].split()

            # unique words use to count document with a term
            unique_words = set()

            for j in range(len(temp_words) - self.n_gram + 1):

                # create the ngram phrase
                n_gram_word = " ".join(temp_words[j:j + self.n_gram])  # Test this

                total_terms[i] += 1

                index = self.words_list.get(n_gram_word)
                if index is not None:
                    tfidf[i][index] += 1
                    unique_words.add(index)

            for x in unique_words:
                doc_with_term[x] += 1
            unique_words.clear()

        # calculate tfidf and normalize it using euclidean normalization
        # eNorm = (e1 + e2 + .. en) / || sqrt(e1^2 + e2^2 + ... + en^2) ||
        for i in range(len(corpus)):

            # number to calculate the denominator of the euclidean norm
            euclideanNorm = 0
            for j in range(len(self.words_list)):
                # tfidf = word_freq / word_count * total_num_doc / doc
                tfidf[i][j] /= total_terms[i]
                tfidf[i][j] *= (log((total_doc + 1) / (doc_with_term[j] + 1)) + 1)
                euclideanNorm += tfidf[i][j] ** 2

            euclideanNorm = sqrt(euclideanNorm)

            # sum all values together then divide by eNorm
            for j in range(len(self.words_list)):
                if euclideanNorm == 0:
                    break
                tfidf[i][j] /= euclideanNorm

        return tfidf


class MultinomialNaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.priors = {}
        self.likelihood = {}

    # x is the tfidf matrix, y is the actual value for each row
    def fit(self, x_train, y_train):
        print("Fiting vectors to Multinomial Naive Bayes")
        """
        classes_count is temp variable use to calculate the unique classes
        self.priors will be use to calculate the priors probabilities
        total_documents is the total number of documents use to fit MNB 
            which is use to calculate prior probabilities
        """
        classes_count = array(y_train)
        unique_words, counts = unique(classes_count, return_counts=True)
        self.priors = {A: B for A, B in zip(unique_words, counts)}
        total_documents = sum(self.priors.values())

        # calculate prior probabilities
        for k, v in self.priors.items():
            self.priors[k] = v / total_documents

        # check if it is 2d list
        if isinstance(x_train[0], list):
            num_features = len(x_train[0])
        else:
            num_features = len(x_train)

        # initialize dict with classes as keys and a list of 0 for features
        tempSet = set(y_train)
        tempSet = sorted(tempSet)

        # classes keep track of tfidf sum for each class
        classes = {k: [0 for i in range(num_features)] for v, k in enumerate(tempSet)}

        # sum all of the tfidf values for particular class
        for tfidf_doc, label in zip(x_train, y_train):
            classes[label][:] = map(add, classes[label], tfidf_doc)

        # class_total use to keep track of the sum tfdif of a row for a class
        class_total = {k: 0 for v, k in enumerate(classes)}

        # calc class_total
        for v, k in enumerate(classes):
            class_total[k] = sum(classes[k])

        # calculate all likelihood
        for i in classes:
            classes[i][:] = divide(array(classes[i]) + 1, class_total[i] + self.alpha * num_features)

        # save likelihood
        self.likelihood = classes
        return

    def predict(self, x_test):
        print("Predicting the result")
        prediction = []

        # calc argmax of each document
        for i in x_test:

            max_prob = 0
            likely_class = None

            # loop through all classes
            for j in self.priors:

                # calculate the probablitiy of being a particular class
                p = float(self.priors[j])
                for k in range(len(i)):
                    if i[k] != 0:
                        p *= power(self.likelihood[j][k], i[k])

                # if probability is greater than previous update class and max probability
                if p > max_prob:
                    max_prob = p
                    likely_class = j

            prediction.append(likely_class)
        return prediction


(tempCorpus, tempActualVal) = getReviews(TrP, 0, 1000)
corpus.extend(tempCorpus)
review_actual_val.extend(tempActualVal)

(tempCorpus, tempActualVal) = getReviews(TrN, 0, 1000)
corpus.extend(tempCorpus)
review_actual_val.extend(tempActualVal)

corpus = filterData(corpus, stopwords)
tfidf_vector = Tfidf_Vectorizer()
tfidf_vector.fit(corpus)
X = tfidf_vector.transform(corpus)


# grabs corpus and filters out corpus
def test(testDirectory, stopwords, startIndex, endIndex):
    corpus = []
    review_actual_val = []

    for x in testDirectory:
        (tempCorpus, tempActualVal) = getReviews(x, startIndex, endIndex)
        corpus.extend(tempCorpus)
        review_actual_val.extend(tempActualVal)

    corpus = filterData(corpus, stopwords)

    return corpus, review_actual_val


MNB = MultinomialNaiveBayes()
MNB.fit(X, review_actual_val)

corpus = []
review_actual_val = []
corpus, review_actual_val = test([TeP, TeN], stopwords, 0, 2000)
X = tfidf_vector.transform(corpus)

p1 = MNB.predict(X)

count1 = 0
actualVal = {'1': 0, '2': 0, '3': 0, '4': 0, '7': 0, '8': 0, '9': 0, '10': 0}
predictVal = {'1': 0, '2': 0, '3': 0, '4': 0, '7': 0, '8': 0, '9': 0, '10': 0}
for x in review_actual_val:
    actualVal[x] += 1

for x in p1:
    predictVal[x] += 1

# Metrics
confusion_matrix = np.zeros((10, 10))
count = 0
for i in range(len(p1)):
    if review_actual_val[i] == p1[i]:
        # print(review_actual_val[i])
        count += 1
        confusion_matrix[int(review_actual_val[i])-1][int(review_actual_val[i])-1] += 1
    else:
        confusion_matrix[int(review_actual_val[i])-1][int(p1[i])-1] += 1

print("[Calculating metrics...]")
print("Reviews correct: ", count, " ;Total reviews: ", len(p1))
print(f"[Accuracy]: {count / len(p1)}")
# print(confusion_matrix)

print("     Precision       Recall")
for curr_class in range(0, 10):
    class_tp = confusion_matrix[curr_class][curr_class]
    class_tn = 0

    class_fp = sum(confusion_matrix[curr_class]) - confusion_matrix[curr_class][curr_class]
    class_fn = confusion_matrix.sum(axis=0)[curr_class]

    for i in range(0, 9):
        for j in range(0,9):
            if i != curr_class or j != curr_class:
                class_tn += confusion_matrix[i][j]

    if class_tp != 0:
        precision = class_tp / (class_tp + class_fp)
        recall = class_tp / (class_tp + class_fn)
        print(f"Class {curr_class + 1}: {precision}      {recall}")





# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', norm='l2')
# Y = vectorizer.fit_transform(["good bad","ok activity"])

# print()
# print(Y[0, 1])
# print(vectorizer.get_feature_names_out())

# print(tfidf_val)

# print(gensim.parsing.stem_text("try writing nonsense"))
