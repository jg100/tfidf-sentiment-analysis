import os
import re
import gensim
import math
import numpy as np
import nltk.corpus
from sklearn.naive_bayes import MultinomialNB
from operator import add
import numpy as np
from nltk.corpus import stopwords

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
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
             "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
             "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
             "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
             "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
             "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
             "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
             "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
             "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
             "just", "don", "should", "now"]

stopwords = nltk.corpus.stopwords.words("english")

# Get reviews from a folder
# store the text in the corpus and store vals in a array also 
def getReviews(path, firstIndex, lastIndex):
    print("[getReviews() function called]")
    corpus = []
    review_actual_val = []

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
        
        # filter out stop words
        tokens_without_sw = [word for word in text_token if not word in stopwords]

        tempCorpus.append(' '.join(tokens_without_sw))

    return tempCorpus


class TfidfVectorizer():
    def __init__(self, n_gram=1):
        self.words_list = {}
        self.n_gram = n_gram

    def __str__(self):
        return "Test corpus: %s\nTest wordsList: %s" % (self.corpus, self.words_list)

    # get every unique word or n gram phrase from corpus
    def fit(self, corpus):
        print("Fitting corpus to vectors")
        unique_words = []

        # loop through the reviews documents
        for i in range(0, len(corpus), 1):
            # split up words in review 
            temp_words = corpus[i].split()

            for j in range(0, len(temp_words) - self.n_gram + 1, 1):

                # create the ngram phrase
                n_gram_word = ""
                for k in range(0, self.n_gram, 1):
                    n_gram_word += temp_words[j + k] + " "

                # remove last char which is a space in string
                n_gram_word = n_gram_word[:-1]

                # add to wordsList if it is not in it already
                if n_gram_word not in unique_words:
                    unique_words += [n_gram_word]

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

            for j in range(0, len(temp_words) - self.n_gram + 1, 1):

                # create the ngram phrase
                n_gram_word = ""
                for k in range(0, self.n_gram, 1):
                    n_gram_word += temp_words[j + k] + " "

                # remove last char which is a space in string
                n_gram_word = n_gram_word[:-1]

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
                tfidf[i][j] *= (math.log((total_doc + 1) / (doc_with_term[j] + 1)) + 1)
                euclideanNorm += tfidf[i][j] ** 2

            euclideanNorm = math.sqrt(euclideanNorm)

            # sum all values together then divide by eNorm
            for j in range(len(self.words_list)):
                if euclideanNorm == 0:
                    break
                tfidf[i][j] /= euclideanNorm

        return tfidf

class MultinomialNB2:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.priors = {}
        self.likelihood = {}

    # x is the tfidf matrix, y is the actual value for each row
    def fit(self, x_train, y_train):

        """
        classes_count is temp variable use to calculate the unique classes
        self.priors will be use to calculate the priors probabilities
        total_documents is the total number of documents use to fit MNB 
            which is use to calculate prior probabilities
        """
        classes_count = np.array(y_train)
        unique, counts = np.unique(classes_count, return_counts=True)
        self.priors = {A: B for A, B in zip(unique, counts)}
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
        class_total = {k: 0 for v,k in enumerate(classes)}

        # calc class_total
        for v,k in enumerate(classes):
            class_total[k] = sum(classes[k])

        # calculate all likelihood 
        for i in classes:
            classes[i][:] = np.divide(np.array(classes[i]) + 1, class_total[i] + self.alpha * num_features)
        
        # save likelihood 
        self.likelihood = classes
        return

    def predict(self, x_test):
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
                        p *= np.power(self.likelihood[j][k], i[k])

                # if probability is greater than previous update class and max probability
                if p > max_prob:
                    max_prob = p
                    likely_class = j
            
            prediction.append(likely_class)    
        return prediction


(tempCorpus, tempActualVal) = getReviews(TrP, 0, 100)
corpus += tempCorpus
review_actual_val += tempActualVal

(tempCorpus, tempActualVal) = getReviews(TrN, 0, 100)
corpus += tempCorpus
review_actual_val += tempActualVal

corpus = filterData(corpus, stopwords)
tfidf_vector = TfidfVectorizer(1)
tfidf_vector.fit(corpus)
X = tfidf_vector.transform(corpus)


# grabs
def test(testDirectory, startIndex, endIndex):
    corpus = []
    review_actual_val = []

    for x in testDirectory:
        (tempCorpus, tempActualVal) = getReviews(x, startIndex, endIndex)
        corpus += tempCorpus
        review_actual_val += tempActualVal

    return corpus, review_actual_val


MNB = MultinomialNB(alpha = 1)
MNB.fit(X, review_actual_val)

MNB2 = MultinomialNB2()
MNB2.fit(X, review_actual_val)

# MNB2 = MultinomialNB2(1)
# MNB2.fit(X, review_actual_val)

corpus = []
review_actual_val = []
corpus, review_actual_val = test([TeP, TeN], 0, 100)
X = tfidf_vector.transform(corpus)

p1 = MNB.predict(X)
p2 = MNB2.predict(X)

count1 = 0
count2 = 0
for i in range(len(p1)):
    if review_actual_val[i] == p1[i]:
        count1 += 1
    if review_actual_val[i] == p2[i]:
        count2 += 1

print("Reviews correct: ", count1, "; ", count2, " ;Total reviews: ", len(p1))



# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', norm='l2')
# Y = vectorizer.fit_transform(["good bad","ok activity"])

# print()
# print(Y[0, 1])
# print(vectorizer.get_feature_names_out())

# print(tfidf_val)

# print(gensim.parsing.stem_text("try writing nonsense"))
