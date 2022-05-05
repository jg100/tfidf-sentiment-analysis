import os
import re
import gensim
import math
import numpy as np
import nltk.corpus
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

directoryPos = "../aclImdb/train/pos"
directoryNeg = "../aclImdb/train/neg"
testPosDirectory = "../aclImdb/test/pos"
testNegDirectory = "../aclImdb/test/neg"

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

        # remove special characters leaving only word and apostrophe
        text_review = re.sub("[^A-Za-z]+", ' ', text_review)

        text_token = text_review.split()
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

        # calculate tfidf and normalize it using euclidean norm
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
            for j in range(len(self.words_list)):
                if euclideanNorm == 0:
                    break
                tfidf[i][j] /= euclideanNorm

        return tfidf


# class MultinomialNB:
#     def __init__(self, n_gram=1):
#         self.words_list = {}
#         self.n_gram = n_gram
#         return


(tempCorpus, tempActualVal) = getReviews(directoryPos, 0, 1000)
corpus += tempCorpus
review_actual_val += tempActualVal

(tempCorpus, tempActualVal) = getReviews(directoryNeg, 0, 1000)
corpus += tempCorpus
review_actual_val += tempActualVal

corpus = filterData(corpus, stopwords)

tfidf_vector = TfidfVectorizer(2)
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


MNB = MultinomialNB(alpha=1)

MNB.fit(X, review_actual_val)

corpus = []
review_actual_val = []
corpus, review_actual_val = test([testPosDirectory, testNegDirectory], 0, 100)
X = tfidf_vector.transform(corpus)

p = MNB.predict(X)


# Calculate the metrics through confusion matrix
confusion_matrix = np.zeros((10, 10))
count = 0
for i in range(len(p)):
    if review_actual_val[i] == p[i]:
        count += 1
        confusion_matrix[int(review_actual_val[i])-1][int(review_actual_val[i])-1] += 1
    else:
        confusion_matrix[int(review_actual_val[i])-1][int(p[i])-1] += 1

print("[Calculating metrics...]")
print("Reviews correct: ", count, " ;Total reviews: ", len(p))
print(confusion_matrix)

print("     Precision       Recall")
for curr_class in range(0, 9):
    class_tp = confusion_matrix[curr_class][curr_class]
    class_tn = 0

    class_fp = sum(confusion_matrix[curr_class]) - confusion_matrix[curr_class][curr_class]
    class_fn = confusion_matrix.sum(axis=0)[curr_class]

    for i in range(0, 9):
        for j in range(0,9):
            if i != curr_class or j != curr_class:
                class_tn += confusion_matrix[i][j]

    print(f"Class {curr_class + 1}: {class_tp / (class_tp + class_fp)}      {class_tp / (class_tp + class_fn)}")





# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', norm='l2')
# Y = vectorizer.fit_transform(["good bad","ok activity"])

# print()
# print(Y[0, 1])
# print(vectorizer.get_feature_names_out())

# print(tfidf_val)

# print(gensim.parsing.stem_text("try writing nonsense"))
