import os
import re

import nltk.corpus
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn_deltatfidf import DeltaTfidfVectorizer


TrP = "./trainPos"
TrN = "./trainNeg"
TeP = "./testPos"
TeN = "./testNeg"

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

corpus = []
review_actual_val = []

(tempCorpus, tempActualVal) = getReviews(TrP, 0, 1000)
corpus.extend(tempCorpus)
review_actual_val.extend(tempActualVal)

(tempCorpus, tempActualVal) = getReviews(TrN, 0, 1000)
corpus.extend(tempCorpus)
review_actual_val.extend(tempActualVal)

corpus = filterData(corpus, stopwords)

tVec = DeltaTfidfVectorizer()
tVec.fit(corpus, review_actual_val)
tfidf = tVec.transform(corpus)

# SVM
SVCClf = SVC()
SVCClf.fit(tfidf, review_actual_val)

def predictionResult(actual, predict):
    correctVal = 0
    totalVal = 0
    
    for i in range(len(actual)):
        if actual[i] == predict[i]:
            correctVal += 1
        totalVal += 1

    print("correctValue: ", correctVal, "; totalValue: ", totalVal)
    print("Percentage: ", round(correctVal/totalVal * 100, 2), "%")

    return

# grabs corpus and filters out corpus
def test(tVec, model, testDirectory, stopwords, startIndex, endIndex):
    corpus = []
    review_actual_val = []

    for x in testDirectory:
        (tempCorpus, tempActualVal) = getReviews(x, startIndex, endIndex)
        corpus.extend(tempCorpus)
        review_actual_val.extend(tempActualVal)
    
    corpus = filterData(corpus, stopwords)
    
    tfidf = tVec.transform(corpus)
    prediction = model.predict(tfidf)

    return review_actual_val, prediction

review_actual_val, prediction = test(tVec, SVCClf, [TeP, TeN], stopwords, 0, 1000)

print(classification_report(review_actual_val, prediction))