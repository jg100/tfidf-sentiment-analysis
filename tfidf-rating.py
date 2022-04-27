import nltk
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import logging
from colorama import Fore, Back

path = "aclImdb/test/pos/"
neg_path = "aclImdb/test/neg/"

test_pos_path = "aclImdb/train/pos/"
test_neg_path = "aclImdb/test/pos/"


# function to load raw data into data frames

def load_data(folder_path):
    print(Fore.GREEN + "Loading data...")
    logging.info("Loading text data into data frame...")
    temp = []
    labels = []
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path + file), 'r') as f:
            temp.append(f.readlines()[0])
            labels.append(file[file.index('_') + 1:file.index('.')])
    df = pd.DataFrame({"reviews": temp,
                       "labels": labels})
    return df


def tokenize(dataset):
    print(Fore.GREEN + "Tokenizing data...")
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    for i in range(dataset.shape[0]):
        dataset["reviews"][i] = tokenizer.tokenize(dataset["reviews"][i])
    return dataset


# Stop word removal
def remove_stop_words(dataset):
    print(Fore.GREEN + "Removing Stop Words...")
    stop_words = set(stopwords.words('english'))
    for i in range(dataset.shape[0]):
        dataset["reviews"][i] = ([token.lower() for token in dataset["reviews"][i] if token not in stop_words])

    return dataset


# takes out any garbage values
def remove_garbage(dataset):
    print(Fore.GREEN + "Removing Garbage Values...")
    garbage = "~`!@#$%^&*()_-+={[}]|\:;'<,>.?/"
    for i in range(dataset.shape[0]):
        dataset.reviews[i] = " ".join([char for char in dataset.reviews[i] if char not in garbage])
    return dataset


'''
def calculate_tfidf(word, tf, N, df_doc, corpus):
    occurences = corpus.reviews.str.count(word)
    for i in range(df_doc.shape[0]):
        if word in df_doc.reviews[i]:
'''





def tfidf_vectorizer(train, test):
    # Number of documents in both sets (total docs)
    N = train.shape[0] + test.shape[0]
    corpus = pd.DataFrame({"reviews": train["reviews"]})
    corpus.reviews.append(test["reviews"], ignore_index=True)

    tfidf = pd.DataFrame(columns=corpus["reviews"])
    print(tfidf.columns)

    return tfidf


# Creating negative and positive data frames


# Logistic regression might work better because of the long sparse nature of our matrix

# Loading the training data
pos_df_train = load_data(path)
neg_df_train = load_data(neg_path)
train_set = pos_df_train.merge(neg_df_train, how="outer")  # deprecation warning

# Loading training set
pos_df_test = load_data(test_pos_path)
neg_df_test = load_data(test_neg_path)
test_set = pos_df_test.merge(neg_df_test, how="outer")

train_set = tokenize(train_set)
train_set = remove_stop_words(train_set)
train_set = remove_garbage(train_set)

test_set = tokenize(test_set)
test_set = remove_stop_words(test_set)
test_set = remove_garbage(test_set)

tfidf_vectorizer(train_set, test_set)

print(test_set)
print(train_set)
