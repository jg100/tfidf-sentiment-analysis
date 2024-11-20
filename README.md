##About the program
In the tfidfNaiveBayes folder contains the tfidf and multinomial naive bayes implementation

The moveReviews.py can be ignorse. It is a python script to move the reviews to the four folders inside.

The tfidf-NaiveBayes.py file is where the implementation occurs. 

There is a getReviews function to get reviews from a folder and store it into a list of strings
where each document is a string.

The filterData gets the list of strings and removes special characters and splits contractions into
two words. Then it removes stop words and then stems the word using the porter stemmer. It combines
the words token together. The function returns a list of strings.

The Tfidf_Vectorizer is has a n_gram option. A fit method takes a list of strings and saves the
n_gram phrases into a dictionary with a unique string and index. The transform method transforms
a list of strings into a 2d list of tfidf values for each string. The transform also performs a
Euclidean normalization on the tfidf after it is computed. 

The multinomial naive bayes class has the option of an alpha value for smoothing. The fit method takes
in a tfidf 2d list or Matrix of tfidf values and the actual values of the tfidf for each document
which is the string review. The fit method calculates the prior probablities and calculates the
likelihood. The predict method takes in a list of tfidf values and predicts it based on the data
that was fitted.

The test function just gets the reviews, preprocess the reviews based on filterData and return the
corpus of reviews as string and the actual review values.

##Running the program

###Requirements
- Make sure to have the latest version of python installed in your system from the python website
  
- Open you computers terminal/console and navigate to the tfidf-sentiment-analysis directory

Execute the following start up command
```buildoutcfg
> pip install -r requirements.txt
```

- Run the program in the command line
```buildoutcfg
> python tfidf-NaiveBayes.py <number of reviews to test>
```

ex: Testing 10 reviews
```buildoutcfg
> python tfidf-NaiveBayes.py 10
```
The result is a statistical representation of the reviews that were tested.

