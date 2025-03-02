#  SENTIMENT ANALYSIS USING LOGISTIC REGRESSION 
Project Overview:
- The "sentiment" dataset consists of 3000 sentences which come from reviews on "imdb.com", "amazon.com", and "yelp.com". Each sentence is labeled according to whether it comes from a positive review or negative review. 
- The data set consists of 3000 sentences, each labeled '1' (if it came from a positive review) or '0' (if it came from a negative review). To be consistent with our notation from lecture, we will change the 
negative review label to '-1'. 
- We will use logistic regression to learn a classifier from this data.
# step for evaluating the model
1. Load and Preprocess data :
   Preprocessing the text data 
- To transform this prediction problem into an linear classification, we will need to preprocess the 
text data. We will do four transformations: 
o Remove punctuation and numbers. 
o Transform all words to lower-case. 
o Remove stop words. 
o Convert the sentences into vectors, using a bag-of-words representation. 
- We begin with first two steps
  Bag of words 
- In order to use linear classifiers on our data set, we need to transform our textual data into numeric 
data. The classical way to do this is known as the bag of words representation. 
- In this representation, each word is thought of as corresponding to a number in "{1, 2, ..., V}" where 
"V" is the size of our vocabulary. And each sentence is represented as a V-dimensional vector    , 
where      is the number of times that word     occurs in the sentence. 
- To do this transformation, we will make use of the "CountVectorizer" class in "scikit-learn". We will 
cap the number of features at 4500, meaning a word will make it into our vocabulary only if it is one 
of the 4500 most common words in the corpus. This is often a useful step as it can weed out spelling 
mistakes and words which occur too infrequently to be useful. 
- Finally, we will also append a '1' to the end of each vector to allow our linear classifier to learn a bias 
term.

2. Fitting a logistic regression model to the training data 
- We could implement our own logistic regression solver using stochastic gradient descent, 
but fortunately, there is already one built into "scikit-learn". 
- Due to the randomness in the SGD procedure, different runs can yield slightly different 
solutions (and thus different error values).

3. Analyzing the margin 
- The logistic regression model produces not just classifications but also conditional 
probability estimates. 
- We will say that "x" has margin "gamma" if (according to the logistic regression model) 
"Pr(y=1|x) > (1/2)+gamma" or "Pr(y=1|x) < (1/2)-gamma". The following function 
margin_counts takes as input as the classifier ("clf", computed earlier), the test set 
("test_data"), and a value of "gamma", and computes how many points in the test set have 
margin of at least "gamma".

4. Words with large influence 
- Finally, we attempt to partially interpret the logistic regression model. 
- Which words are most important in deciding whether a sentence is positive? As a first approximation 
to this, we simply take the words whose coefficients in "w" have the largest positive values. 
- Likewise, we look at the words whose coefficients in "w" have the most negative values, and we 
think of these as influential in negative predictions.
