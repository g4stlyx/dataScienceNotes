# Case Study: Training a Naive Bayes Classifier for Sentiment Analysis

* Sentiment analysis: analyzing text to discern emotion or sentiment. (e.g analyzing reviews)
* Naive Bayes Algorithm: classification technique based on Bayes Theorem.
    * P(A|B) = P(B|A) * P(A) / P(B)
    * e.g an email is a spam or not (spam or ham problem)
* Our aim in this example is `to predict the star rating of a review.`


### Tokenization and Vectorization

* Tokenization: converting a sequence of text into SMALLER PARTS, known as tokens.
* Vectorization: converting each token into NUMBERS, for model to understand them.
    * Each token'd be encoded into an array of 2 values representing their frequency.
    * This is called BAG OF WORDS (BoW) representation.

### Handling Imbalanced Data

* resampling: undersampling, oversampling, smote
* weighted classes
* data augmentation: generating new samples from existing data
* weighted ML algorithms