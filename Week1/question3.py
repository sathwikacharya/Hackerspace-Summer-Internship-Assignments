# Importing the libraries
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import time
import pandas as pd
from nltk.tokenize import word_tokenize


# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# Cleaning the texts
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = ' '.join(review)
    corpus.append(review)
   
stem_raw = []
stem_clean = []
lemm_raw = []
lemm_clean = []

for j in range(0,1000):
    start = time.time()
    tokens = list(map(stemmer.stem, dataset['Review'][j]))
    stem_raw.append(time.time()-start)
    
    start = time.time()
    tokens = list(map(stemmer.stem, corpus[j]))
    stem_clean.append(time.time()-start)
    
    start = time.time()
    tokens = list(map(lemmatizer.lemmatize, dataset['Review'][j]))
    lemm_raw.append(time.time()-start)
    
    start = time.time()
    tokens = list(map(lemmatizer.lemmatize, corpus[j]))
    lemm_clean.append(time.time()-start)

