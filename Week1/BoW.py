from TextPreprocessing import clean
from nltk.tokenize import word_tokenize
import numpy as np

def TF(word, document):
    return document.count(word)

def TF_scaled(word,document):
  return document.count(word)/(len(word_tokenize(document)))

def df(documents,term):
    ctr = 0
    for doc in documents:
        if term in doc:
            ctr+=1
    return ctr

def idf(documents,term):
    n = len(documents)
    return 1+np.log((1+n)/(1+df(documents,term)))

def TFIDF(documents):
    preprocessed = list(map(clean, documents))
    vocabulary = list(set(" ".join(preprocessed).split()))
    vocabulary.sort()
    X = np.zeros([len(vocabulary), len(documents)],dtype = 'float')
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i,j] = TF_scaled(vocabulary[i], preprocessed[j])*idf(preprocessed,vocabulary[i])
    return X
    
    
def BagOfWords(documents):
    preprocessed = list(map(clean, documents))
    vocabulary = list(set(" ".join(preprocessed).split()))
    vocabulary.sort()
    X = np.zeros([len(vocabulary), len(documents)],dtype = 'float')
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i,j] = TF(vocabulary[i], preprocessed[j])
    return X

def similarity(X1, X2):
  return np.dot(X1,X2)/(np.linalg.norm(X1)*np.linalg.norm(X2))

def score(X):
  scores = []
  covered = []
  X = X.T
  m = X.shape[0]
  for i in range(m):
    for j in range(m):
      if i!=j and ((i,j) not in covered and (j,i) not in covered):
        scores.append(([i,j],similarity(X[i],X[j])))
        covered.append((i,j))
  return scores



documents = ["I have a dream, that one day out in the red hills of Georgia the suns of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood",
                "Gravity is like madness, all it needs is a little push. A little push to make the dreams work and free the slaves",
                "Then was the age of Revolution with the Iron Age.",
                "Now we are in the Digital Age."
        ]

BoW = BagOfWords(documents)
tfidf = TFIDF(documents)
sim_scores = score(tfidf)