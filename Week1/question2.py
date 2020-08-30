#importing the libraries
import numpy as np
import pandas as pd

#importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)'''

def cost_function(X, Y, B):
 m = len(Y)
 J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
 return J

def batch_gradient_descent(X, Y, B, alpha, iterations):
  cost_history = [0] * iterations
  m = len(Y)
  for iteration in range(iterations):
       h = X.dot(B)
       loss = h - Y
       gradient = X.T.dot(loss) / m
       B = B - alpha * gradient
       cost = cost_function(X, Y, B)
       cost_history[iteration] = cost
  return B, cost_history



m = 40
f = 4
X_train = X[:m,:f]
X_train = np.c_[np.ones(len(X_train),dtype='int64'),X_train]
y_train = y[:m]
X_test = X[m:,:f]
X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]
y_test = y[m:]

# Initial Coefficients
B = np.zeros(X_train.shape[1])
alpha = 0.005
iter_ = 100
newB, cost_history = batch_gradient_descent(X_train, y_train, B, alpha, iter_)

def pred(x_test, newB):
   return x_test.dot(newB)

a = pred(X_test,newB)

def r2(y_,y):
 sst = np.sum((y-y.mean())**2)
 ssr = np.sum((y_-y)**2)
 r2 = 1-(ssr/sst)
 return(r2)

r2(y_,y_test)