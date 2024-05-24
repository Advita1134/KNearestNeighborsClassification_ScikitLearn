from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

print(type(KNeighborsClassifier))

# Data #
# This creates data for the algorithm to train and test on.
## n_samples is the number of rows of the data.
## n_features is the number of columns.
## centers is the number of clusters (number of centers).
## cluster_std (cluster standard deviation) shows how spread apart the data is (the higher the number, the farther away).
X,y = make_blobs(n_samples = 20, n_features = 1, centers = 2, cluster_std = .5) 
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Model #
model = KNeighborsClassifier() # This model uses supervised learning.

# Training #
model.fit(X_train, y_train)

# Predicting #
y_pred = model.predict(X_test)
print(y_test)
print(y_pred)

answer = accuracy_score(y_test,y_pred) # This compares the y_test (the cluster the test data should be in) to y_pred (the cluster the test data ended up in after the model).
print(answer)

trainzeros = len(X_train) * [0]
testzeros = len(X_test) * [0]
plt.scatter(X_train, trainzeros, label = "training data", s = 20, c = y_train)
# plt.scatter(X_test, testzeros, label = "testing data", s = 100, c = y_test, marker = "x") # Answers given by make_blobs (it is a bit randomized)
plt.scatter(X_test, testzeros, label = "testing data", s = 100, c = y_pred, marker = "x") # Answers of the algorithm
plt.legend()
