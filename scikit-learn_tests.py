import numpy as np
import urllib
from sklearn import svm, tree, cluster
import matplotlib.pyplot as plt

# load the CSV file as a numpy matrix
dataset = np.loadtxt("diabetes.csv", delimiter=",")
print(dataset.shape)
# separate the data from the target attributes
data = dataset[:,0:8]
target = dataset[:,8]
#The following line must be used from the version 0.19, since the target must be an array of many arrays of targets, one for each data sample. The weird thing is, right now the previous line is raising a deprecation warning, saying that this is not gonna work anymore in the version 0.19 and suggests to use the new way. When I use the new way, it says that it is still wrong and ask to use the old way. Anyway, I will keep the old manner (1d-array). 
#target = dataset[:,8].reshape(-1,1)
print data
print target
#Using SVM for classification
#All other algorithms can be used following this pattern: instantiate the algorithm with (or without) parameters; use the method fit an pass the data and the targets; use the method predict and pass data for prediction.
#Usually, the constructor receives parameters for the algorithm. However, in this case, we are going to use the default ones (defined by the library itself).
clf = svm.SVC()
#Using all the samples except the last one for training (data and targets)
clf.fit(data[0:-1,:],target[0:-1])
#Using the last sample for prediction
print clf.predict(data[-1,:])
#Using decision tree for classification
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data[0:-1,:],target[0:-1])
print clf.predict(data[-1,:])
#Using kmeans for clusterization
clus = cluster.KMeans(2)
clus = clus.fit_predict(data)
#Plotting multidimensional data
plt.figure(1)
plt.subplot(221)
plt.scatter(data[:, 0], data[:, 1], c=clus)
plt.title("Dimensions 1 and 2")
plt.subplot(222)
plt.scatter(data[:, 2], data[:, 3], c=clus)
plt.title("Dimensions 3 and 4")
plt.subplot(223)
plt.scatter(data[:, 4], data[:, 5], c=clus)
plt.title("Dimensions 5 and 6")
plt.subplot(224)
plt.scatter(data[:, 6], data[:, 7], c=clus)
plt.title("Dimensions 7 and 8")
plt.savefig("teste")
