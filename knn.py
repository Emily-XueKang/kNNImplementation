from classifier import classifier
import math
import operator
import random
from scipy.io import arff
import pandas as pd
import numpy as np

class KNN(classifier):
	'''Implementation of kNN algrithm
	'''
	def __init__(self, k=3):
		self.k = k

	def get_distance(self, instance1, instance2):
		'''Given two data points(instances)
		calculate the euclidean distance between them
		'''
		points = zip(instance1, instance2)
		#sqrt((i1-j1)^2 + (i2-j2)^2 + ... + (in-jn)^)
		diffs_squared_distance = [pow(int(i)-int(j), 2) for (i, j) in points]
		return math.sqrt(sum(diffs_squared_distance))

	def get_neighbors(self, training_set, test_instance):
		distances = []
		for training_instance in training_set:
			dist = self.get_distance(training_instance[0], test_instance)
			distances.append((training_instance, dist))
		sorted_distances = sorted(distances, key=operator.itemgetter(1))#sort by 'dist'

		#get the training instances, do not need distance value here
		sorted_training_instances = [tuple[0] for tuple in sorted_distances]

		#select top k elements
		return sorted_training_instances[:self.k]

	def get_majority_class(self, neighbors):
		votes = {}
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			if response in votes:
				votes[response] += 1
			else:
				votes[response] = 1
		sorted_votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
		return sorted_votes[0][0]

	def get_accuracy(self, test_set, predictions):
		correct = 0
		for x in range(len(test_set)):
			if test_set[x][-1] == predictions[x]:
				correct += 1
		return (correct/float(len(test_set)))

	def fit(self, X, Y):
		pass

	def predict(self, X_test, train, test):
		predictions = []
		for x in range(len(X_test)):
			neighbors = self.get_neighbors(train, test[x][0])
			majority = self.get_majority_class(neighbors)
			predictions.append(majority)
		print('Accuracy score of k = ' 
			+ str(self.k) +' : '+ 
			str(self.get_accuracy(test, predictions)))

Data = arff.loadarff('PhishingData.arff')
DF = pd.DataFrame(Data[0])
DF = DF.astype('int')
X = DF.iloc[:, 0:9]
Y = DF.iloc[:, 9]
X_array = X.values
Y_array = Y.values
splitpoint = int(len(Y_array)*0.8)
#print(splitpoint)
X_train = X_array[:splitpoint]
Y_train = Y_array[:splitpoint]
X_test = X_array[splitpoint:]
Y_test = Y_array[splitpoint:]
#reformat train/test datasets for convenience
train = list(zip(X_train, Y_train))
test = list(zip(X_test, Y_test))
for k in range(2, 33):
	my_knn = KNN(k)
	my_knn.predict(X_test,train,test)