from sklearn import linear_model as lm
from sklearn import naive_bayes as nb
from sklearn import neural_network as nn
from sklearn import svm 
from sklearn import ensemble as en
from sklearn import cross_validation as cv
from sklearn import tree
from sklearn.externals.six import StringIO

# import pydot
import statistics as st
import numpy as np
import pandas as pd



# model: no testing data
def modelCompareTraining(trainFile="../result/training.csv", resultFile="../result/evalutionTraining.csv"):
	print("model compare in training")
	# data
	data = readData(trainFile)
	k = 10
	# trainingY = csv[:,0]
	# trainingX = csv[:,1:]
	total = len(data)
	# where is the y
	X = data[:,:-1]
	Y = data[:,-1]
	fNum = len(X)

	# 10-fold cross validation
	kf = cv.KFold(n=total, n_folds=k, shuffle=False, random_state=None)
	# accuracy list
	lgAccus = list()
	nbAccus = list()
	nnAccus = list()
	svmAccus = list()
	dtAccus = list()

	adaAccus = list()
	rfAccus = list()
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		
		# logistic regression
		lgAccus.append(lm.LogisticRegression().fit(X_train, Y_train).score(X_test, Y_test))
		# naive bayes
		nbAccus.append(nb.GaussianNB().fit(X_train, Y_train).score(X_test, Y_test))
		# neural network: python no

		# svm
		svmAccus.append(svm.SVC(kernel='rbf').fit(X_train, Y_train).score(X_test, Y_test))
		# decision tree
		dtAccus.append(tree.DecisionTreeClassifier().fit(X_train, Y_train).score(X_test, Y_test))
		# adaboost
		adaAccus.append(en.GradientBoostingClassifier().fit(X_train, Y_train).score(X_test, Y_test))
		# random forest: need to adjust n_estimator
		rfAccus.append(en.RandomForestClassifier().fit(X_train, Y_train).score(X_test, Y_test))


	# write decision tree file
	# dtModel = tree.DecisionTreeClassifier().fit(X, Y)
	# dot_data = StringIO() 
	# tree.export_graphviz(dtModel, out_file=dot_data) 
	# graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
	# graph.write_pdf("tree.pdf") 

	print(st.mean(lgAccus), st.mean(nbAccus), st.mean(svmAccus), st.mean(dtAccus), st.mean(adaAccus), st.mean(rfAccus))
	return st.mean(lgAccus)


# model: training and testing 
def modelCompareTesting(trainFile="../result/featureTrain.csv", testFile="../result/featureTest.csv", resultFile = "../result/evaluationTesting.csv"):
	print("model compare in testing")
	# data 
	testing = readData(trainFile)
	training = readData(testFile)
	trainingY = csv[:,0]
	trainingX = csv[:,1:]
	testingY = csv[:0]
	trainingX = csv[:,1:]

	# model
	# logistic regression

	# naive bayes

	# neural network

	# svm

	# decision tree

	# 

# inputFile: csv, input header, first column:y, else columns:x
def readData(inputFile):
	# if csv has header then skip_header=1
	csv = np.genfromtxt(inputFile, delimiter=",", skip_header=1)
	return csv



# feature selection

if __name__ == "__main__":
	modelCompareTraining()
