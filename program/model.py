from sklearn import linear_model as lm
from sklearn import naive_bayes as nb
from sklearn import neural_network as nn
from sklearn import svm 
from sklearn import ensemble as en
from sklearn import cross_validation as cv
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from io import StringIO
# from sklearn.externals.six import StringIO

import matplotlib.pyplot as plt
import pydotplus
import statistics as st
import numpy as np
import pandas as pd



# model: no testing data
def modelCompareTraining(trainFile="../result/temp_feature.csv", resultFile="../result/evalutionTraining.csv"):
	print("model compare in training")
	# data
	data = readData(trainFile)
	k = 10
	# trainingY = csv[:,0]
	# trainingX = csv[:,1:]
	
	# where is the y
	X = data[:,1:]
	Y = data[:,0]
	(nrow, ncol) = X.shape
	

	# 10-fold cross validation
	kf = cv.KFold(n=nrow, n_folds=k, shuffle=False, random_state=None)
	# accuracy list
	lgAccus = list()
	lgAUC = list()

	nbAccus = list()
	nbAUC = list()

	svmAccus = list()
	svmAUC = list()

	lsvmAccus = list()
	lsvmAUC = list()

	dtAccus = list()
	dtAUC = list()
	dtAUClog = list()

	adaAccus = list()
	adaAUC = list()
	rfAccus = list()
	rfAUC = list()
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		
		# logistic regression
		lgAccus.append(lm.LogisticRegression().fit(X_train, Y_train).score(X_test, Y_test))

		# naive bayes
		nbAccus.append(nb.GaussianNB().fit(X_train, Y_train).score(X_test, Y_test))
		# neural network: python no

		# svm: rbf
		svmAccus.append(svm.SVC(kernel='rbf').fit(X_train, Y_train).score(X_test, Y_test))
		lsvmAccus.append(svm.LinearSVC().fit(X_train, Y_train).score(X_test, Y_test))
		# decision tree
		dtModel = tree.DecisionTreeClassifier().fit(X_train, Y_train)
		dtAccus.append(dtModel.score(X_test, Y_test))

		probas = dtModel.predict_proba(X_test)
		dtAUC.append(calculateAUC(probas, Y_test))

		probasLog = dtModel.predict_log_proba(X_test)
		dtAUClog.append(calculateAUC(probasLog, Y_test))

		# # adaboost
		# adaAccus.append(en.GradientBoostingClassifier().fit(X_train, Y_train).score(X_test, Y_test))
		# # random forest: need to adjust n_estimator
		# rfAccus.append(en.RandomForestClassifier(max_features=(int(ncol/2))).fit(X_train, Y_train).score(X_test, Y_test))


	# write decision tree file
	# outputTree(X, Y, fileName)

	# get feature importance in decision tree ans randomforest
	# rfModel = en.RandomForestClassifier(max_features=(int(ncol/2))).fit(X_train, Y_train)
	# print(rfModel.feature_importances_)

	# pydotplus is better

	print(st.mean(dtAUC), st.mean(dtAUClog))
	# print(st.mean(lgAccus), st.mean(nbAccus), st.mean(svmAccus), st.mean(lsvmAccus), st.mean(dtAccus), st.mean(adaAccus), st.mean(rfAccus))
	# return st.mean(lgAccus)


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
def outputTree(X, Y, fileName):
	dtModel = tree.DecisionTreeClassifier().fit(X, Y)
	dot_data = StringIO() 
	tree.export_graphviz(dtModel, out_file=dot_data) 
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
	graph.write_pdf(fileName)
	 # "tree.pdf"

# inputFile: csv, input header, first column:y, else columns:x
def readData(inputFile):
	# if csv has header then skip_header=1
	csv = np.loadtxt(inputFile, delimiter=",", skiprows=1)
	return csv

def writeEvalution(outputFile, modelResult):
	with open(outputFile, 'a') as fo:
		for (model, accu) in modelResult.items():
			fo.write(model+":"+str(acct)+'\n')

def calculateAUC(Y_proba, Y_test):
	fpr, tpr, thresholds = roc_curve(Y_test, Y_proba[:,1])
	roc_auc = auc(fpr, tpr)
	print(roc_auc)
	return roc_auc



# feature selection

if __name__ == "__main__":
	modelCompareTraining()
