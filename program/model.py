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
def modelCompareTraining(trainFile="../processedData/training_feature.csv", resultFile="../result/evalutionTraining.csv", importanceFile ="../result/featureImportance.csv"):
	print("model compare in training")
	# data
	data = readData(trainFile)
	k = 10
	# trainingY = csv[:,0]
	# trainingX = csv[:,1:]
	
	# where is the y
	X = data[:,2:]
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

	c=0
	for train_index, test_index in kf:
		print("iteration:"+str(c))
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		
		# logistic regression
		print("logistic regression")
		lgModel = lm.LogisticRegression().fit(X_train, Y_train)
		lgAccus.append(lgModel.score(X_test, Y_test))
		lgProbas = lgModel.predict_proba(X_test)
		lgAUC.append(calculateAUC(lgProbas, Y_test))

		# naive bayes
		print("nb")
		nbModel = nb.GaussianNB().fit(X_train, Y_train)
		nbAccus.append(nbModel.score(X_test, Y_test))
		nbProbas = nbModel.predict_proba(X_test)
		nbAUC.append(calculateAUC(nbProbas, Y_test))
		# neural network: python no

		# svm: rbf
		print("svm")
		svmAccus.append(svm.SVC(kernel='rbf').fit(X_train, Y_train).score(X_test, Y_test))
		lsvmAccus.append(svm.LinearSVC().fit(X_train, Y_train).score(X_test, Y_test))
		# decision tree
		dtModel = tree.DecisionTreeClassifier().fit(X_train, Y_train)
		dtAccus.append(dtModel.score(X_test, Y_test))
		dtProbas = dtModel.predict_log_proba(X_test)
		dtAUC.append(calculateAUC(dtProbas, Y_test))

		# adaboost
		print("adaboost")
		adaModel = en.GradientBoostingClassifier().fit(X_train, Y_train)
		adaAccus.append(adaModel.score(X_test, Y_test))
		adaProbas = adaModel.predict_log_proba(X_test)
		adaAUC.append(calculateAUC(dtProbas, Y_test))
		# random forest: need to adjust n_estimator
		print("random forest")
		rfModel = en.RandomForestClassifier(max_features=(int(ncol/2))).fit(X_train, Y_train)
		rfAccu.append(rfModel.score(X_test, Y_test))
		rfProbas = rfModel.predict_log_proba(X_test)
		rfAUC.append(calculateAUC(rfProbas, Y_test))
		c=c+1
	# write decision tree file
	# outputTree(X, Y, fileName)

	# get feature importance in decision tree ans randomforest
	rfModel = en.RandomForestClassifier(max_features=(int(ncol/2))).fit(X_train, Y_train)
	with open(featureImportance, 'w') as fo:
		fo.write(",".join([str(a) for a in rfModel.feature_importances_]))
	print(rfModel.feature_importances_)

	# pydotplus is better
	with open(resultFile, 'w') as fr:
		fr.write("Model, Accuracy, AUC\n")
		fr.write("Logistic Regression,"+str(st.mean(lgAccus))+','+str(st.mean(lgAUC))+'\n')
		fr.write("NB,"+str(st.mean(nbAccus))+','+str(st.mean(nbAUC))+'\n')
		fr.write("svm,"+str(st.mean(svmAccus))+','+'0'+'\n')
		fr.write("lsvm,"+str(st.mean(lsvmAccus))+','+'0'+'\n')
		fr.write("decision tree,"+str(st.mean(dtAccus))+','+str(st.mean(dtAUC))+'\n')
		fr.write("adaboost,"+str(st.mean(adaAccus))+','+str(st.mean(adaAUC))+'\n')
		fr.write("adaboost,"+str(st.mean(rfAccus))+','+str(st.mean(rfAUC))+'\n')

	# print(st.mean(dtAUC), st.mean(dtAUClog))
	# print(st.mean(lgAccus), st.mean(nbAccus), st.mean(svmAccus), st.mean(lsvmAccus), st.mean(dtAccus), st.mean(adaAccus), st.mean(rfAccus))
	return None
	# return st.mean(lgAccus)


# model: training and testing 
def modelCompareTesting(trainFile="../processedData/training_feature.csv", testFile="../processedData/testing_feature.csv", resultFile = "../result/proba_test2.csv"):
	print("model compare in testing")
	# data 
	training = readData(trainFile)
	testing= readData(testFile)
	Y_train = training[:,0]
	X_train = training[:,2:]
	Y_test = testing[:,0]
	X_test = testing[:,2:]
	eids = testing[:,1].tolist()

	modelAccu = dict()
	modelAUC = dict()
	# logistic regression
	# lgProbas = lm.LogisticRegression().fit(X_train, Y_train).predict_proba(X_test)

	# naive bayes
	# nbProbas = nb.GaussianNB().fit(X_train, Y_train).predict_proba(X_test)
	# neural network: python no

	# svm: rbf
	# svmProbas = svm.SVC(kernel='rbf').fit(X_train, Y_train).decistion_function(X_test))
	# lsvmProbas = lsvmAccus.append(svm.LinearSVC().fit(X_train, Y_train).score(X_test, Y_test))
	# decision tree
	# dfProbas = tree.DecisionTreeClassifier().fit(X_train, Y_train).predict_proba(X_test)

	# adaboost
	# adaProbas = tree.DecisionTreeClassifier().fit(X_train, Y_train).predict_proba(X_test)
	with open(resultFile, 'w') as fo:
		for index in range(len(eids)):
			fo.write(str(int(eids[index]))+','+str(dfProbas[index][1])+'\n')





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
	# modelCompareTesting()
	modelCompareTraining()
