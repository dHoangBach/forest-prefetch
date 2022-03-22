import sys
import csv
import numpy as np
import os.path
import json
import timeit

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

sys.path.append('../../code/')
import Forest
import Tree

def testModel(roundSplit,XTrain,YTrain,XTest,YTest,model,name):
	print("Fitting", name)
	model.fit(XTrain,YTrain)

	print("Testing ", name)
	start = timeit.default_timer()
	YPredicted = model.predict(XTest)
	end = timeit.default_timer()

	print("Total time: " + str(end - start) + " ms")
	print("Throughput: " + str(len(XTest) / (float(end - start)*1000)) + " #elem/ms")

	print("Saving model")
	if (issubclass(type(model), DecisionTreeClassifier)):
		mymodel = Tree.Tree()
	else:
		mymodel = Forest.Forest()

	mymodel.fromSKLearn(model,roundSplit)

	if not os.path.exists("text"):
		os.makedirs("text")

	with open("text/"+name+".json",'w') as outFile:
		outFile.write(mymodel.str())

	SKPred = model.predict(XTest)
	MYPred = mymodel.predict_batch(XTest)
	accuracy = accuracy_score(YTest, SKPred)
	print("Accuracy:", accuracy)

	# This can now happen because of classical majority vote
	# for (skpred, mypred) in zip(SKPred,MYPred):
	# 	if (skpred != mypred):
	# 		print("Prediction mismatch!!!")
	# 		print(skpred, " vs ", mypred)

	print("Saving model to PKL on disk")
	joblib.dump(model, "text/"+name+".pkl")

	print("*** Summary ***")
	print("#Examples\t #Features\t Accuracy\t Avg.Tree Height")
	print(str(len(XTest)) + "\t" + str(len(XTest[0])) + "\t" + str(accuracy) + "\t" + str(mymodel.getAvgDepth()))
	print()

def fitModels(roundSplit,XTrain,YTrain,XTest = None,YTest = None,createTest = False):
	if XTest is None or YTest is None:
		XTrain,XTest,YTrain,YTest = train_test_split(XTrain, YTrain, test_size=0.25)
		createTest = True

	if createTest:
		with open("test.csv", 'w') as outFile:
			for x,y in zip(XTest, YTest):
				line = str(y)
				for xi in x:
					line += "," + str(xi)

				outFile.write(line + "\n")

	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=8,max_depth=5),"DT_5_1")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=8,max_depth=10),"DT_10_1")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=8,max_depth=15),"DT_15_1")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=8,max_depth=20),"DT_20_1")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=8,max_depth=25),"DT_25_1")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=1,n_jobs=8,max_depth=30),"DT_30_1")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=8,max_depth=5),"RF_5_25")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=8,max_depth=10),"RF_10_25")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=8,max_depth=15),"RF_15_25")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=8,max_depth=20),"RF_20_25")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=8,max_depth=25),"RF_25_25")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=25,n_jobs=8,max_depth=30),"RF_30_25")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=50,n_jobs=8,max_depth=5),"RF_5_50")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=50,n_jobs=8,max_depth=10),"RF_10_50")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=50,n_jobs=8,max_depth=15),"RF_15_50")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=50,n_jobs=8,max_depth=20),"RF_20_50")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=50,n_jobs=8,max_depth=25),"RF_25_50")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=50,n_jobs=8,max_depth=30),"RF_30_50")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=100,n_jobs=8,max_depth=5),"RF_5_100")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=75,n_jobs=8,max_depth=5),"RF_5_75")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=75,n_jobs=8,max_depth=10),"RF_10_75")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=75,n_jobs=8,max_depth=15),"RF_15_75")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=75,n_jobs=8,max_depth=20),"RF_20_75")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=75,n_jobs=8,max_depth=25),"RF_25_75")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=75,n_jobs=8,max_depth=30),"RF_30_75")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=100,n_jobs=8,max_depth=10),"RF_10_100")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=100,n_jobs=8,max_depth=15),"RF_15_100")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=100,n_jobs=8,max_depth=20),"RF_20_100")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=100,n_jobs=8,max_depth=25),"RF_25_100")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=100,n_jobs=8,max_depth=30),"RF_30_100")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=125,n_jobs=8,max_depth=5),"RF_5_125")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=125,n_jobs=8,max_depth=10),"RF_10_125")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=125,n_jobs=8,max_depth=15),"RF_15_125")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=125,n_jobs=8,max_depth=20),"RF_20_125")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=125,n_jobs=8,max_depth=25),"RF_25_125")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=125,n_jobs=8,max_depth=30),"RF_30_125")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=150,n_jobs=8,max_depth=5),"RF_5_150")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=150,n_jobs=8,max_depth=10),"RF_10_150")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=150,n_jobs=8,max_depth=15),"RF_15_150")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=150,n_jobs=8,max_depth=20),"RF_20_150")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=150,n_jobs=8,max_depth=25),"RF_25_150")
	testModel(roundSplit,XTrain,YTrain,XTest,YTest,RandomForestClassifier(n_estimators=150,n_jobs=8,max_depth=30),"RF_30_150")