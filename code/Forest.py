import numpy as np
import json
from sklearn.tree import _tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

import Tree

class Forest:
	def __init__(self):
		self.trees = []

	def fromSKLearn(self,forest,roundSplit = False):

		if (issubclass(type(forest), AdaBoostClassifier)):
			sumW = sum([w for w in forest.estimator_weights_])

			# https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/weight_boosting.py#L297
			# see: decision_function
			if (forest.algorithm == "SAMME"):
				for e,w in zip(forest.estimators_,forest.estimator_weights_):
					tree = Tree.Tree()
					tree.fromSKLearn(e, roundSplit, "SAMME", w/sumW)
					self.trees.append(tree)
			else:
				for e in forest.estimators_:
					tree = Tree.Tree()
					tree.fromSKLearn(e, roundSplit, "SAMME.R", 1.0/sumW)
					self.trees.append(tree)
		elif (issubclass(type(forest), RandomForestClassifier)) or (issubclass(type(forest), ExtraTreesClassifier)):
			for e in forest.estimators_:
					tree = Tree.Tree()
					tree.fromSKLearn(e, roundSplit, "RandomForest", 1.0/len(forest.estimators_))
					self.trees.append(tree)
		else:
			raise NotImplementedError("fromSKLearn() is not implemented for class ", type(forest))

		
	def fromJSON(self, jsonFile):
		with open(jsonFile) as data_file:    
			data = json.load(data_file)

		for x in data:
			tree = Tree.Tree()
			tree.fromJSON(x)

			self.trees.append(tree)

	def str(self):
		s = "["
		for tree in self.trees:
			s += tree.str() + ","
		s = s[:-1] + "]"
		return s

	def pstr(self):
		parsed = json.loads(self.str())
		return json.dumps(parsed, indent=4)

	## SOME STATISTICS FUNCTIONS ##

	def getSubTrees(self, minProb, maxNumNodes):
		subTrees = []
		for t in self.trees:
			subTree, prob, size = t.getSubTree(minProb,maxNumNodes)
			subTrees.append(subTree)
		return subTrees

	def getAvgDepth(self):
		return sum([t.getAvgDepth() for t in self.trees]) / len(self.trees)

	def getTotalNumNodes(self):
		return sum([t.getNumNodes() for t in self.trees])

	def predict(self,x):
		pred = [0 for i in range(self.getNumClasses())]

		for t in self.trees:
			c = t.predict(x)
			pred[c] += 1

		return np.argmax(pred)


	def predict_batch(self,X):
		YPred = []

		for x in X:
			pred = [0 for i in range(self.getNumClasses())]

			for t in self.trees:
				c = t.predict(x)
				pred[c] += 1

			YPred.append(np.argmax(pred))

		return YPred

	def getNumClasses(self):
		return self.trees[0].getNumClasses()