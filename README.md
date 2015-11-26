# stacked_generalization
Python implementation of stacked generalization classifier, as described [here](http://machine-learning.martinsewell.com/ensembles/stacking/). 

Plays nice with sklearn classifiers, or any model class that has a `fit` and `predict` method.

# Example usage

	from sklearn.datasets import load_digits
	from stacked_generalizer import StackedGeneralizer
	from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
	from sklearn.linear_model import LogisticRegression
	import numpy as np

	VERBOSE = True
	N_FOLDS = 5
	
	# load data and shuffle observations
	data = load_digits()

	X = data.data
	y = data.target

	shuffle_idx = np.random.permutation(y.size)

	X = X[shuffle_idx]
	y = y[shuffle_idx]

	# hold out 20 percent of data for testing accuracy
	train_prct = 0.8
	n_train = round(X.shape[0]*train_prct)

	# define base models
	base_models = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
	               RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
	               ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini')]

	# define blending model
	blending_model = LogisticRegression()

	# initialize multi-stage model
	sg = StackedGeneralizer(base_models, blending_model, 
		                    n_folds=N_FOLDS, verbose=VERBOSE)

	# fit model
	sg.fit(X[:n_train],y[:n_train])

	# test accuracy
	pred = sg.predict(X[n_train:])
	pred_classes = [np.argmax(p) for p in pred]

	_ = sg.evaluate(y[n_train:], pred_classes)

                 precision    recall  f1-score   support

	          0       0.97      1.00      0.99        33
	          1       0.97      1.00      0.99        38
	          2       1.00      1.00      1.00        42
	          3       1.00      0.98      0.99        41
	          4       0.97      0.94      0.95        32
	          5       0.95      0.98      0.96        41
	          6       1.00      0.95      0.97        37
	          7       0.94      0.97      0.96        34
	          8       0.94      0.94      0.94        34
	          9       0.96      0.96      0.96        27

	avg / total       0.97      0.97      0.97       359