"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
from regression import logreg, utils
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def test_prediction():
	"""Test prediction
	Compute predictions and see if the MSE is somewhat decent
	"""
	np.random.seed(42)
	X_train, X_test, y_train, y_test = utils.loadDataset(split_percent=0.8)
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform (X_test)

	lr = logreg.LogisticRegressor(num_feats = 6) 
	lr.train_model(X_train, y_train, X_test, y_test)

	X_test = np.column_stack((X_test, np.ones(X_test.shape[0]))) 

	# Predict value and calculate MSE
	y_pred = lr.make_prediction(X_test)
	MSE = np.mean((y_pred-y_test)**2)

	# Ideally less than 0.5
	assert MSE < 0.5 

def test_loss_function():
	""" Test loss function
	Mnaually calculated values and testing to see if it's about the same
	"""
	np.random.seed(42)
	y_pred = np.array([0.5, 0.5, 0.5, 0.5])
	y_true = np.array([0, 1, 1, 0])
	lr = logreg.LogisticRegressor(num_feats = 4)
	approx = round(lr.loss_function(y_true, y_pred), 2)
	assert (approx == 0.69)

def test_gradient():
	""" Test Gradient
	Example values from: 
	https://www.youtube.com/watch?v=UP7tehNv-iI
	"""
	lr = logreg.LogisticRegressor(num_feats = 4)
	# Manually initialize W value
	lr.W = np.array([0,0,0])
	X_test = np.array([[3, 2, 1],
                  [3, 2, 1]])
	y_test = np.array([1, 1])
	gradient = lr.calculate_gradient(y_test, X_test) 
	assert np.all(gradient == np.array([-1.5, -1.0, -.5]))   

def test_training():
	""" Test training of model and make sure that weights are being updated
	"""
	np.random.seed(42)
	X_train, X_test, y_train, y_test = utils.loadDataset(split_percent=0.8)
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	lr = logreg.LogisticRegressor(num_feats = 6) 
	lr.train_model(X_train, y_train, X_test, y_test)

	assert (len(lr.loss_hist_train) == len(lr.loss_hist_val))
	pass