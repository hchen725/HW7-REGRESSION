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
from sklearn.linear_model import LogisticRegression 

def test_prediction():
	pass

def test_loss_function():
	# Create a prediction and true value and compare to manually calculated values
	y_pred = np.array([0.5, 0.5, 0.5, 0.5])
	y_true = np.array([0, 1, 1, 0])
	lr = logreg.LogisticRegressor(num_feats = 4)
	approx = round(lr.loss_function(y_true, y_pred), 2)
	
	assert (approx == 0.69)

def test_gradient():
	pass

def test_training():
	pass

def test_compare_scikit():
	# # Train using sklearn and compare results
	# X_train, X_test, y_train, y_test = utils.loadDataset(split_percent = 0.8)
	# # Test sklearn
	# sklearn_lr = LogisticRegression(C = 0.01)
	# sklearn_lr.fit(X_train, y_train)
	# sklearn_pred = sklean_lr.predict(X_test)

	pass