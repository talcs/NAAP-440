import random

import numpy as np
from sklearn import linear_model


class LinearRegressionWithCustomActivation:
	def __init__(self, activation_func, inverse_activation_func):
		self.activation_func = activation_func
		self.inverse_activation_func = inverse_activation_func
		self.linear_regression = linear_model.LinearRegression()
		
	def fit(self, X, y):
		# Training the model to predict inverse-activated Y's
		# Such that the forward activation will be required to map predictions to the desired output range
		inverse_activated_y = self.inverse_activation_func(y)
		self.linear_regression.fit(X, inverse_activated_y)
		
	def predict(self, X):
		return self.activation_func(self.linear_regression.predict(X))

class LinearRegressionWithPolynomialActivation:
	def __init__(self, polynomial_degree):
		activation_func = lambda x : (x + 1) ** polynomial_degree
		inverse_activation_func = lambda x : x ** (1 / float(polynomial_degree)) - 1
		self.regressor = LinearRegressionWithCustomActivation(activation_func, inverse_activation_func)
		
	def fit(self, X, y):
		self.regressor.fit(X, y)
		
	def predict(self, X):
		return self.regressor.predict(X)
		
class LinearRegressionWithExpActivation:
	def __init__(self):
		activation_func = lambda x : np.exp(x)
		inverse_activation_func = lambda x : np.log(x)
		self.regressor = LinearRegressionWithCustomActivation(activation_func, inverse_activation_func)
		
	def fit(self, X, y):
		self.regressor.fit(X, y)
		
	def predict(self, X):
		return self.regressor.predict(X)
		
class LinearRegressionWithLogActivation:
	def __init__(self):
		activation_func = lambda x : np.log(x)
		inverse_activation_func = lambda x : np.exp(x)
		self.regressor = LinearRegressionWithCustomActivation(activation_func, inverse_activation_func)
		
	def fit(self, X, y):
		self.regressor.fit(X, y)
		
	def predict(self, X):
		return self.regressor.predict(X)
		
		
class LinearRegressionWithSigmoidActivation:
	def __init__(self):
		activation_func = lambda x : 1.0 / (1.0 + np.exp(-x))
		inverse_activation_func = lambda x : -np.log(1.0 / x - 1)
		self.regressor = LinearRegressionWithCustomActivation(activation_func, inverse_activation_func)
		
	def fit(self, X, y):
		self.regressor.fit(X, y)
		
	def predict(self, X):
		return self.regressor.predict(X)

		
class LinearRegressionOnRandomFeatureSubset:
	def __init__(self):
		self.linear_regression = linear_model.LinearRegression()
		
	def fit(self, X, y):
		if X.shape[1] > 8:
			sample_size = random.randint(min(3, X.shape[1]), X.shape[1]//2)
		else:
			sample_size = X.shape[1]
		self.feature_subset = random.sample(list(range(X.shape[1])), sample_size)
		X_subset = np.zeros((X.shape[0], sample_size))
		for i, feature in enumerate(self.feature_subset):
			X_subset[:,i] = X[:,feature]
		self.linear_regression.fit(X_subset, y)
		
	def predict(self, X):
		X_subset = np.zeros((X.shape[0], len(self.feature_subset)))
		for i, feature in enumerate(self.feature_subset):
			X_subset[:,i] = X[:,feature]
			
		return self.linear_regression.predict(X_subset)

class CustomVotingRegressor:
	def __init__(self, estimators, reduction = np.mean):
		self.estimators = estimators
		self.reduction = reduction
		
	def fit(self, X, y):
		for est in self.estimators:
			est.fit(X, y)
		
	def predict(self, X):
		predictions = [est.predict(X) for est in self.estimators]
		
		return self.reduction(predictions, axis = 0)

class CustomWeightedVotingRegressor:
	def __init__(self, estimator_ctors):
		self.estimators = []
		for estimator_ctor, num_est in estimator_ctors.items():
			for i in range(num_est):
				self.estimators.append(estimator_ctor())
		
	def fit(self, X, y):
		mses = []
		for est in self.estimators:
			est.fit(X, y)
			mses.append(np.linalg.norm(y - est.predict(X)))
		self.weights = (1.0 / np.array(mses))
		self.weights /= np.linalg.norm(self.weights, ord = 1)
		
	def predict(self, X):
		predictions = np.array([est.predict(X) for est in self.estimators])
		
		return self.weights @ predictions

class PerturbatedLinearRegressorEnsemble:
	def __init__(self, feature_noise_dist = (0, 0.3), y_noise_dist = (0, 0.002), num_regressors = 70, reduction = np.mean):
		self.models = []
		for i in range(num_regressors):
			self.models.append(linear_model.LinearRegression())
		self.feature_noise_dist = feature_noise_dist
		self.y_noise_dist = y_noise_dist
		self.reduction = reduction
		
	def fit(self, X, y):
		for model in self.models:
			mX = X + np.random.normal(*self.feature_noise_dist, X.shape)
			mY = y + np.random.normal(*self.y_noise_dist, y.shape)
			
			model.fit(mX, mY)
		
	def predict(self, X):
		predictions = [model.predict(X) for model in self.models]
		
		return self.reduction(predictions, axis = 0)