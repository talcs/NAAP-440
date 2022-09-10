import sys
import os
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import ensemble
from sklearn import svm
from sklearn import preprocessing


SEEDS = [20220905 + i for i in range(5)]

PRODUCE_FIGURES = True
ABLATION_STUDY = False

LOG_PARAMS_AND_MACS = True

compute_mae = lambda predictions, ground_truth : np.mean(np.abs(predictions - ground_truth))


class LinearRegressionWithPolynomialActivation:
	def __init__(self, polynomial_degree):
		self.polynomial_degree = polynomial_degree
		self.linear_regression = linear_model.LinearRegression()
		
	def fit(self, X, y):
		# Polynomial activation = (InnerProduct + 1)^dgree
		pol_y = y ** (1 / float(self.polynomial_degree)) - 1
		self.linear_regression.fit(X, pol_y)
		
	def predict(self, X):
		return (self.linear_regression.predict(X) + 1) ** self.polynomial_degree


def set_random_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	

def compute_monotonicity_score(predictions):
	# The dataset is already sorted by MaxAccuracy
	sorting = sorted(list(range(len(predictions))), key = lambda x : predictions[x])
	num_violations = 0
	for i, scheme_index in enumerate(sorting):
		for j in range(i):
			if sorting[j] > scheme_index:
				num_violations += 1
	max_violations = (len(predictions) * (len(predictions) - 1)) // 2
	
	return num_violations, max_violations, 1 - num_violations / float(max_violations), sorting

def find_median_index(entries, keyfield):
	indices = list(range(len(entries)))
	indices.sort(key = lambda x : entries[x][keyfield])
	
	return indices[len(indices) // 2]
	

def run_experiment(trainset, testset, model_ctor, features, seeds = SEEDS):
	model = model_ctor()
	X_train = trainset[features].astype('float64')
	Y_train = trainset.MaxAccuracy
	X_test = testset[features].astype('float64')
	
	GT_test = testset.MaxAccuracy
	
	seed_results = []	
	for seed in seeds:
		set_random_seed(seed)
		scaler = preprocessing.StandardScaler().fit(X_train)
		model.fit(scaler.transform(X_train), Y_train)
		Y_test = model.predict(scaler.transform(X_test))	
	
		mae = compute_mae(Y_test, GT_test)
		num_violations, max_violations, ms, sorting = compute_monotonicity_score(Y_test)
		seed_results.append({'mae' : mae, 'num_violations' : num_violations, 'max_violations' : max_violations, 'ms' : ms, 'sorting' : sorting})
	
	median_index = find_median_index(seed_results, 'num_violations')
		
	return seed_results[median_index]
	

def run_experiments(dataset, output_dir):
	output_csv = None
	figures_dir = None
	if output_dir is not None:
		os.mkdir(output_dir)
		output_csv = os.path.join(output_dir, 'results.csv')
		if PRODUCE_FIGURES:
			figures_dir = os.path.join(output_dir, 'figures')
			os.mkdir(figures_dir)
	dataset = dataset.sort_values('MaxAccuracy')
	if LOG_PARAMS_AND_MACS:
		dataset['NumParams'] = np.log(dataset['NumParams'])
		dataset['NumMACs'] = np.log(dataset['NumMACs'])
	trainset = dataset[dataset.IsTest == 0]
	testset = dataset[dataset.IsTest == 1]
	
	scheme_features = ['NumParams', 'NumMACs', 'NumLayers', 'NumStages', 'FirstLayerWidth', 'LastLayerWidth']
	scheme_limited_features = ['NumParams', 'NumStages']
	quantitative_features = []
	for epoch in range(1,18+1):
		quantitative_features += [f'e{epoch}{metric}' for metric in ('LossMean','LossMedian','Accuracy')]
	qfeatures_3 = quantitative_features[:3*3]
	qfeatures_6 = quantitative_features[:6*3]
	qfeatures_9 = quantitative_features[:9*3]
	qfeatures_12 = quantitative_features[:12*3]
	qfeatures_15 = quantitative_features[:15*3]
	qfeatures_18 = quantitative_features[:18*3]
	
	# Notes:
	# As described in the paper, the ablation studies showed that:
	#  1. When only using scheme features (no features from epochs), the limited set of features (NumParams + NumStages) works best
	#  2. When combining the scheme features with features from the epochs, using the full set of scheme features usually works best
	feature_sets = []
	if ABLATION_STUDY:
		# Ablation study on scheme features...
		feature_sets.extend([
			('SchemeFull', scheme_features),
			('SchemeFull\\NumParams', scheme_features[1:]),
			('SchemeFull\\NumMACs', scheme_features[:1] + scheme_features[2:]),
			('SchemeFull\\NumLayers', scheme_features[:2] + scheme_features[3:]),
			('SchemeFull\\NumStages', scheme_features[:3] + scheme_features[4:]),
			('SchemeFull\\FirstLayerWidth', scheme_features[:4] + scheme_features[5:]),
			('SchemeFull\\LastLayerWidth', scheme_features[:5]),
		])
	# Only feature sets for the baseline report
	feature_sets.extend([
		('Scheme' , scheme_limited_features),		
		('Scheme + Quantitative 3 epochs', scheme_features + qfeatures_3),
		('Scheme + Quantitative 6 epochs', scheme_features + qfeatures_6),
		('Scheme + Quantitative 9 epochs', scheme_features + qfeatures_9),
	])
	if ABLATION_STUDY:
		# Ablation study on combination of scheme features and features from training
		feature_sets.extend([
			('Scheme + Quantitative 12 epochs', scheme_features + qfeatures_12),
			('Scheme + Quantitative 15 epochs', scheme_features + qfeatures_15),
			('Scheme + Quantitative 18 epochs', scheme_features + qfeatures_18),
			('SchemeLimited + Quantitative 3 epochs', scheme_limited_features + qfeatures_3),
			('SchemeLimited + Quantitative 6 epochs', scheme_limited_features + qfeatures_6),
			('SchemeLimited + Quantitative 9 epochs', scheme_limited_features + qfeatures_9),
			('SchemeLimited + Quantitative 12 epochs', scheme_limited_features + qfeatures_12),
			('SchemeLimited + Quantitative 15 epochs', scheme_limited_features + qfeatures_15),
			('SchemeLimited + Quantitative 18 epochs', scheme_limited_features + qfeatures_18),
			('Quantitative 3 epochs', qfeatures_3),
			('Quantitative 6 epochs', qfeatures_6),
			('Quantitative 9 epochs', qfeatures_9),
			('Quantitative 12 epochs', qfeatures_12),
			('Quantitative 15 epochs', qfeatures_15),
			('Quantitative 18 epochs', qfeatures_18),
		])
	models = [
		('1-NN', lambda : KNeighborsRegressor(n_neighbors = 1, algorithm = 'brute')),
		('3-NN', lambda : KNeighborsRegressor(n_neighbors = 3, algorithm = 'brute')),
		('5-NN', lambda : KNeighborsRegressor(n_neighbors = 5, algorithm = 'brute')),
		('7-NN', lambda : KNeighborsRegressor(n_neighbors = 7, algorithm = 'brute')),
		('9-NN', lambda : KNeighborsRegressor(n_neighbors = 9, algorithm = 'brute')),
		('Linear Regression', lambda : linear_model.LinearRegression()),
		('Linear Regression D=0.5', lambda : LinearRegressionWithPolynomialActivation(0.5)),
		('Linear Regression D=0.25', lambda : LinearRegressionWithPolynomialActivation(0.25)),
		('Decision Tree', lambda : tree.DecisionTreeRegressor()),
		('Gradient Boosting (N=25)', lambda : ensemble.GradientBoostingRegressor(n_estimators = 25)),
		('Gradient Boosting (N=50)', lambda : ensemble.GradientBoostingRegressor(n_estimators = 50)),
		('Gradient Boosting (N=100)', lambda : ensemble.GradientBoostingRegressor(n_estimators = 100)),
		('Gradient Boosting (N=200)', lambda : ensemble.GradientBoostingRegressor(n_estimators = 200)),
		('AdaBoost (N=25)', lambda : ensemble.AdaBoostRegressor(n_estimators = 25)),
		('AdaBoost (N=50)', lambda : ensemble.AdaBoostRegressor(n_estimators = 50)),
		('AdaBoost (N=100)', lambda : ensemble.AdaBoostRegressor(n_estimators = 100)),
		('AdaBoost (N=200)', lambda : ensemble.AdaBoostRegressor(n_estimators = 200)),
		('SVR (RBF kernel)', lambda : svm.SVR(kernel = 'rbf')),
		('SVR (Polynomial kernel)', lambda : svm.SVR(kernel = 'poly')),
		('SVR (Linear kernel)', lambda : svm.SVR(kernel = 'linear')),
		('Random Forest (N=25)', lambda : ensemble.RandomForestRegressor(n_estimators=25)),
		('Random Forest (N=50)', lambda : ensemble.RandomForestRegressor(n_estimators=50)),
		('Random Forest (N=100)', lambda : ensemble.RandomForestRegressor(n_estimators=100)),
		('Random Forest (N=200)', lambda : ensemble.RandomForestRegressor(n_estimators=200)),
	]
	if output_csv:
		with open(output_csv, 'w') as f:
			f.write('Model,Features,MeanAbsoluteError,NumViolations,MaxViolations,MonotonicityScore\n')
	for model_name, model_ctor in models:
		if not output_csv:
			print(f'Model: {model_name}:')
			print(f'---------------')
		for feature_set_name, feature_set in feature_sets:
			result = run_experiment(trainset, testset, model_ctor, feature_set, SEEDS)
			result['mae'] = round(result['mae'], 4)
			result['ms'] = round(result['ms'], 4)
			if not output_csv:
				print(f'{feature_set_name}: MAE: {result["mae"]}, Monotonicity Score: {result["ms"]}, {result["num_violations"]} of {result["max_violations"]} violations')
			if output_csv:
				with open(output_csv, 'a') as f:
					f.write(f'{model_name},{feature_set_name},{result["mae"]},{result["num_violations"]},{result["max_violations"]},{result["ms"]}\n')
			if figures_dir:
				import matplotlib.pyplot as plt
				conf_matrix = np.zeros((len(testset), len(testset)))
				for i, sorted_index in enumerate(result["sorting"]):
					conf_matrix[sorted_index,i] = (1 - abs(sorted_index - i) / float(len(result["sorting"])))**2
				plt.imshow(conf_matrix, cmap = 'plasma', origin = 'lower')
				plt.xlabel('Model Index')
				plt.ylabel('Predicted Model Index')
				plt.savefig(os.path.join(figures_dir, f'{model_name}_{feature_set_name}.jpg'), bbox_inches = 'tight')
				
if __name__ == '__main__':
	dataset_path = sys.argv[1]
	output_csv = sys.argv[2] if len(sys.argv) > 2 else None
	
	dataset = pd.read_csv(dataset_path)
	run_experiments(dataset, output_csv)
	