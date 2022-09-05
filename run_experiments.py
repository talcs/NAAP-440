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
from sklearn import kernel_ridge
from sklearn import preprocessing


SEED = 20220905
random.seed(SEED)
np.random.seed(SEED)

DEBUG = False

LOG_PARAMS_AND_MACS = True

compute_mae = lambda predictions, ground_truth : np.mean(np.abs(predictions - ground_truth))

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

def run_experiment(trainset, testset, model_ctor, features):
	model = model_ctor()
	X_train = trainset[features].astype('float64')
	Y_train = trainset.MaxAccuracy
	X_test = testset[features].astype('float64')
	
	scaler = preprocessing.StandardScaler().fit(X_train)
	model.fit(scaler.transform(X_train), Y_train)
	Y_test = model.predict(scaler.transform(X_test))
	
	GT_test = testset.MaxAccuracy
	
	mae = compute_mae(Y_test, GT_test)
	num_violations, max_violations, ms, sorting = compute_monotonicity_score(Y_test)
	
	return mae, num_violations, max_violations, ms, sorting
	

def run_experiments(dataset, output_dir):
	output_csv = None
	figures_dir = None
	if output_dir is not None:
		os.mkdir(output_dir)
		output_csv = os.path.join(output_dir, 'results.csv')
		figures_dir = os.path.join(output_dir, 'figures')
		os.mkdir(figures_dir)
	if LOG_PARAMS_AND_MACS:
		dataset['NumParams'] = np.log(dataset['NumParams'])
		dataset['NumMACs'] = np.log(dataset['NumMACs'])
	trainset = dataset[dataset.IsTest == 0]
	testset = dataset[dataset.IsTest == 1]
	
	scheme_features = ['NumParams', 'NumMACs', 'NumLayers', 'NumStages', 'FirstLayerWidth', 'LastLayerWidth']
	quantitative_features = []
	for epoch in range(1,18+1):
		quantitative_features += [f'e{epoch}{metric}' for metric in ('LossMean','LossMedian','Accuracy')]
	qfeatures_3 = quantitative_features[:3*3]
	qfeatures_6 = quantitative_features[:6*3]
	qfeatures_9 = quantitative_features[:9*3]
	qfeatures_12 = quantitative_features[:12*3]
	qfeatures_15 = quantitative_features[:15*3]
	qfeatures_18 = quantitative_features[:18*3]
	feature_sets = [
		('Scheme', scheme_features),
		('Quantitative 3 epochs', qfeatures_3),
		('Quantitative 6 epochs', qfeatures_6),
		('Quantitative 9 epochs', qfeatures_9),
		('Quantitative 12 epochs', qfeatures_12),
		('Quantitative 15 epochs', qfeatures_15),
		('Quantitative 18 epochs', qfeatures_18),
		('Scheme + Quantitative 3 epochs', scheme_features + qfeatures_3),
		('Scheme + Quantitative 6 epochs', scheme_features + qfeatures_6),
		('Scheme + Quantitative 9 epochs', scheme_features + qfeatures_9),
		('Scheme + Quantitative 12 epochs', scheme_features + qfeatures_12),
		('Scheme + Quantitative 15 epochs', scheme_features + qfeatures_15),
		('Scheme + Quantitative 18 epochs', scheme_features + qfeatures_18),
	]
	models = [
		('1-NN', lambda : KNeighborsRegressor(n_neighbors = 1, algorithm = 'brute')),
		('3-NN', lambda : KNeighborsRegressor(n_neighbors = 3, algorithm = 'brute')),
		('5-NN', lambda : KNeighborsRegressor(n_neighbors = 5, algorithm = 'brute')),
		('7-NN', lambda : KNeighborsRegressor(n_neighbors = 7, algorithm = 'brute')),
		('9-NN', lambda : KNeighborsRegressor(n_neighbors = 9, algorithm = 'brute')),
		('Linear Regression', lambda : linear_model.LinearRegression()),
		('Decision Treee', lambda : tree.DecisionTreeRegressor()),
		('Gradient Boosting,N=50', lambda : ensemble.GradientBoostingRegressor(n_estimators = 50)),
		('Gradient Boosting,N=100', lambda : ensemble.GradientBoostingRegressor(n_estimators = 100)),
		('Gradient Boosting,N=200', lambda : ensemble.GradientBoostingRegressor(n_estimators = 200)),
		('AdaBoost,N=50', lambda : ensemble.AdaBoostRegressor(n_estimators = 50)),
		('AdaBoost,N=100', lambda : ensemble.AdaBoostRegressor(n_estimators = 100)),
		('AdaBoost,N=200', lambda : ensemble.AdaBoostRegressor(n_estimators = 200)),
		('SVR (RBF kernel)', lambda : svm.SVR(kernel = 'rbf')),
		('SVR (Polynimial kernel)', lambda : svm.SVR(kernel = 'poly')),
		('SVR (Linear kernel)', lambda : svm.SVR(kernel = 'linear')),
		('Kernel Ridge', lambda : kernel_ridge.KernelRidge()),
		('Random Forest,N=50', lambda : ensemble.RandomForestRegressor(n_estimators=50)),
		('Random Forest,N=100', lambda : ensemble.RandomForestRegressor(n_estimators=100)),
		('Random Forest,N=200', lambda : ensemble.RandomForestRegressor(n_estimators=200)),
	]
	if DEBUG:
		feature_sets = feature_sets[:1]
		models = models[:1]
	if output_csv:
		with open(output_csv, 'w') as f:
			f.write('Model,Features,MeanAbsoluteError,NumViolations,MaxViolations,MonotonicityScore\n')
	for model_name, model_ctor in models:
		if not output_csv:
			print(f'Model: {model_name}:')
			print(f'---------------')
		for feature_set_name, feature_set in feature_sets:
			mae, num_violations, max_violations, ms, sorting = run_experiment(trainset, testset, model_ctor, feature_set)
			mae = round(mae, 4)
			ms = round(ms, 4)
			if not output_csv:
				print(f'{feature_set_name}: MAE: {mae}, Monotonicity Score: {ms}, {num_violations} of {max_violations} violations')
			if output_csv:
				with open(output_csv, 'a') as f:
					f.write(f'{model_name},{feature_set_name},{mae},{num_violations},{max_violations},{ms}\n')
			if figures_dir:
				import matplotlib.pyplot as plt
				conf_matrix = np.zeros((len(testset), len(testset)))
				for i, sorted_index in enumerate(sorting):
					conf_matrix[sorted_index,i] = (1 - abs(sorted_index - i) / float(len(sorting)))**2
				plt.imshow(conf_matrix, cmap = 'plasma', origin = 'lower')
				plt.xlabel('Model Index')
				plt.ylabel('Predicted Model Index')
				plt.savefig(os.path.join(figures_dir, f'{model_name}_{feature_set_name}.jpg'), bbox_inches = 'tight')
				
if __name__ == '__main__':
	dataset_path = sys.argv[1]
	output_csv = sys.argv[2] if len(sys.argv) > 2 else None
	
	dataset = pd.read_csv(dataset_path)
	run_experiments(dataset, output_csv)
	