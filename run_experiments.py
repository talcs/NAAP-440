import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import preprocessing


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
	
	return num_violations, max_violations, 1 - num_violations / float(max_violations)

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
	num_violations, max_violations, ms = compute_monotonicity_score(Y_test)
	# Maybe draw the pairs
	
	return mae, num_violations, max_violations, ms
	

def run_experiments(dataset):
	trainset = dataset[dataset.IsTest == 0]
	testset = dataset[dataset.IsTest == 1]
	
	scheme_features = ['NumParams', 'NumMacs', 'NumLayers', 'NumStages', 'FirstLayerWidth', 'LastLayerWidth']
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
		('2-NN', lambda : KNeighborsRegressor(n_neighbors = 2, algorithm = 'brute')),
		('3-NN', lambda : KNeighborsRegressor(n_neighbors = 3, algorithm = 'brute')),
		('4-NN', lambda : KNeighborsRegressor(n_neighbors = 4, algorithm = 'brute')),
		('5-NN', lambda : KNeighborsRegressor(n_neighbors = 5, algorithm = 'brute')),
		('Linear Regression', lambda : linear_model.LinearRegression()),
		('Decision Treee', lambda : tree.DecisionTreeRegressor()),
		('Random Forest,N=100', lambda : ensemble.RandomForestRegressor(n_estimators=100)),
		('MLP', lambda : neural_network.MLPRegressor(hidden_layer_sizes = (30, 30))),
	]
	for model_name, model_ctor in models:
		print(f'Model: {model_name}:')
		print(f'---------------')
		for feature_set_name, feature_set in feature_sets:
			mae, num_violations, max_violations, ms = run_experiment(trainset, testset, model_ctor, feature_set)
			mae = round(mae, 4)
			print(f'{feature_set_name}: MAE: {mae}, Monotonicity Score: {ms}, {num_violations} of {max_violations} violations')
				

if __name__ == '__main__':
	dataset_path = sys.argv[1]
	
	dataset = pd.read_csv(dataset_path)
	run_experiments(dataset)
	