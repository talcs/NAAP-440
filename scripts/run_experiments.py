import sys
import os
import random

import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn import preprocessing

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_regressors import LinearRegressionWithPolynomialActivation, LinearRegressionOnRandomFeatureSubset, CustomVotingRegressor
from custom_regressors import LinearRegressionWithExpActivation, LinearRegressionWithLogActivation, CustomWeightedVotingRegressor
from custom_regressors import PerturbatedLinearRegressorEnsemble, LinearRegressionWithSigmoidActivation


SEEDS = [20220905 + i for i in range(5)]

PRODUCE_FIGURES = True
ABLATION_STUDY = False

LOG_PARAMS_AND_MACS = True

compute_mae = lambda predictions, ground_truth : np.mean(np.abs(predictions - ground_truth))


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
	

def run_experiment(trainset, testset, model_ctor, features, seeds = SEEDS, add_random_noise_to_predictions = False):
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
		if add_random_noise_to_predictions:
			# Adding a negligible Gaussian noise to the predictions,
			# used to strongly penalize settings that predict the same value to many samples (penalize on violations)
			Y_test += np.random.normal(0, 1e-5, size = Y_test.shape)	
		mae = compute_mae(Y_test, GT_test)
		num_violations, max_violations, ms, sorting = compute_monotonicity_score(Y_test)
		seed_results.append({'mae' : mae, 'num_violations' : num_violations, 'max_violations' : max_violations, 'ms' : ms, 'sorting' : sorting, 'pred' : Y_test})
	
	median_index = find_median_index(seed_results, 'num_violations')
		
	return seed_results[median_index]


def create_confusion_matrix(sorting, save_path):
	import matplotlib.pyplot as plt
	plt.clf()
	conf_matrix = np.zeros((len(sorting), len(sorting)))
	for i, sorted_index in enumerate(sorting):
		conf_matrix[sorted_index,i] = (1 - abs(sorted_index - i) / float(len(sorting)))**2
	plt.imshow(conf_matrix, cmap = 'plasma', origin = 'lower')
	plt.xlabel('Model Index')
	plt.ylabel('Predicted Model Index')
	plt.savefig(save_path, bbox_inches = 'tight')
	
def create_scatter_plot(gt_series, pred_series, save_path):
	import matplotlib.pyplot as plt
	plt.clf()
	plt.scatter(range(1, len(gt_series) + 1), gt_series, label = 'GT accuracy')
	plt.scatter(range(1, len(pred_series) + 1), pred_series, label = 'Predicted accuracy')
	plt.xlabel('Model Index')
	plt.ylabel('Accuracy on CIFAR10 test set')
	rmin_y_value = round(min(min(gt_series), min(pred_series)), 2)
	rmax_y_value = round(max(max(gt_series), max(pred_series)), 2)
	plt.yticks(np.arange(rmin_y_value, rmax_y_value+0.02, 0.03))
	plt.grid()
	plt.legend()
	plt.savefig(save_path, bbox_inches = 'tight')
	
def get_feature_name_groups():
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
	
	feature_name_groups = (scheme_features, scheme_limited_features, quantitative_features,
							qfeatures_3, qfeatures_6, qfeatures_9, qfeatures_12, qfeatures_15, qfeatures_18)
	
	
	return feature_name_groups
	
def get_feature_sets(feature_name_groups):
	(scheme_features, scheme_limited_features, quantitative_features,
	qfeatures_3, qfeatures_6, qfeatures_9, qfeatures_12, qfeatures_15, qfeatures_18) = feature_name_groups
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
		
	return feature_sets
	
def get_regression_models(extended = False):
	models = [
		('1-NN', lambda : neighbors.KNeighborsRegressor(n_neighbors = 1, algorithm = 'brute'), {'interpolation'}),
		('3-NN', lambda : neighbors.KNeighborsRegressor(n_neighbors = 3, algorithm = 'brute'), {'interpolation'}),
		('5-NN', lambda : neighbors.KNeighborsRegressor(n_neighbors = 5, algorithm = 'brute'), {'interpolation'}),
		('7-NN', lambda : neighbors.KNeighborsRegressor(n_neighbors = 7, algorithm = 'brute'), {'interpolation'}),
		('9-NN', lambda : neighbors.KNeighborsRegressor(n_neighbors = 9, algorithm = 'brute'), {'interpolation'}),
		('Linear Regression', lambda : linear_model.LinearRegression(), {'interpolation', 'dual_extrapolation', 'left_extrapolation', 'right_extrapolation'}),
		('Linear Regression (D=0.5)', lambda : LinearRegressionWithPolynomialActivation(0.5), {'interpolation', 'dual_extrapolation', 'left_extrapolation', 'right_extrapolation'}),
		('Linear Regression (D=0.25)', lambda : LinearRegressionWithPolynomialActivation(0.25), {'interpolation', 'dual_extrapolation', 'left_extrapolation', 'right_extrapolation'}),
		('Decision Tree', lambda : tree.DecisionTreeRegressor(), {'interpolation'}),
		('Gradient Boosting (N=25)', lambda : ensemble.GradientBoostingRegressor(n_estimators = 25), {'interpolation'}),
		('Gradient Boosting (N=50)', lambda : ensemble.GradientBoostingRegressor(n_estimators = 50), {'interpolation'}),
		('Gradient Boosting (N=100)', lambda : ensemble.GradientBoostingRegressor(n_estimators = 100), {'interpolation'}),
		('Gradient Boosting (N=200)', lambda : ensemble.GradientBoostingRegressor(n_estimators = 200), {'interpolation'}),
		('AdaBoost (N=25)', lambda : ensemble.AdaBoostRegressor(n_estimators = 25), {'interpolation'}),
		('AdaBoost (N=50)', lambda : ensemble.AdaBoostRegressor(n_estimators = 50), {'interpolation'}),
		('AdaBoost (N=100)', lambda : ensemble.AdaBoostRegressor(n_estimators = 100), {'interpolation'}),
		('AdaBoost (N=200)', lambda : ensemble.AdaBoostRegressor(n_estimators = 200), {'interpolation'}),
		('SVR (RBF kernel)', lambda : svm.SVR(kernel = 'rbf', epsilon = 0.001, C=1e-1), {'interpolation'}),
		('SVR (Polynomial kernel)', lambda : svm.SVR(kernel = 'poly', epsilon = 0.001, C=1e-1), {'interpolation', 'dual_extrapolation', 'left_extrapolation', 'right_extrapolation'}),
		('SVR (Linear kernel)', lambda : svm.SVR(kernel = 'linear', epsilon = 0.001, C=1e-1), {'interpolation', 'dual_extrapolation', 'left_extrapolation', 'right_extrapolation'}),
		('Random Forest (N=25)', lambda : ensemble.RandomForestRegressor(n_estimators=25), {'interpolation'}),
		('Random Forest (N=50)', lambda : ensemble.RandomForestRegressor(n_estimators=50), {'interpolation'}),
		('Random Forest (N=100)', lambda : ensemble.RandomForestRegressor(n_estimators=100), {'interpolation'}),
		('Random Forest (N=200)', lambda : ensemble.RandomForestRegressor(n_estimators=200), {'interpolation'}),
	]
	if extended:
		models.extend([
			('Linear Regression (D=2)', lambda : LinearRegressionWithPolynomialActivation(2), {'interpolation', 'dual_extrapolation', 'left_extrapolation', 'right_extrapolation'}),
			('Linear Regression (Exp)', lambda : LinearRegressionWithExpActivation(), {'interpolation', 'dual_extrapolation', 'left_extrapolation', 'right_extrapolation'}),
			('Linear Regression (Log)', lambda : LinearRegressionWithLogActivation(), {'interpolation', 'dual_extrapolation', 'left_extrapolation', 'right_extrapolation'}),
			('Linear Regression (Sigmoid)', lambda : LinearRegressionWithSigmoidActivation(), {'interpolation', 'dual_extrapolation', 'left_extrapolation', 'right_extrapolation'})
		])
	
	return models


def run_experiments(dataset, output_dir = None):
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
	feature_name_groups = get_feature_name_groups()
	feature_sets = get_feature_sets(feature_name_groups)
	models = get_regression_models()
	if output_csv:
		with open(output_csv, 'w') as f:
			f.write('Model,Features,MeanAbsoluteError,NumViolations,MaxViolations,MonotonicityScore\n')
	for model_name, model_ctor, regression_mode_ignored in models:
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
				conf_matrix_path = os.path.join(figures_dir, f'{model_name}_{feature_set_name}_matrix.jpg')
				scatter_path = os.path.join(figures_dir, f'{model_name}_{feature_set_name}_scatter.jpg')
				create_confusion_matrix(result['sorting'], conf_matrix_path)
				create_scatter_plot(testset.MaxAccuracy, result['pred'], scatter_path)
				
if __name__ == '__main__':
	dataset_path = sys.argv[1]
	output_dir = sys.argv[2] if len(sys.argv) > 2 else None
	
	dataset = pd.read_csv(dataset_path)
	run_experiments(dataset, output_dir)
	