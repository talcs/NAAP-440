import sys
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import neural_network

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_experiments import run_experiment, create_confusion_matrix, create_scatter_plot, get_regression_models, set_random_seed
from custom_regressors import LinearRegressionWithPolynomialActivation, LinearRegressionWithExpActivation, LinearRegressionWithLogActivation


PRODUCE_FIGURES = True
LOG_PARAMS_AND_MACS = True
MAX_THREADS = 4

MODES = ['interpolation', 'dual_extrapolation', 'left_extrapolation', 'right_extrapolation']
MODEL_CTORS = get_regression_models(extended = True)
SCHEME_FEATURES = ('NumParams', 'NumMACs', 'NumLayers', 'NumStages', 'FirstLayerWidth', 'LastLayerWidth', 'NumSkipConnections', 'NumPartialRFLayers')
QUANT_FEATURES = {
	3 : ('e1LossMean', 'e1LossMedian', 'e1Accuracy', 'e2LossMean', 'e2LossMedian', 'e2Accuracy', 'e3LossMean', 'e3LossMedian', 'e3Accuracy'),
}
QUANT_FEATURES[6] = QUANT_FEATURES[3] + ('e4LossMean', 'e4LossMedian', 'e4Accuracy', 'e5LossMean', 'e5LossMedian', 'e5Accuracy', 'e6LossMean', 'e6LossMedian', 'e6Accuracy')
QUANT_FEATURES[9] = QUANT_FEATURES[6] + ('e7LossMean', 'e7LossMedian', 'e7Accuracy', 'e8LossMean', 'e8LossMedian', 'e8Accuracy', 'e9LossMean', 'e9LossMedian', 'e9Accuracy')
SEARCH_FEATURE_SETS = [
	('Scheme', SCHEME_FEATURES),
	('Scheme+Quant.3', SCHEME_FEATURES + QUANT_FEATURES[3]),
	('Scheme+Quant.6', SCHEME_FEATURES + QUANT_FEATURES[6]),
	('Scheme+Quant.9', SCHEME_FEATURES + QUANT_FEATURES[9]),
]

SEARCH_STOCHASTIC_BRANCHING_RANDOM_SEED = 20221015

def get_all_neighbors(curr_set):
	neighbors = []
	for i in range(len(curr_set)):
		neighbors.append(tuple(((1 - k) if j == i else k) for (j,k) in enumerate(curr_set)))
		
	return neighbors

def run_experiment_list(feature_set, feature_subsets, trainset, testset, model_name):
	model_ctor = None
	for model_name_, model_ctor_, model_modes in MODEL_CTORS:
		if model_name_ == model_name:
			model_ctor = model_ctor_
			break
	results = {}
	for curr_set in feature_subsets:
		curr_feature_set = [v for (flag, v) in zip(curr_set, feature_set) if flag > 0]
		result = run_experiment(trainset.copy(), testset.copy(), model_ctor, curr_feature_set, add_random_noise_to_predictions = True)
		result['final_score'] = round(result['mae'], 3) * (1 - result['ms'])**0.5  # (1- MonotonicityScore) is the violation rate
		result['curr_set'] = curr_set
		result['curr_feature_set'] = curr_feature_set
		results[curr_set] = result
	
	return results

def search(trainset, testset, model_name, model_ctor, feature_set, break_on_local_minima_eps = 3e-5, max_descent_steps = 30, max_states_to_check = np.inf, max_branching_factor = np.inf, max_threads = MAX_THREADS): # use eps=0 for no tolerance
	initial_set = (1,) * len(feature_set)
	queue = {np.inf: [initial_set]}
	done = {}
	queued = set()
	best_achieved_score = np.inf
	stop = False
	num_descent_steps = 0
	max_sets = max_branching_factor * len(feature_set)
	with ProcessPoolExecutor(max_workers = max_threads) as executor:
		while len(queue) > 0 and num_descent_steps <= max_descent_steps and (not stop):# and (not stop):
			num_descent_steps += 1
			min_anscestor_score = min(queue)
			next_sets = queue[min_anscestor_score]
			del queue[min_anscestor_score]
			next_sets = list({s for s in next_sets if s not in done})
			if len(next_sets) > max_sets:
				set_random_seed(SEARCH_STOCHASTIC_BRANCHING_RANDOM_SEED)
				random.shuffle(next_sets)
				queue[min_anscestor_score] = next_sets[max_sets:] # return prunned states back to the queue
				next_sets = next_sets[:max_sets]
			num_threads = min(len(next_sets), max_threads)
			num_samples_in_batch = int(round(len(next_sets) / float(num_threads)))
			batches = []
			for i in range(num_threads):
				if i < num_threads - 1:
					batches.append(next_sets[i*num_samples_in_batch:(i+1)*num_samples_in_batch])
				else:
					batches.append(next_sets[i*num_samples_in_batch:])
			futures = []
			last_done = {}
			
			for b in batches:
				futures.append(executor.submit(run_experiment_list, feature_set, b, trainset, testset, model_name))
			for fu in futures:
				results = fu.result()
				done.update(results)
				last_done.update(results)
			print(f'\r       Analyzed {len(done)} feature subsets so far ({num_descent_steps - 1} descent steps)...', end = '')
			if len(done) == max_states_to_check:
				stop = True
				break
			for curr_set, result in last_done.items():
				if result['final_score'] < best_achieved_score:
					best_achieved_score = result['final_score']
				elif result['final_score'] - break_on_local_minima_eps > best_achieved_score:
					continue
				
				# Append all neighbors that haven't been analyzed yet
				neighbors = get_all_neighbors(curr_set)
				for neighbor in neighbors:
					if sum(neighbor) == 0:
						continue
					if neighbor not in done:# and neighbor not in queued:
						if result['final_score'] not in queue:
							queue[result['final_score']] = []
						queue[result['final_score']].append(neighbor)
						queued.add(neighbor)	
	print('')
	
	return done

def feature_selection(dataset, output_dir = None):
	dataset = dataset.sort_values('MaxAccuracy')
	if LOG_PARAMS_AND_MACS:
		dataset.NumParams = np.log(dataset.NumParams)
		dataset.NumMACs = np.log(dataset.NumMACs)
	trainset = dataset[dataset.IsTest == 0]
	testset = dataset[dataset.IsTest == 1]
	
	out_csv_path = None
	figures_dir = None
	if output_dir is not None:
		os.mkdir(output_dir)
		out_csv_path = os.path.join(output_dir, 'results.csv')
		if PRODUCE_FIGURES:
			figures_dir = os.path.join(output_dir, 'figures')
			os.mkdir(figures_dir)
	if out_csv_path is not None:
		with open(out_csv_path, 'w') as f:
			f.write('Mode,Model,FeatureSetName,Flags,FeatureSubset,MAE,NumViolations,MaxViolations,MonotonicityScore,FinalScore\n')
	
	for mode in MODES:
		if mode == 'interpolation':
			mode_trainset = trainset
			mode_testset = testset
		elif mode == 'left_extrapolation':
			mode_trainset = trainset[len(trainset)//2:]
			mode_testset = testset[:len(testset)//2]
		elif mode == 'right_extrapolation':
			mode_trainset = trainset[:len(trainset)//2]
			mode_testset = testset[len(testset)//2:]
		elif mode == 'dual_extrapolation':
			mode_trainset = trainset[len(trainset)//4:3*len(trainset)//4]
			mode_testset = pd.concat((testset[:len(testset)//4], testset[-len(testset)//4:]))
		print(f'Mode: {mode}')
		for model_name, model_ctor, model_modes in MODEL_CTORS:
			if mode not in model_modes:
				continue
			print(f'  -> Model: {model_name}')
			for feature_set_name, feature_set in SEARCH_FEATURE_SETS:
				print(f'    -> Feature set: {feature_set_name}')
				results = search(mode_trainset, mode_testset, model_name, model_ctor, feature_set, 
									break_on_local_minima_eps = np.inf, max_descent_steps = len(feature_set), max_branching_factor = 3)
				best_result = min(results.values(), key = lambda x : x['final_score'])
				if out_csv_path is not None:
					with open(out_csv_path, 'a') as f:
						for res_key, res in results.items():
							flags = ';'.join(str(v) for v in res['curr_set'])
							feature_subset = ';'.join(str(v) for v in res['curr_feature_set'])
							f.write(f'{mode},{model_name},{feature_set_name},{flags},{feature_subset},{res["mae"]},{res["num_violations"]},{res["max_violations"]},{res["ms"]},{res["final_score"]}\n')
					if figures_dir:
						conf_matrix_path = os.path.join(figures_dir, f'{mode}_{model_name}_{feature_set_name}_matrix.jpg')
						scatter_path = os.path.join(figures_dir, f'{mode}_{model_name}_{feature_set_name}_scatter.jpg')
						create_confusion_matrix(best_result['sorting'], conf_matrix_path)
						create_scatter_plot(mode_testset.MaxAccuracy, best_result['pred'], scatter_path)
				else:
					print(best_result)
	
	


if __name__ == '__main__':
	dataset_path = sys.argv[1]
	output_dir = sys.argv[2] if len(sys.argv) > 2 else None
	
	dataset = pd.read_csv(dataset_path)
	feature_selection(dataset, output_dir)
