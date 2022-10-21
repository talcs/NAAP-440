import sys
import os

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from conf_matrix_creator import conf_matrix_creator

def get_feature_list(dataframe):
	d = dataframe[~dataframe.Flags.str.contains('0')]
	
	return d.FeatureSubset.iloc[0].replace('NumParams','LogNumParams').replace('NumMACs', 'LogNumMACs').split(';')

def fix_alg_names_and_order(algs):
	algs = list(algs)
	lin_reg_indices = [i for (i,a) in enumerate(algs) if 'Linear Regression' in a]
	first_group_end_index = None
	for i, ind in enumerate(lin_reg_indices):
		if first_group_end_index is None and i > 0 and ind - lin_reg_indices[i-1] > 1:
			first_group_end_index = lin_reg_indices[i-1]
		if first_group_end_index is not None:
			alg = algs.pop(ind)
			algs.insert(first_group_end_index + 1, alg)
			first_group_end_index += 1
	fixed_alg_names = []
	for alg in algs:
		if not alg.startswith('Linear Regression'):
			fixed_alg_names.append(alg)
			continue
		if alg == 'Linear Regression':
			fixed_alg_names.append(alg)
			continue
		if '(' in alg:
			fixed_alg_names.append(alg)
			continue
		parts = alg.split(' ', 3)
		fixed_alg_names.append('Linear Regression (' + parts[2] + ')')
		
	return algs, fixed_alg_names

def create_feature_conf_matrix_figure(input_csv, mode, featureset_name, best_rate, save_path):
	data = pd.read_csv(input_csv)
	data = data[data.Mode == mode]
	data = data[data.FeatureSetName == featureset_name]
	algs = data.Model.unique()
	algs, fixed_alg_names = fix_alg_names_and_order(algs)
	feature_list = get_feature_list(data)
	matrix = np.zeros((len(algs), len(feature_list)))
	for i, alg in enumerate(algs):
		dalg = data[data.Model == alg].sort_values('FinalScore')
		best_abs = int(round(len(dalg) * best_rate))
		best_flags = dalg.Flags.iloc[:best_abs]
		for flags in best_flags:
			for j, flag in enumerate(flags.split(';')):
				if flag == '1':
					matrix[i,j] += 1
		matrix[i,:] /= best_abs
	
	settings = {
		'figsize' : (len(algs) // 2,len(algs) // 2),
		'colormap' : 'magma_r',
		'colorbar' : {
			'view' : True,
			'arange' : (0, 1.001, 0.1),
			'text_formatter' : lambda tick_value : '{0:.0f}%'.format(tick_value*100),
		},
		'xticklabels' : {
			'labels' : feature_list,
			'location' : 'top',
			'rotation' : 75,
		},
		'yticklabels' : {
			'labels' : fixed_alg_names,
		},
		'cell_text' : {
			'vertical_alignment' : 'center',
			'horizontal_alignment' : 'center',
			'size' : 'x-small',
			'color_function' : lambda cell_value : 'black' if cell_value < 0.5 else 'white',
			'text_formatter' : lambda cell_value : '{0:.0f}%'.format(cell_value*100),
		},
	}
	conf_matrix_creator(matrix, settings, save_path)
		

if __name__ == '__main__':
	input_csv = sys.argv[1]
	mode = sys.argv[2]
	featureset_name = sys.argv[3]
	best_rate = float(sys.argv[4])
	save_path = sys.argv[5] if len(sys.argv) > 5 else None
	
	create_feature_conf_matrix_figure(input_csv, mode, featureset_name, best_rate, save_path)
	