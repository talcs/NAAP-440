import sys
import re

import pandas as pd

MAX_EPOCHS = 90


def rename_columns(dataframe):
	return dataframe.rename(columns={'FeatureSetName' : 'Features', 'MAE' : 'MeanAbsoluteError'})

def filter_interpolation(dataframe):
	return dataframe[dataframe.Mode == 'interpolation']
	
def unify_featureset_names(dataframe):
	# Turning "Scheme+Quant.X" to "Scheme + Quantitative X epochs"
	dataframe['Features'] = dataframe['Features'].str.replace('Scheme\+Quant.', 'Scheme + Quantitative ')
	dataframe['Features'][dataframe['Features'].str.contains('Quantitative')] = dataframe['Features'][dataframe['Features'].str.contains('Quantitative')].astype('str') + ' epochs'
	
	return dataframe

def rename_columns_interpolation_all_features_filter(dataframe):
	d = rename_columns(dataframe)
	d = filter_interpolation(d)
	d = d[~d.Flags.str.contains('0')]
	d = unify_featureset_names(d)
	
	return d

def rename_columns_interpolation_best_features_filter(dataframe):
	d = rename_columns(dataframe)
	d = filter_interpolation(d)
	new_data = d[0:0] # empty
	models = pd.unique(d.Model)
	feature_types = pd.unique(d.Features)
	for model in models:
		#print(model, end = ' ')
		dm = d[d.Model == model]
		for features in feature_types:
			#print (features, end = '   ')
			df = dm[dm.Features == features]
			#print(len(df), end = '  ')
			df = df[df.FinalScore == df.FinalScore.min()] # The final score uses rounded MAE, they may be multiple candidates
			#print(len(df), end = '  ')
			best_rows = df[df.MeanAbsoluteError == df.MeanAbsoluteError.min()] # Choosing candidate(s) with real lowest MAE out of final score winners
			#print(len(best_rows), end = '  ')
			new_data = new_data.append(best_rows[:1], ignore_index = True)
	new_data = unify_featureset_names(new_data)
	#print(new_data.Features)
	
	return new_data


FILTERS = {
	'RC_INTER_AF' : rename_columns_interpolation_all_features_filter,
	'RC_INTER_BF' : rename_columns_interpolation_best_features_filter,
}

"""
\begin{table}[]
\centering
\resizebox{\linewidth}{!}{%
\begin{tabular}{cc||c|c|c|c|c|c|c|c}
\hline
\hline
\textbf{Exponents} & \textbf{Log} & \multicolumn{7}{|c|}{\textbf{Predicted Top-1 Accuracy (Absolute Error)}} & \textbf{MAE} \\ \hline
 &  & \textbf{ResNet-18} & \textbf{ResNet-34} &\textbf{ResNet-50} &\textbf{ResNet-101} &\textbf{ResNet-152} &\textbf{Wide ResNet-50} &\textbf{Wide ResNet-101} & \\ 
 &  & GT=0.698 & GT=0.733 & GT=0.761 & GT=0.774 & GT=0.783 & GT=0.785 & GT=0.788 & \\
 \hline
1 & No & 0.752 (0.054) & 0.744 (0.011) & 0.739 (0.022) & 0.753 (0.021) & 0.763 (0.020) & 0.767 (0.017) & 0.877 (0.088) & 0.033 \\
1 & Yes & 0.731 (0.033) & 0.739 (0.006) & 0.739 (0.022) & 0.764 (0.010) & 0.775 (0.008) & 0.780 (0.005) & 0.822 (0.034) & \textbf{0.017} \\
1,2 & No & 0.761 (0.064) & 0.752 (0.019) & 0.746 (0.015) & 0.750 (0.024) & 0.755 (0.028) & 0.758 (0.026) & 0.975 (0.186) & 0.052 \\
1,2 & Yes & 0.738 (0.041) & 0.739 (0.005) & 0.738 (0.024) & 0.760 (0.014) & 0.772 (0.011) & 0.777 (0.008) & 0.833 (0.045) & 0.021 \\
1,2,3 & No & 0.765 (0.067) & 0.757 (0.024) & 0.750 (0.011) & 0.750 (0.024) & 0.752 (0.031) & 0.754 (0.031) & 1.154 (0.365) & 0.079 \\
1,2,3 & Yes & 0.744 (0.047) & 0.740 (0.007) & 0.737 (0.024) & 0.757 (0.017) & 0.769 (0.014) & 0.774 (0.011) & 0.846 (0.057) & 0.025 \\
1,2,3,4 & No & 0.000 (0.698) & 0.001 (0.732) & 0.001 (0.760) & 0.013 (0.760) & 0.043 (0.740) & 0.065 (0.720) & 1.484 (0.696) & 0.729 \\
1,2,3,4 & Yes & 0.749 (0.051) & 0.742 (0.009) & 0.738 (0.024) & 0.755 (0.019) & 0.766 (0.017) & 0.771 (0.014) & 0.860 (0.071) & 0.029 \\
1,2,3,4,5 & No & 0.000 (0.698) & 0.000 (0.733) & 0.000 (0.761) & 0.004 (0.769) & 0.020 (0.763) & 0.033 (0.752) & 24.434 (23.646) & 4.017 \\
1,2,3,4,5 & Yes & 0.753 (0.055) & 0.744 (0.011) & 0.739 (0.022) & 0.753 (0.021) & 0.763 (0.020) & 0.768 (0.017) & 0.876 (0.087) & 0.033 \\
\hline
\hline
\end{tabular}
}
\caption[]{CLS-LOC Top-1 leave-one-out regression errors, using model count of parameters as the only feature.}
\label{tbl:regressionFromNumParamsTop1}
\end{table}
"""

def exp_results_to_latex_table(exp_results, output_tex):
	acceleration_catgory_names = ['Quantitative 0 epochs'] # Doesn't exist in the data
	acceleration_categories = [1]
	num_epoch_categories = [0]
	for feature_set_name in exp_results[0].Features.unique(): # Assuming that feature set names are unified over all data sources
		if 'Scheme' == feature_set_name:
			continue
		num_epochs = int(re.search(r'(\d+)', feature_set_name).group(1))
		if num_epochs > 9:
			continue
		num_epoch_categories.append(num_epochs)
		acceleration = 1 - num_epochs / float(MAX_EPOCHS)
		acceleration_categories.append(acceleration)
		acceleration_catgory_names.append(feature_set_name)
	acceleration_categories.sort(reverse = True)
	full_algorithm_list_sorted = []
	known_algorithms = set()
	for exp_result in exp_results: # merging a sorted algorithms list from all data sources
		algorithms = exp_result.Model.unique()
		for alg in algorithms:
			if alg not in known_algorithms:
				known_algorithms.add(alg)
				full_algorithm_list_sorted.append(alg)
	text = """
\\begin{{table*}}[]
\\centering
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{l|{0}}}
\\hline
\\hline
 &  \\multicolumn{{{1:d}}}{{c}}{{\\textbf{{MAE / Monotonicity Score / \\#Monotonicity Violations}}}} \\\\
 \\hline
""".format('|c' * len(acceleration_categories), len(acceleration_categories))
	text += ' & ' + ' & '.join(['{0:.1f}\% acceleration'.format(100 * v) for v in acceleration_categories]) + '\\\\ \n'
	text += '\\textbf{Algorithm} & ' + ' & '.join(['({0:d} epochs)'.format(v) for v in num_epoch_categories]) + '\\\\ \n'
	text += '\\hline\n'
	for alg in full_algorithm_list_sorted:
		for i, exp_result in enumerate(exp_results):
			model_data = exp_result[exp_result.Model == alg]
			# Scheme is always included now
			scheme_data = model_data[model_data.Features.str.contains('Scheme')]
			if i == 0:
				text += alg # + ' & ' + ('Yes' if scheme else 'No') 
			for acceleration, cat_name in zip(acceleration_categories, acceleration_catgory_names):
				text += ' & '
				if ' 0 epochs' in cat_name:
					acc_data = scheme_data[scheme_data.Features == 'Scheme']
				else:
					acc_data = scheme_data[scheme_data.Features == cat_name]
				if len(acc_data) == 0:
					text += '-'
				else:
					text += '{0:.3f} / {2:.3f} / {1:d} '.format(acc_data.MeanAbsoluteError.item(), acc_data.NumViolations.item(), acc_data.MonotonicityScore.item())
			text += '\\\\ \n'
		if len(exp_results) > 1:
			text += '\\hline\n'
	text += """
\\hline
\\end{tabular}
}
\\caption[]{bla.}
\\label{tbl:bla}
\\end{table*}
"""

	with open(output_tex, 'w') as f:
		f.write(text)
		
def process_inputs(input_csvs_and_optional_filters):
	dataframes = []
	for input_csv_and_optional_filter in input_csvs_and_optional_filters:
		input_csv = input_csv_and_optional_filter
		filter_id = None
		if ',' in input_csv_and_optional_filter:
			input_csv, filter_id = input_csv_and_optional_filter.split(',')
		
		exp_results = pd.read_csv(input_csv)
		if filter_id is not None:
			filter = FILTERS[filter_id]
			exp_results = filter(exp_results)
		dataframes.append(exp_results)
		
	return dataframes

if __name__ == '__main__':
	if len(sys.argv) < 3:
		sys.stderr.write('Usage: exp_results_to_latex_table.py <input_file1,[optional_filter]> [input_file2,[optional_filter]] ... <output tex file>\n')
		sys.exit(2)
	input_csvs_and_optional_filters = sys.argv[1:-1]
	output_tex = sys.argv[-1]
	
	dataframes = process_inputs(input_csvs_and_optional_filters)
	"""
	input_csv = input_csv_and_optional_filter
	filter_id = None
	if ',' in input_csv_and_optional_filter:
		input_csv, filter_id = input_csv_and_optional_filter.split(',')
	
	exp_results = pd.read_csv(input_csv)
	if filter_id is not None:
		filter = FILTERS[filter_id]
		exp_results = filter(exp_results)
	"""
	
	exp_results_to_latex_table(dataframes, output_tex)
	