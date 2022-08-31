import sys
import os
import re
import json

MAX_EPOCH = 90

TRAIN_FILENAME_XTOR = re.compile(r'^scheme_(\d+).txt$')
EPOCH_XTOR = re.compile(r'Starting epoch (\d+) of')
LOSS_XTOR = re.compile(r'Loss mean, median: (\d+\.\d+), (\d+\.\d+)')
ACCURACY_XTOR = re.compile(r'Test accuracy: (\d+\.\d+)')
SCHEME_XTOR = re.compile(r'Network scheme: (.+)$')
PROFILE_XTOR = re.compile(r'Network #params: (\d+), #MACs: (\d+)')

def extract_data_from_file(path):
	data = {'model_id' : TRAIN_FILENAME_XTOR.search(os.path.basename(path)).group(1)}
	with open(path, 'r') as f:
		epoch = 0
		loss_mean = 0
		loss_median = 0
		accuracy = 0
		scheme = None
		num_params = 0
		num_macs = 0
		for l in f:
			epoch_re = EPOCH_XTOR.search(l)
			if epoch_re:
				epoch = int(epoch_re.group(1))
			scheme_re = SCHEME_XTOR.search(l)
			if scheme_re:
				scheme = json.loads(scheme_re.group(1))
				data['num_layers'] = len(scheme)
				data['num_stages'] = len([b for b in scheme if b['stride'] > 1])
				data['first_layer_width'] = scheme[0]['width']
				data['last_layer_width'] = scheme[-1]['width']
			profile_re = PROFILE_XTOR.search(l)
			if profile_re:
				num_params = int(profile_re.group(1))
				num_macs = int(profile_re.group(2))
				data['num_params'] = num_params
				data['num_macs'] = num_macs
			loss_re = LOSS_XTOR.search(l)
			if loss_re:
				loss_mean = float(loss_re.group(1))
				loss_median = float(loss_re.group(2))
				data[f'e{epoch}LossMean'] = round(loss_mean, 4)
				data[f'e{epoch}LossMedian'] = round(loss_median, 4)
			accuracy_re = ACCURACY_XTOR.search(l)
			if accuracy_re:
				accuracy = float(accuracy_re.group(1))
				data[f'e{epoch}Accuracy'] = accuracy
	
	return data

def get_csv_header():
	header = 'ModelId,NumParams,NumMacs,NumLayers,NumStages,FirstLayerWidth,LastLayerWidth'
	for epoch in range(1, MAX_EPOCH + 1):
		header += ','
		header += ','.join(f'e{epoch}{metric}' for metric in ('LossMean','LossMedian','Accuracy'))
		
	return header
	
def file_data_to_csv_line(data):
	line = ','.join(str(data[field]) for field in ('model_id', 'num_params', 'num_macs', 'num_layers', 'num_stages', 'first_layer_width', 'last_layer_width'))
	for epoch in range(1, MAX_EPOCH + 1):
		line += ','
		line += ','.join(str(data[f'e{epoch}{metric}']) for metric in ('LossMean','LossMedian','Accuracy'))
		
	return line

def data_dir_to_csv(input_dir, output_file):
	filenames = [v for v in os.listdir(input_dir) if TRAIN_FILENAME_XTOR.search(v)]
	filenames.sort(key = lambda x : int(TRAIN_FILENAME_XTOR.search(x).group(1)))
	with open(output_file, 'w') as f:
		f.write(get_csv_header() + '\n')
		for fn in filenames:
			data = extract_data_from_file(os.path.join(input_dir, fn))
			f.write(file_data_to_csv_line(data) + '\n')

if __name__ == '__main__':
	input_dir = sys.argv[1]
	output_file = sys.argv[2]

	data_dir_to_csv(input_dir, output_file)