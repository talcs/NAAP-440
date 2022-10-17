import sys
import os
import re
import json

MAX_EPOCH = 90
TEST_SET_SIZE = 40
DEFAULT_DATASET_VERSION = 1

TRAIN_FILENAME_XTOR = re.compile(r'^scheme_(\d+).txt$')
EPOCH_XTOR = re.compile(r'Epoch (\d+),')
LOSS_XTOR = re.compile(r'Loss mean, median: (\d+\.\d+), (\d+\.\d+)')
ACCURACY_XTOR = re.compile(r'Test accuracy: (\d+\.\d+)')
SCHEME_XTOR = re.compile(r'Network scheme: (.+)$')
PROFILE_XTOR = re.compile(r'Network #params: (\d+), #MACs: (\d+)')

def extract_data_from_file(path):
	record = {}
	record['model_id'] = int(TRAIN_FILENAME_XTOR.search(os.path.basename(path)).group(1))
	with open(path, 'r') as f:
		epoch = 0
		max_accuracy = 0
		for l in f:
			epoch_re = EPOCH_XTOR.search(l)
			if epoch_re:
				epoch = int(epoch_re.group(1))
			scheme_re = SCHEME_XTOR.search(l)
			if scheme_re:
				scheme = json.loads(scheme_re.group(1))
				record['num_layers'] = len(scheme)
				record['num_stages'] = len([b for b in scheme if b['stride'] > 1])
				record['first_layer_width'] = scheme[0]['width']
				record['last_layer_width'] = scheme[-1]['width']
				record['num_skip_connections'] = len([b for b in scheme if b['residual']])
				record['num_prf_layers'] = len([b for b in scheme if b['stride'] > 1 and b['kernel_size'] == 1]) # PRF == Partial Receptive Field
			profile_re = PROFILE_XTOR.search(l)
			if profile_re:
				num_params = int(profile_re.group(1))
				num_macs = int(profile_re.group(2))
				record['num_params'] = num_params
				record['num_macs'] = num_macs
			loss_re = LOSS_XTOR.search(l)
			if loss_re:
				loss_mean = float(loss_re.group(1))
				loss_median = float(loss_re.group(2))
				record[f'e{epoch}LossMean'] = round(loss_mean, 4)
				record[f'e{epoch}LossMedian'] = round(loss_median, 4)
			accuracy_re = ACCURACY_XTOR.search(l)
			if accuracy_re:
				accuracy = float(accuracy_re.group(1))
				record[f'e{epoch}Accuracy'] = accuracy
				if accuracy > max_accuracy:
					max_accuracy = accuracy
		record['max_accuracy'] = max_accuracy
	
	return record

def get_csv_header(version):
	if version == 1:
		scheme_fields_addition = ''
	elif version == 2:
		scheme_fields_addition = ',NumSkipConnections,NumPartialRFLayers'
	else:
		raise Exception(f'Unknown version: {version}')
	header = f'ModelId,IsTest,NumParams,NumMACs,NumLayers,NumStages,FirstLayerWidth,LastLayerWidth{scheme_fields_addition},MaxAccuracy'
	for epoch in range(1, MAX_EPOCH + 1):
		header += ','
		header += ','.join(f'e{epoch}{metric}' for metric in ('LossMean','LossMedian','Accuracy'))
	return header
	
def file_data_to_csv_line(record, version):
	if version == 1:
		scheme_fields = ('model_id', 'is_test', 'num_params', 'num_macs', 'num_layers', 'num_stages', 'first_layer_width', 'last_layer_width', 'max_accuracy')
	elif version == 2:
		scheme_fields = ('model_id', 'is_test', 'num_params', 'num_macs', 'num_layers', 'num_stages', 'first_layer_width', 'last_layer_width', 'num_skip_connections', 'num_prf_layers', 'max_accuracy')
	else:
		raise Exception(f'Unknown version: {version}')
	line = ','.join(str(record[field]) for field in scheme_fields)
	for epoch in range(1, MAX_EPOCH + 1):
		line += ','
		line += ','.join(str(record[f'e{epoch}{metric}']) for metric in ('LossMean','LossMedian','Accuracy'))
		
	return line

def data_dir_to_csv(input_dir, output_file, version = DEFAULT_DATASET_VERSION):
	filenames = [v for v in os.listdir(input_dir) if TRAIN_FILENAME_XTOR.search(v)]
	data = [extract_data_from_file(os.path.join(input_dir, fn)) for fn in filenames]
	# Dividing the accuracy-sorted data into TEST_SET_SIZE bins
	data.sort(key = lambda x : x['max_accuracy'])
	bin_size = int(round(len(data) / TEST_SET_SIZE))
	bin_middle_index = bin_size // 2
	for i, record in enumerate(data):
		# The central record of each bin is allocated for the test set.
		record['is_test'] = int(i - bin_middle_index >= 0 and (i - bin_middle_index) % bin_size == 0)
	with open(output_file, 'w') as f:
		f.write(get_csv_header(version) + '\n')
		for record in sorted(data, key = lambda x : x['model_id']):
			f.write(file_data_to_csv_line(record, version) + '\n')

if __name__ == '__main__':
	input_dir = sys.argv[1]
	output_file = sys.argv[2]
	version = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DATASET_VERSION

	data_dir_to_csv(input_dir, output_file, version)