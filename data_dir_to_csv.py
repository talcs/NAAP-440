import sys
import os
import re

MAX_EPOCH = 90

TRAIN_FILENAME_XTOR = re.compile(r'^scheme_(\d+).txt$')
EPOCH_XTOR = re.compile(r'Starting epoch (\d+) of')
LOSS_XTOR = re.compile(r'Loss mean, median: (\d+\.\d+), (\d+\.\d+)')
ACCURACY_XTOR = re.compile(r'Test accuracy: (\d+\.\d+)')

def extract_data_from_file(path):
	data = {'model_id' : TRAIN_FILENAME_XTOR.search(os.path.basename(path)).group(1)}
	with open(path, 'r') as f:
		epoch = 0
		loss_mean = 0
		loss_median = 0
		accuracy = 0
		for l in f:
			epoch_re = EPOCH_XTOR.search(l)
			if epoch_re:
				epoch = int(epoch_re.group(1))
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
	header = 'ModelId'
	for epoch in range(1, MAX_EPOCH + 1):
		header += ','
		header += ','.join(f'e{epoch}{metric}' for metric in ('LossMean','LossMedian','Accuracy'))
		
	return header
	
def file_data_to_csv_line(data):
	line = data['model_id']
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