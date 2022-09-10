import sys
import os
import random
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from thop import profile


DEVICE = 'cuda:0'
MAX_THREADS = 4
SEED = 20220819
RECIPE = {
	'max_epochs' : 90,
	'batch_size' : 256,
	'physical_batch_size' : 256,
	'lr' : 0.1,
	'wd' : 1e-4,
	'lr_gamma' : 0.1,
	'lr_gamma_freq' : 1,
	'restart_rate' : 3,
	'momentum' : 0.9,
}

class Net(nn.Module):
	def __init__(self, scheme):
		super().__init__()
		self.components = nn.ModuleList()
		self.residual_connections = [b['residual'] for b in scheme]
		prev_tensor_width = 3
		for block in scheme:
			kernel_size, width, stride = block['kernel_size'], block['width'], block['stride']
			padding = kernel_size // 2
			self.components.append(nn.Sequential(
				nn.Conv2d(prev_tensor_width, width, kernel_size, stride, padding, bias = False),
				nn.BatchNorm2d(width),
				nn.ReLU()
				)
			)
			prev_tensor_width = width
		
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(prev_tensor_width, 10)
		
		# weight initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out")
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		prev = x
		for block, has_res_conn in zip(self.components, self.residual_connections):
			x = block(x)
			if has_res_conn:
				x += prev
			prev = x
		x = self.pool(x)
		x = torch.flatten(x, 1)	   
		x = self.classifier(x)
		
		return x	

def cifar10_dataset(train):
	transform = transforms.Compose(
			[transforms.ToTensor(),
			 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	return torchvision.datasets.CIFAR10(root='./data', train=train,
												download=True, transform=transform)

def set_random_seed(seed):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
def set_lr(optimizer, lr):
	for g in optimizer.param_groups:
		g['lr'] = lr
		
def get_lr(optimizer):
	return optimizer.param_groups[0]['lr']
	
def thin_datetime():
	s = str(datetime.now())
	
	return s[:-7].replace(':','').replace('-','').replace(' ', '')

def create_sample(scheme, device, recipe, seed, output_file):
	with open(output_file, 'w', buffering = 1) as f:
		f.write(f'{thin_datetime()} Network scheme: {json.dumps(scheme)}\n')
		set_random_seed(seed) # For the network random weight initializations
		net = Net(scheme).to(device)
		eval_net = Net(scheme).to(device)
		eval_net.eval()
		profile_net = Net(scheme)
		profile_net.eval()
		f.write(f'{thin_datetime()} Network has been init on device {device}\n')
		dummy_input = torch.zeros(1, 3, 32, 32)
		macs, params = profile(profile_net, inputs=(dummy_input, ), verbose = False)
		f.write(f'{thin_datetime()} Network #params: {int(params)}, #MACs: {int(macs)}\n')
		set_random_seed(seed) # For the training set data shuffling
		# First process should run separately to download the dataset without race condition
		trainset = cifar10_dataset(train = True)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=recipe['batch_size'],
												  shuffle=True, num_workers=1)
		f.write(f'{thin_datetime()} Training set is ready\n')
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr = recipe['lr'], momentum = recipe['momentum'], weight_decay = recipe['wd'])
		physical_batch_size = recipe['physical_batch_size']
		f.write(f'{thin_datetime()} Ready to start training\n')
		testset = cifar10_dataset(train = False)
		testloader = torch.utils.data.DataLoader(testset, batch_size=recipe['batch_size'],
												  shuffle=False, num_workers=1)
		for epoch in range(recipe['max_epochs']):
			if epoch != 0 and epoch % recipe['lr_gamma_freq'] == 0:
				set_lr(optimizer, get_lr(optimizer) * recipe['lr_gamma'])
			if recipe['restart_rate'] != 0 and epoch != 0 and epoch % recipe['restart_rate'] == 0:
				set_lr(optimizer, recipe['lr'])
			lr_to_print = '{0:g}'.format(get_lr(optimizer))
			f.write(f'{thin_datetime()} Epoch {epoch+1}, LR: {lr_to_print}\n')
			batch_losses = []
			for i, data in enumerate(trainloader):
				inputs, labels = data
				optimizer.zero_grad()
				virtual_batch_loss = 0.0
				for j in range(0, len(inputs), physical_batch_size):
					# Allowing big SGD batches and allowing to split them for physical computation, before taking the step
					from_index = j
					to_index = min(j + physical_batch_size, len(inputs))
					inputs_j = inputs[from_index:to_index].to(device)
					labels_j = labels[from_index:to_index].to(device)
					outputs = net(inputs_j)
					loss = criterion(outputs, labels_j) * len(inputs_j) / float(len(inputs))
					loss.backward()
					virtual_batch_loss += loss.item()
				batch_losses.append(virtual_batch_loss)
				optimizer.step()
			f.write(f'{thin_datetime()} Loss mean, median: {round(np.mean(batch_losses), 5)}, {round(np.median(batch_losses), 5)}\n')		
			correct = 0
			total = 0
			with torch.no_grad():
				eval_net.load_state_dict(net.state_dict())
				for data in testloader:
					inputs, labels = data
					inputs = inputs.to(device)
					outputs = eval_net(inputs).cpu().numpy()
					winners = np.argmax(outputs, axis = 1)
					correct += len([v for v in winners.astype('int') - labels.cpu().numpy().astype('int') if v == 0])
					total += len(inputs)
			accuracy = correct / float(total)
			f.write(f'{thin_datetime()} Test accuracy: {accuracy}\n')
					
					
def add_scheme_to_pool(index, scheme, device, recipe, seed, output_file, executor, futures):
	if os.path.isfile(output_file):
		print(f'Warning: Output file already existing for scheme {index+1}: {output_file}. Skipping scheme...')
		
		return
	futures.append(executor.submit(create_sample, scheme, device, recipe, seed, output_file))

def create_dataset(schemes, output_dir, device = DEVICE, recipe = RECIPE, seed = SEED, max_threads = MAX_THREADS, allow_existing_output_dir = False):
	if os.path.isdir(output_dir):
		if (not allow_existing_output_dir):
			raise Exception(f'Specified output directory already exists: {output_dir}')
	else:
		os.mkdir(output_dir)
	if len(schemes) == 0:
		return
	print(f'{datetime.now()} Making sure that the CIFAR10 dataset is ready')
	# Pre initializing dataset to ensure that it is ready to use
	cifar10_dataset(train = True)
	cifar10_dataset(train = False)
	scheme_to_output_file = lambda index : os.path.join(output_dir, f'scheme_{index+1}.txt')
	futures = []
	print(f'{datetime.now()} Starting pool execution of all schemes')
	with ProcessPoolExecutor(max_workers = max_threads) as executor:
		for i, scheme in enumerate(schemes):
			output_file = scheme_to_output_file(i)
			add_scheme_to_pool(i, scheme, device, recipe, seed, output_file, executor, futures)
		count_num_finished = lambda : len([1 for future in futures if future.done()])
		print(f'{datetime.now()} {len(futures)} jobs were added to pool', flush = True)
		num_finished = -1
		while True:
			# Waiting on all schemes to finish		
			curr_num_finished = count_num_finished()
			if curr_num_finished > num_finished:
				print(f'{datetime.now()} {curr_num_finished} of {len(futures)} jobs finished', flush = True)
				num_finished = curr_num_finished
			if num_finished == len(futures):
				break
			time.sleep(10)
		for future in futures:
			exc =  future.exception()
			if exc:
				print(exc)
		
if __name__ == '__main__':
	schemes_file = sys.argv[1]
	output_path = sys.argv[2]
	
	with open(schemes_file, 'r') as f:
		schemes = json.load(f)
	
	create_dataset(schemes, output_path)