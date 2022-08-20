import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np


MAX_THREADS = 4
SEED = 20220819
RECIPE = {
	'max_epochs' : 30,
	'batch_size' : 256,
	'physical_batch_size' : 256,
	'lr' : 0.1,
	'wd' : 1e-4,
	'lr_gamma' : 0.1,
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
				nn.Conv2d(kernel_size, width, prev_tensor_width, stride, padding, bias = False),
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

def create_sample(scheme, device, recipe, seed):
	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	net = Net(scheme).to(device)
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
											download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=recipe['batch_size'],
											  shuffle=True, num_workers=1)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr = recipe['lr'], momentum = recipe['momentum'], weight_decay = recipe['weight_decay'])
	physical_batch_size = recipe['physical_batch_size']
	for epoch in range(recipe['max_epochs']):
		running_loss = 0.0
		num_samples = 0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			optimizer.zero_grad()
			for j in range(0, len(inputs), physical_batch_size):
				# Allowing big SGD batches and allowing to split them for physical computation, before taking the step
				from_index = j
				to_index = min(j + physical_batch_size, len(inputs))
				inputs_j = inputs[from_index:to_index].to(device)
				labels_j = labels[from_index:to_index].to(device)
				outputs = net(inputs_j)
				loss = criterion(outputs, labels_j)
				loss.backward()
				running_loss += loss.item()
				num_samples += to_index - from_index
			optimizer.step()


def create_dataset(schemes, output_file, device = DEVICE, recipe = RECIPE, seed = SEED, max_threads = MAX_THREADS):
	for scheme in schemes:
	