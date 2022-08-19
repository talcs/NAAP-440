import sys
import json

SETTINGS = {
	'max_stages' : 3, # Max stride=2 layers along the network
	'min_conv_blocks' : 3, # Network's minimal depth
	
	'conv_blocks' : [
		{
			'kernel_size' : [3],
			'width' : [12, 16],
			'stride' : [2],
			'residual' : [False],
		},
		{
			'kernel_size' : [3],
			'width' : [16, 24],
			'stride' : [2],
			'residual' : [False],
		},
		{
			'kernel_size' : [1, 3],
			'width' : [16, 24, 32],
			'stride' : [1, 2],
			'residual' : [False, True] # Residual will only be allowed if the input and output tensors have the same shape
		},
		{
			'kernel_size' : [1, 3],
			'width' : [32, 40],
			'stride' : [1, 2], # Applying stride = 2 is only possible if previous layer had stride = 1
			'residual' : [False, True] # Residual will only be allowed if the input and output tensors have the same shape
		},
	]
}

def scheme_generator(prefix = None, settings = SETTINGS):
	# This function runs BFS (calling itself recursively) to find all possible architectures according to the settings
	if prefix is None:
		prefix = []
	candidates = []
	if len(prefix) >= settings['min_conv_blocks']:
		candidates.append(prefix)
	if len(prefix) == len(settings['conv_blocks']):
		return candidates
	can_apply_stride2 = len([b for b in prefix if b['stride'] > 1]) < settings['max_stages']
	block_index = len(prefix)
	block_constraints = settings['conv_blocks'][block_index]
	for ks in block_constraints['kernel_size']:
		for w in block_constraints['width']:
			for s in block_constraints['stride']:
				if s > 1 and (not can_apply_stride2): 
					# Too many layers with s=2, skipping this branch
					continue
				for r in block_constraints['residual']:
					if r and (s != 1 or w != prefix[-1]['width']):
						# Cannot apply residual connections when tensors do not agree on shape, skipping this branch
						continue
					cand_block = {'kernel_size' : ks, 'width' : w, 'stride' : s, 'residual' : r}
					candidates += scheme_generator(prefix + [cand_block])
					
	return candidates

if __name__ == '__main__':
	output_file = sys.argv[1]
	
	schemes = scheme_generator()	
	print(f'-I- Done. {len(schemes)} schemes have been generated to file {output_file}.')
	with open(output_file, 'w') as f:
		json.dump(schemes, f, indent = 2)