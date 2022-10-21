import matplotlib.pyplot as plt
import numpy as np


def conf_matrix_creator(mat, settings, dst_path = None):
	colormap = settings['colormap'] if 'colormap' in settings else None
	figsize = settings['figsize'] if 'figsize' in settings else None
	plt.figure(figsize = figsize)
	plt.imshow(mat, cmap =  colormap)
	
	view_colorbar = settings['colorbar']['view'] if 'colorbar' in settings else True
	if view_colorbar:
		ticks = np.arange(*settings['colorbar']['arange']) if 'colorbar' in settings and 'arange' in settings['colorbar'] else None
		cbar = plt.colorbar(ticks = ticks)
		if 'colorbar' in settings and 'text_formatter' in settings['colorbar']:
			cbar.ax.set_yticklabels([settings['colorbar']['text_formatter'](v) for v in ticks])
	if 'cell_text' in settings:
		for x in range(mat.shape[1]):
			for y in range(mat.shape[0]):
				text_color = settings['cell_text']['color_function'](mat[y,x]) if 'color_function' in settings['cell_text'] else 'black'
				va = settings['cell_text']['vertical_alignment'] if 'vertical_alignment' in settings['cell_text'] else 'center'
				ha = settings['cell_text']['horizontal_alignment'] if 'horizontal_alignment' in settings['cell_text'] else 'center'
				size = settings['cell_text']['size'] if 'size' in settings['cell_text'] else 'x-large'
				text = settings['cell_text']['text_formatter'](mat[y,x]) if 'text_formatter' in settings['cell_text'] else str(mat[y,x])
				plt.text(x, y, text, va = va, ha = ha, size = size, color = text_color)
	axes = plt.axes()
	if 'xticklabels' in settings:
		if 'labels' in settings['xticklabels']:
			labels = settings['xticklabels']['labels']
			axes.set_xticks(range(len(labels)))
			axes.set_xticklabels(labels)
		if 'location' in settings['xticklabels']:
			location = settings['xticklabels']['location']
			# By default it will be at the bottom, so only regarding case of top location
			if location == 'top':
				axes.xaxis.tick_top()
		if 'rotation' in settings['xticklabels']:
			rotation = settings['xticklabels']['rotation']
			plt.xticks(rotation = rotation)
	if 'yticklabels' in settings:
		if 'labels' in settings['yticklabels']:
			labels = settings['yticklabels']['labels']
			axes.set_yticks(range(len(labels)))
			axes.set_yticklabels(labels)
		if 'location' in settings['yticklabels']:
			location = settings['yticklabels']['location']
			# By default it will be at the left, so only regarding case of right location
			if location == 'right':
				axes.yaxis.tick_right()
		if 'rotation' in settings['yticklabels']:
			rotation = settings['yticklabels']['rotation']
			plt.yticks(rotation = rotation)
	
	if dst_path is None:
		plt.show()
	else:
		plt.savefig(dst_path, bbox_inches='tight')
	

def conf_matrix_example():
	mat = np.zeros((5,8))
	for y in range(mat.shape[0]):
		for x in range(mat.shape[1]):
			mat[y,x] = y * x / float((mat.shape[0] - 1) * (mat.shape[1] - 1))
	
	
	settings = {
		'figsize' : (8,5),
		'colormap' : 'magma_r',
		'colorbar' : {
			'view' : True,
			'arange' : (0, 1.001, 0.1),
			'text_formatter' : lambda tick_value : '{0:.0f}%'.format(tick_value*100),
		},
		'xticklabels' : {
			'labels' : ['aaaa', 'bbbbb', 'cccccc', 'ddddd', 'eeee', 'ffff', 'gggg', 'hhhhh'],
			'location' : 'top',
			'rotation' : 45,
		},
		'yticklabels' : {
			'labels' : ['ZZZZZZ', 'YYYYYY', 'XXXXXXX', 'WWWWWWW', 'VVVVVVV'],
		},
		'cell_text' : {
			'vertical_alignment' : 'center',
			'horizontal_alignment' : 'center',
			'size' : 'x-large',
			'color_function' : lambda cell_value : 'black' if cell_value < 0.5 else 'white',
			'text_formatter' : lambda cell_value : '{0:.0f}%'.format(cell_value*100),
		},
	}
	
	conf_matrix_creator(mat, settings)


if __name__ == '__main__':
	conf_matrix_example()