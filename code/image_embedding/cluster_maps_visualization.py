import pickle
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from skimage.filters import gaussian
from skimage.morphology import label
from skimage import io
from skimage import filters

main_tag = '0'
# 'DIV', 'tab' or 'www.linuxmint.com'
vis_type = './label'
# './tag' or './class' or './website'

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return color_list[:,:-1]
    #return base.from_list(cmap_name, color_list, N)

def random_color():
	r = lambda : random.randint(0,255)
	return (r(),r(),r())

def hsv_random_color():
	r = random.randint(0,255)
	hsv = (r*1.0/256, 0.7, 0.9)
	rgb = colorsys.hsv_to_rgb(*hsv)
	#print (rgb)
	return rgb

# Normalizes a matrx using the min-max method
def normalize(matrix):
	minimum = np.amin(matrix)
	maximum = np.amax(matrix)
	return (matrix - minimum) / (maximum - minimum)

def percentage_histograms(tag_maps):
	# Compute the tag count map - a map with the total number of activations per neuron
	no_rows = tag_maps[main_tag].shape[0]
	no_cols = tag_maps[main_tag].shape[1]

	tag_count_map = np.ones((no_rows, no_cols))
	for tag, tag_map in tag_maps.items():
		tag_count_map = tag_count_map + tag_map
	#print (tag_count_map)

	# Compute percentage_maps
	percentage_maps = {}
	for tag, tag_map in tag_maps.items():
		percentage_map = np.round((tag_map / tag_count_map), 2)
		percentage_maps[tag] = percentage_map
		#print (tag, percentage_map)

		# Compute and save histogram
		histo = np.histogram(percentage_map, bins = 7, range = (0.0, 1.0))
		histo = (np.append(histo[0],0), histo[1])
		plt.xlim((0,100))
		plt.ylim((0,100))
		plt.xticks(np.arange(0, 100, step = 10), fontsize = 12)
		plt.yticks(np.arange(0, 100, step = 10), fontsize = 12)
		plt.plot(histo[1]*100.0, histo[0]*100.0/(no_rows*no_cols), linewidth = 3.0)
		plt.legend([tag + ' Tag'], fontsize = 16)
		plt.xlabel('Activation Level', fontsize = 14)
		plt.ylabel('Percentage of Neurons', fontsize = 14)
		plt.savefig(vis_type + '/tag_percentage_histo/' + tag + '.png')
		plt.clf()

def upscale_map(matrix, upscale):
	no_rows = matrix.shape[0]
	no_cols = matrix.shape[1]
	large_matrix = []
	for row in range(no_rows):
		upscaled_row = []
		for col in range(no_cols):
			for _ in range(upscale):
				upscaled_row.append(matrix[row, col])
		for _ in range(upscale):
			large_matrix.append(upscaled_row)
	large_matrix = np.array(large_matrix)
	return large_matrix

# Compute upscaled percentage maps
def upscale_smooth_maps(percentage_maps, upscale, smooth):
	no_rows = percentage_maps[main_tag].shape[0]
	no_cols = percentage_maps[main_tag].shape[1]

	large_percentage_maps = {}
	upscale = 10
	for tag, percentage_map in percentage_maps.items():
		# Upscale
		large_percentage_map = []
		for row in range(no_rows):
			upscaled_row = []
			for col in range(no_cols):
				for _ in range(upscale):
					upscaled_row.append(percentage_map[row, col])
			for _ in range(upscale):
				large_percentage_map.append(upscaled_row)
		large_percentage_map = np.array(large_percentage_map)

		# Smooth
		large_percentage_map = np.round(gaussian(large_percentage_map, sigma = smooth), 2)
		large_percentage_maps[tag] = large_percentage_map

	return large_percentage_maps

def percentage_map(tag_maps):
	not_activated = 'THIS IS LOCATION NOT ACTIVATED'

	# Compute the tag count map - a map with the total number of activations per neuron
	no_rows = tag_maps[main_tag].shape[0]
	no_cols = tag_maps[main_tag].shape[1]

	tag_count_map = np.ones((no_rows, no_cols))
	for tag, tag_map in tag_maps.items():
		tag_count_map = tag_count_map + tag_map
	#print (tag_count_map)

	# Compute percentage_maps
	percentage_maps = {}
	for tag, tag_map in tag_maps.items():
		percentage_map = (tag_map / tag_count_map)
		percentage_maps[tag] = percentage_map
		#print (tag, percentage_map)

	upscale = 10
	smooth = 6.0
	percentage_maps = upscale_smooth_maps(percentage_maps, upscale, smooth)
	no_rows = percentage_maps[main_tag].shape[0]
	no_cols = percentage_maps[main_tag].shape[1]

	# For each neuron, compute the strongest activated tag (object type)
	strongest_tag_map = [[[not_activated, 0.0] for _ in range(no_cols)] for _ in range(no_rows)]
	strongest_tag_map = np.array(strongest_tag_map)
	#print (strongest_tag_map)
	for tag, percentage_map in percentage_maps.items():
		for row in range(no_rows):
			for col in range(no_cols):
				if percentage_map[row,col] > float(strongest_tag_map[row,col,1]):
					strongest_tag_map[row,col,1] = str(percentage_map[row,col])
					strongest_tag_map[row,col,0] = tag

	#print (strongest_tag_map[:,:,0])

	# Index the map in order to apply connected components
	index_dict = {}
	reverse_index_dict = {}
	index_dict[not_activated] = 0
	reverse_index_dict[0] = not_activated
	index = 1
	for tag, _ in percentage_maps.items():
		index_dict[tag] = index
		reverse_index_dict[index] = tag
		index += 1

	index_map = []
	for row in range(no_rows):
		index_row = []
		for col in range(no_cols):
			index_row.append(index_dict[strongest_tag_map[row,col,0]])
		index_map.append(index_row)
	index_map = np.array(index_map)
	#print (index_map)

	# Run connected components
	labels = label(index_map, connectivity = 2)
	#print (labels)

	# Color the percentage map
	color_dict = {}
	color_map = []
	num_colors = 10
	color_list = discrete_cmap(num_colors, 'jet')
	for row in range(no_rows):
		color_row = []
		for col in range(no_cols):
			cc_label = labels[row, col]
			if not cc_label in color_dict:
				color_label = int(strongest_tag_map[row,col,0])
				color_dict[cc_label] = color_list[color_label]
				#color_dict[cc_label] = hsv_random_color()
				if cc_label == 0:
					color_dict[cc_label] = (0.0, 0.0, 0.0)
			color_row.append(color_dict[cc_label])
		color_map.append(color_row)
	color_map = np.array(color_map)
	#print (color_map.shape)
	io.imsave(vis_type + '/tag_perc_map.png', color_map)

	# Save individual percentage maps
	for tag, percentage_map in percentage_maps.items():
		tag_color_map = np.copy(color_map)
		for c in range(3):
			tag_color_map[:, :, c] = np.where(strongest_tag_map[:,:,0] == tag, color_map[:, :, c], 0.0)
		io.imsave(vis_type + '/tag_percentage_maps/' + tag + '.png', tag_color_map)

	return color_map

# Computes the visualization of the clusters identified by the Self-Organizing Map
def activation_map(tag_maps):
	# Compute the tag count map - a map with the total number of activations per neuron
	no_rows = tag_maps[main_tag].shape[0]
	no_cols = tag_maps[main_tag].shape[1]

	tag_count_map = np.ones((no_rows, no_cols))
	for tag, tag_map in tag_maps.items():
		tag_count_map = tag_count_map + tag_map
	#print (tag_count_map)

	# Compute percentage_maps
	percentage_maps = {}
	for tag, tag_map in tag_maps.items():
		percentage_map = (tag_map / tag_count_map)
		percentage_maps[tag] = percentage_map
		#print (tag, percentage_map)

	# Assign random colors
	color_dict = {}
	num_colors = 10
	color_list = discrete_cmap(10, 'jet')
	for tag, _ in tag_maps.items():
		color_label = int(tag)
		color_dict[tag] = color_list[color_label]
		#color_dict[tag] = hsv_random_color()

	# Compute activation maps
	activation_maps = {}
	total_activation_map = np.zeros((no_rows, no_cols, 3))
	for tag, percentage_map in percentage_maps.items():
		activation_map = np.zeros((no_rows, no_cols, 3))
		for row in range(no_rows):
			for col in range(no_cols):
				for c in range(3):
					activation_map[row,col,c] += percentage_map[row, col] * color_dict[tag][c]

		io.imsave(vis_type + '/tag_activation_maps/' + tag + '.png', activation_map)
		total_activation_map += activation_map
		activation_maps[tag] = activation_map
	
	io.imsave(vis_type + '/tag_activ_map.png', total_activation_map)
	return total_activation_map

def smooth_u(matrix):
	no_rows = matrix.shape[0]
	no_cols = matrix.shape[1]
	new_matrix = np.zeros((no_rows, no_cols))
	add = [(-1,0), (0,1), (1,0), (0,-1)]
	for row in range(no_rows):
		for col in range(no_cols):
			#print (matrix[row, col])
			if matrix[row, col] < 0.1:
				sn = 0.0
				cn = 0.0
				for k in range(len(add)):
					new_row = row + add[k][0]
					new_col = col + add[k][1]
					if new_row >= 0 and new_row < no_rows and new_col >= 0 and new_col < no_cols:
						sn += matrix[new_row, new_col]
						cn += 1.0
				new_matrix[row, col] = sn/cn
			else:
				new_matrix[row, col] = matrix[row, col]
	return new_matrix

def topology_landscape(matrix, typename):
	no_rows = matrix.shape[0]
	no_cols = matrix.shape[1]
	i = np.linspace(0,no_rows,no_rows)
	j = np.linspace(0,no_cols, no_cols)
	iv, jv = np.meshgrid(i, j, indexing = 'ij')

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(iv, jv, matrix, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.savefig(vis_type + '/' + typename + '_plot.png')
	plt.clf()

def sw_activation_map(tag_maps):
	# Compute the tag count map - a map with the total number of activations per neuron
	no_rows = tag_maps[main_tag].shape[0]
	no_cols = tag_maps[main_tag].shape[1]

	tag_count_map = np.ones((no_rows, no_cols))
	for tag, tag_map in tag_maps.items():
		tag_count_map = tag_count_map + tag_map
	sw_activation_map = normalize(tag_count_map)
	topology_landscape(sw_activation_map, 'sw_landscape')
	io.imsave(vis_type + '/sw_activation_map.png', sw_activation_map)

# Computes the Unified Distance Matrix from the neural grid (image grid)
def topology_map(image_grid, upscale, smooth):
	no_lines = len(image_grid)
	no_columns = len(image_grid[0])

	moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
	u_matrix = np.zeros((2*no_lines-1, 2*no_columns-1), dtype = np.float16)

	print ('Minimum image grid weight', np.min(image_grid))
	print ('Maximum image grid weight', np.max(image_grid))

	for line in range(no_lines):
		for column in range(no_columns):
			for move in moves:
				new_line = line + move[0]
				new_column = column + move[1]
				if new_line < 0 or new_column < 0:
					continue
				if new_line >= no_lines or new_column >= no_columns:
					continue
				u_mat_line = 2 * line + move[0]
				u_mat_column = 2 * column + move[1]
				u_matrix[u_mat_line][u_mat_column] = \
					np.linalg.norm(image_grid[line][column] - image_grid[new_line][new_column])

	u_matrix = normalize(u_matrix)
	print (u_matrix)

	topology_map = smooth_u(u_matrix)
	fig, ax = filters.try_all_threshold(topology_map)
	plt.savefig(vis_type + '/thresholding.png')
	plt.clf()
	topology_landscape(topology_map, 'landscape')
	#topology_map = upscale_map(topology_map, upscale)
	#topology_map = np.round(gaussian(topology_map, sigma = smooth), 2)
	io.imsave(vis_type + '/u_matrix.png', u_matrix)
	io.imsave(vis_type + '/topology_map.png', topology_map)
	return u_matrix


input_dir = './data/'

random.seed(0)
print (vis_type)
tag_maps_location = input_dir + vis_type + '_maps.p'
tag_maps = pickle.load(open(tag_maps_location, 'rb'))
print (tag_maps.keys())
sw_activation_map(tag_maps)

activation_map(tag_maps)
percentage_histograms(tag_maps)
percentage_map(tag_maps)
image_grid_location = input_dir + 'image_grid.p'
image_grid = pickle.load(open(image_grid_location, 'rb'))
topology_map(image_grid, 5, 1.0)
