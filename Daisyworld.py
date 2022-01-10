#!/usr/bin/env python


import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import convolve
import scipy.stats
from itertools import product
from random import sample

np.set_printoptions(precision=3)


# this is the standard deviation used to calculate the Gaussian kernel for determining the local heatmap
SIGMA_TEMP = 3

# standard deviation of offspring probability matrix
SIGMA_OFFSPRING = 2

# heat transfer coefficient
Q = 38

BASELINE_SOLAR_LUMINOSITY = 1.0
SOLAR_LUMINOSITY_ROC = 0.02 # rate of change of the climate (if climate-change == True when creating model)

MUTATION_PROB = 0.1 # probability that a mutation occurs
# if a mutation occurs to the mean or standard deviation of the optimal growth curve, constrain the range to keep things realistic
MUTATION_MEAN_RANGE = [12, 28]
MUTATION_STDDEV_RANGE = [0, 5]

SURFACE_ALBEDO = 0.5

def gaussian_func(x, a, b, c):
	''' Returns f(x) for arbitrary real constants:
		a = height of the peak
		b = position of the center of the peark
		c = standard deviation (must be non-zero)

		https://en.wikipedia.org/wiki/Gaussian_function
	'''

	return a * np.exp(- ((x - b)**2 / (2 * (c**2))))

class Daisyworld(object):
	"""docstring for Daisyworld"""
	def __init__(self, size, N, local_temp_regulation, climate_change, evolution):
		super(Daisyworld, self).__init__()
		self.size = size

		# this are flags for evolution, local temp regulation, and climate change
		self.evolution = evolution
		self.local_temp_regulation = local_temp_regulation
		self.climate_change = climate_change

		self.solar_luminosity = BASELINE_SOLAR_LUMINOSITY

		self.surface_albedo = SURFACE_ALBEDO

		# this holds Daisy objects, if present on a given tile
		self.landscape = np.empty((size, size), dtype=Daisy)
		# self.populate_landscape()
		#self.populate_landscape_random(N)
		self.populate_landscape_cluster(N, num_cells_row=2, num_cells_col=2)

		# this is a 2D array of albedo values
		self.albedo_map = None
		self.update_albedo_map()

		# this is a 2D array of local temperature
		self.temp_map = None
		self.update_temp_map()

		# this is a 2D array that is used to plot things; i.e. it doesn't have any model data
		self.image_data = None
		self.update_image_data()

		self.black_offspring_probability = None
		self.white_offspring_probability = None
		self.update_probability_maps()

		self.epoch = 0

		self.next_daisy_id = 0

		# this is a dictionary that stores two lists.
		# each list is a running history of the number of daisies of each type
		self.black_daisy_history = {'num' : [],
									'average_mean' : [],
									'average_stddev' : [],
									'average_health' : []}
		self.white_daisy_history = {'num' : [],
									'average_mean' : [],
									'average_stddev' : [],
									'average_health' : []}
		self.update_daisy_history()

		self.average_temp_history = []
		self.solar_luminosity_history = []
		self.update_temp_and_luminosity_history()


	def increase_global_temp(self, amount):
		self.global_temp = self.global_temp + amount

	def decrease_global_temp(self, amount):
		self.global_temp = self.global_temp - amount

	def update_temp_and_luminosity_history(self):

		# print("Average Temperature: {}".format(np.mean(self.temp_map)))

		# get average of temperature map
		self.average_temp_history.append(np.mean(self.temp_map))
		self.solar_luminosity_history.append(self.solar_luminosity)
		

	def update_daisy_history(self):

		# running totals for black daisies
		black_current_num = 0
		black_current_mean = 0
		black_current_stddev = 0
		black_current_health = 0

		# running totals for white daisies
		white_current_num = 0
		white_current_mean = 0
		white_current_stddev = 0
		white_current_health = 0

		# count the daisies
		for row_idx in range(self.size):
			for col_idx in range(self.size):
				if type(self.landscape[row_idx, col_idx]) is Daisy:
					if self.landscape[row_idx, col_idx].color == 'black':
						black_current_num = black_current_num + 1
						black_current_mean = black_current_mean + self.landscape[row_idx, col_idx].optimal_growth_mean
						black_current_stddev = black_current_stddev + self.landscape[row_idx, col_idx].optimal_growth_stddev
						black_current_health = black_current_health + self.landscape[row_idx, col_idx].health

					if self.landscape[row_idx, col_idx].color == 'white':
						white_current_num = white_current_num + 1
						white_current_mean = white_current_mean + self.landscape[row_idx, col_idx].optimal_growth_mean
						white_current_stddev = white_current_stddev + self.landscape[row_idx, col_idx].optimal_growth_stddev
						white_current_health = white_current_health + self.landscape[row_idx, col_idx].health

		# update black running totals
		self.black_daisy_history['num'].append(black_current_num)
		if black_current_num == 0:
			self.black_daisy_history['average_mean'].append(0)
			self.black_daisy_history['average_stddev'].append(0)
			self.black_daisy_history['average_health'].append(0)
		else:
			self.black_daisy_history['average_mean'].append(black_current_mean / black_current_num)
			self.black_daisy_history['average_stddev'].append(black_current_stddev / black_current_num)
			self.black_daisy_history['average_health'].append(black_current_health / black_current_num)

		# update white running totals
		self.white_daisy_history['num'].append(white_current_num)
		if white_current_num == 0:
			self.white_daisy_history['average_mean'].append(0)
			self.white_daisy_history['average_stddev'].append(0)
			self.white_daisy_history['average_health'].append(0)
		else:
			self.white_daisy_history['average_mean'].append(white_current_mean / white_current_num)
			self.white_daisy_history['average_stddev'].append(white_current_stddev / white_current_num)
			self.white_daisy_history['average_health'].append(white_current_health / white_current_num)


	def populate_landscape(self):

		self.landscape[2,2] = Daisy("black", unique_id=0, health=1, local_temp_regulation=self.local_temp_regulation)
		self.landscape[2,4] = Daisy("black", unique_id=1, health=1, local_temp_regulation=self.local_temp_regulation)
		self.landscape[3,3] = Daisy("black", unique_id=2, health=0.75, local_temp_regulation=self.local_temp_regulation)
		self.landscape[3,5] = Daisy("black", unique_id=3, health=0.5, local_temp_regulation=self.local_temp_regulation)

		self.landscape[10,10] = Daisy("white", unique_id=4, health=0.25, local_temp_regulation=self.local_temp_regulation)
		self.landscape[11,12] = Daisy("white", unique_id=5, health=1, local_temp_regulation=self.local_temp_regulation)
		self.landscape[12,11] = Daisy("white", unique_id=6, health=0.5, local_temp_regulation=self.local_temp_regulation)
		self.landscape[5,7] = Daisy("white", unique_id=7, health=0.5, local_temp_regulation=self.local_temp_regulation)

		self.next_daisy_id = 8

	def populate_landscape_cluster(self, N, num_cells_row=2, num_cells_col=2):

		num_per_cell = int(N / (num_cells_row * num_cells_col))
		# print(num_per_cell)

		num_tiles_per_row = self.size / num_cells_row
		num_tiles_per_col = self.size / num_cells_col

		# this is messy, but is basically divides up the landscape into cells to populate with a certain color of Daisy. It allows for arbitrary number of cells.
		row_coords = []
		col_coords = []
		for row_cell_num in range(num_cells_row-1):
			row_coords.append((int(np.floor(row_cell_num*num_tiles_per_row)), int(np.floor(row_cell_num*num_tiles_per_row + num_tiles_per_row - 1))))
		row_coords.append((int(row_coords[-1][-1] + 1), int(self.size-1)))

		
		for col_cell_num in range(num_cells_col-1):
			col_coords.append((int(np.floor(col_cell_num*num_tiles_per_col)), int(np.floor(col_cell_num*num_tiles_per_col + num_tiles_per_col - 1))))
		col_coords.append((int(col_coords[-1][-1] + 1), int(self.size-1)))

		for row_idx, row_coord in enumerate(row_coords):
			for col_idx, col_coord in enumerate(col_coords):
				# get random unqiue coordinates
				coords = sample(list(product(range(row_coord[0], row_coord[1]), range(col_coord[0], col_coord[1]))), k=num_per_cell)

				# set color
				if row_idx % 2 == 0 and col_idx % 2 == 0:
					# if row number is even and col number is even -> black
					color = 'black'
				elif row_idx % 2 == 0 and col_idx % 2 != 0:
					# if row number is even and col number is odd -> white
					color = 'white'
				elif row_idx % 2 != 0 and col_idx % 2 == 0:
					# if row number is odd and col number is even -> white
					color = 'white'
				else:
					# both row and col and odd -> black
					color = 'black'
				
				
				for index, item in enumerate(coords):
					# sample health from gaussian dist, but always positive
					health = 0.3 * np.random.randn() + 1
					if health > 1:
						health = 1.0 - (health - 1.0)

					# make Daisy
					self.landscape[item[0], item[1]] = Daisy(color, unique_id=index, health=health, local_temp_regulation=self.local_temp_regulation)

					# distribute the age
					self.landscape[item[0], item[1]].age = self.landscape[item[0], item[1]].max_age * np.random.rand()

				

	def populate_landscape_random(self, N=20):
		
		# get random unqiue coordinates
		coords = sample(list(product(range(self.size), repeat=2)), k=N)
		for index, item in enumerate(coords):
			# randomly choose color
			color = np.random.choice(a=['black', 'white'], p=[.5, .5])
			# sample health from gaussian dist, but always positive
			health = 0.3 * np.random.randn() + 1
			if health > 1:
				health = 1.0 - (health - 1.0)

			# make Daisy
			self.landscape[item[0], item[1]] = Daisy(color, unique_id=index, health=health, local_temp_regulation=self.local_temp_regulation)


	def update_albedo_map(self):
		'''Generate a 2D numpy array from the landscape albedo values. Use the surface albedo value if a daisy isn't present in a given tile.'''
		# print("Updating albedo map...")
		self.albedo_map = np.zeros((self.size, self.size))
		for row_idx in range(self.size):
			for col_idx in range(self.size):
				if type(self.landscape[row_idx, col_idx]) is Daisy:
					# if a daisy is present, use the albedo value of that particular daisy
					self.albedo_map[row_idx, col_idx] = self.landscape[row_idx, col_idx].albedo
					# print(self.albedo_map[row_idx, col_idx])
				else:
					# otherwise use the surface albedo level
					self.albedo_map[row_idx, col_idx] = self.surface_albedo
					# print(self.albedo_map[row_idx, col_idx])

		# plot_heatmap(self.albedo_map)

	def update_temp_map(self):
		'''Updates the 2D array that represents the temperature at each landscape tile.
		This is calculated by convolving a Gaussian kernel over the albedo map.'''

		# we'll calculate local albedo using a 5x5 Gaussian smoothing kernel
		# see: https://matthew-brett.github.io/teaching/smoothing_intro.html
		# print("Updating temp map...")
		x = np.arange(-3, 4, 1)
		y = np.arange(-3, 4, 1)
		x2d, y2d = np.meshgrid(x, y)
		# calculate our kernel (i.e. a discrete approximation to the Gaussian function)
		kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * SIGMA_TEMP ** 2))

		# we should probably normalize it
		kernel_2d = kernel_2d / np.sum(kernel_2d)

		# convolve the kernel over the landscape albedo values
		albedo_map = convolve(self.albedo_map, kernel_2d) # this defaults to wrapping

		# create a blank temp map
		self.temp_map = np.empty((self.size, self.size), dtype=float)
		for row_idx in range(self.size):
			for col_idx in range(self.size):
				# update temp based on albedo map
				self.temp_map[row_idx, col_idx] = albedo_map[row_idx, col_idx] * self.solar_luminosity * Q
				# print(self.temp_map[row_idx, col_idx] )

		# plot_heatmap(self.temp_map)


	def update_image_data(self):
		'''This takes a landscape of Daisy objects and returns an array with scalar data
		that can be consumed by matplotlib.pyplot.imshow (the values will be mapped to colors)'''
		# print("Updating image data...")
		self.image_data = np.zeros((self.size, self.size))
		for row_idx in range(self.size):
			for col_idx in range(self.size):
				if type(self.landscape[row_idx, col_idx]) is Daisy:
					self.image_data[row_idx, col_idx] = float(self.landscape[row_idx, col_idx])

	def update_probability_maps(self):

		self.black_offspring_probability  = self.calculate_probability_map("black")
		self.white_offspring_probability = self.calculate_probability_map("white")


	def get_daisy_health_map(self, daisy_color):
		# print("Getting daisy health map...")
		health_map = np.zeros((self.size, self.size))
		for row_idx in range(self.size):
			for col_idx in range(self.size):
				if type(self.landscape[row_idx, col_idx]) is Daisy and self.landscape[row_idx, col_idx].color == daisy_color:
					health_map[row_idx, col_idx] = self.landscape[row_idx, col_idx].health

		return health_map


	def calculate_probability_map(self, daisy_color):
		'''For a given color, use a Gaussian kernel convolution to calculate the probability at each square that a new daisy will be seeded.
		Locations with lots of neighboring daisies (of a given color) will be likely to be seeded with a new daisy'''
		# print("Getting propability map...")
		# use a 5x5 Gaussian kernel
		x = np.arange(-4, 5, 1)
		y = np.arange(-4, 5, 1)
		x2d, y2d = np.meshgrid(x, y)
		# calculate our kernel (i.e. a discrete approximation to the Gaussian function)
		# no need to normalize this kernel because we'll be normalizing our probability distribution
		kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * SIGMA_OFFSPRING ** 2))
		# kernel_2d = kernel_2d / np.sum(kernel_2d)

		# get the daisy health map for a given color
		health_map = self.get_daisy_health_map(daisy_color)

		if np.sum(health_map) < 0.001:
			return np.zeros((self.size, self.size))

		# convolve the kernel over the landscape health values
		probability_map = convolve(health_map, kernel_2d) # this defaults to wrapping
		probability_map = probability_map / np.sum(probability_map)

		return probability_map


	def update(self):
		'''
		'''
		# print("\tepoch = {}".format(self.epoch))
		# first create a copy of the landscape; this is so we can update everything "synchonously"
		# we do a deep copy so we make copies of the daisy objects, which are mutable
		# new_landscape = landscape.deepcopy()
		new_landscape = np.empty((self.size, self.size), dtype=Daisy)
		
		# plot_heatmap(black_offspring_probability)
		# plot_heatmap(white_offspring_probability)

		# first let's populate new daisies
		for row_idx in range(self.size):
			for col_idx in range(self.size):

				# if there is a daisy....
				if type(self.landscape[row_idx, col_idx]) is Daisy:
					# print("{}, {}: Daisy".format(row_idx, col_idx))

					# first make sure the Daisy isn't too old
					if self.landscape[row_idx, col_idx].age > self.landscape[row_idx, col_idx].max_age:
						# don't copy daisy to next landscape
						continue

					# next update the daisy age
					self.landscape[row_idx, col_idx].age = self.landscape[row_idx, col_idx].age + 1

					local_temp = self.temp_map[row_idx, col_idx]
					# print("\tLocal temp: ", local_temp)

					# adjust daisy health
					new_health = self.landscape[row_idx, col_idx].adjust_health(local_temp)
					# print("\tNew Health", new_health)

					# if it's still alive, copy to new landscape
					if new_health > 0:
						new_landscape[row_idx, col_idx] = self.landscape[row_idx, col_idx]
					# else:
					# 	print("\t\t***Daisy died from low health***\n")
				
				# if there's no daisy...
				else:
					# calculate the odds that a new daisy of each color would be populated here
					# remember that this takes into account daisy health
					# check if less then zero because sometimes the probabilities come back as very small numbers just under zero, and we can't have negative probs
					black_daisy_prob = self.black_offspring_probability[row_idx, col_idx]
					if black_daisy_prob < 0.0001:
						black_daisy_prob = 0
					white_daisy_prob = self.white_offspring_probability[row_idx, col_idx]
					if white_daisy_prob < 0.0001:
						white_daisy_prob = 0
					leftover_prob = 1 - black_daisy_prob - white_daisy_prob

					choices = ['black', 'white', 'none']
					probability_vector = [black_daisy_prob, white_daisy_prob, leftover_prob]
					# print("\t", probability_vector)

					outcome = np.random.choice(a=choices, p=probability_vector)
					if outcome in ['black', 'white']:

						# make sure there are at least two daisies for the selected color
						if outcome == 'black' and self.black_daisy_history['num'][-1] < 2:
							continue
						if outcome == 'white' and self.white_daisy_history['num'][-1] < 2:
							continue

						# create a new daisy....
						if self.evolution:
							# play GOD
							# let's assume that because this daisy is being spawned, it is the product of the two closest parent Daisies.
							# get the mean and standard deviation for the growth curve of each parent
							potential_parents = [{'distance' : np.inf}, {'distance' : np.inf}]
							for row_parent_idx in range(self.size):
								for col_parent_idx in range(self.size):
									if type(self.landscape[row_parent_idx, col_parent_idx]) is Daisy:

										# calculate distance
										potential_parent = {'unique_id' : self.landscape[row_parent_idx, col_parent_idx].unique_id,
														'distance' : np.sqrt((row_idx - row_parent_idx)**2 + (col_idx - col_parent_idx)**2),
														'optimal_growth_mean' : self.landscape[row_parent_idx, col_parent_idx].optimal_growth_mean,
														'optimal_growth_stddev' : self.landscape[row_parent_idx, col_parent_idx].optimal_growth_stddev}

										if potential_parent['distance'] < potential_parents[0]['distance'] and not np.isinf(potential_parents[1]['distance']):
											# check the first one first, assuming the second one has already been replaced
											potential_parents[0] = potential_parent
										elif potential_parent['distance'] < potential_parents[1]['distance']:
											# then check the second one!
											potential_parents[1] = potential_parent

							parent_means = [potential_parents[0]['optimal_growth_mean'], potential_parents[1]['optimal_growth_mean']]
							parent_stddevs = [potential_parents[0]['optimal_growth_stddev'], potential_parents[1]['optimal_growth_stddev']]

							# choose a mean and a standard deviation (equal chance of inheriting from either parent)
							child_mean = np.random.choice(a=parent_means, p=[0.5, 0.5])
							child_stddev = np.random.choice(a=parent_stddevs, p=[0.5, 0.5])

							if np.random.rand() < MUTATION_PROB:
								# mutate the mean (choose a new random value from the specified range)
								child_mean = MUTATION_MEAN_RANGE[0] + (np.random.rand() * (MUTATION_MEAN_RANGE[1] - MUTATION_MEAN_RANGE[0]))

							if np.random.rand() < MUTATION_PROB:
								# mutate the std dev
								child_stddev = MUTATION_STDDEV_RANGE[0] + (np.random.rand() * (MUTATION_STDDEV_RANGE[1] - MUTATION_STDDEV_RANGE[0]))

							new_landscape[row_idx, col_idx] = Daisy(color=outcome,
																	optimal_growth_mean=child_mean,
																	optimal_growth_stddev=child_stddev,
																	unique_id=self.next_daisy_id,
																	local_temp_regulation=self.local_temp_regulation)
							self.next_daisy_id = self.next_daisy_id + 1

						else:
							# eh, no evolution this time around -- just use the defaults, dammit
							new_landscape[row_idx, col_idx] = Daisy(color=outcome, unique_id=self.next_daisy_id, local_temp_regulation=self.local_temp_regulation)
							self.next_daisy_id = self.next_daisy_id + 1
							# print("\tCreating new {} daisy".format(outcome))


		# update the landscape
		self.landscape = new_landscape

		# update albedo, temp, probability, daisy history, and image data
		self.update_albedo_map()
		self.update_temp_map()
		self.update_probability_maps()
		self.update_daisy_history()
		self.update_image_data()
		self.update_temp_and_luminosity_history()

		# cycle the solar luminosity (but give the system time to hit an equilibrium first)
		if self.climate_change and self.epoch > 100:
			self.solar_luminosity = np.sin((self.epoch-100)*SOLAR_LUMINOSITY_ROC)*0.15 + BASELINE_SOLAR_LUMINOSITY

		self.epoch = self.epoch + 1

class Daisy(object):
	"""docstring for Daisy"""
	def __init__(self, color, unique_id, local_temp_regulation, optimal_growth_mean=None, optimal_growth_stddev=None, health=1.0):
		super(Daisy, self).__init__()
		self.color = color

		self.unique_id = unique_id

		self.local_temp_regulation = local_temp_regulation

		if color not in ['black', 'white']:
			raise ValueError('Daisy color must be black or white.')

		# set the optimal growth mean, with a default depending on Daisy color
		if optimal_growth_mean == None:
			if color == 'white':
				# black d
				self.optimal_growth_mean = 24
			elif color == 'black':
				self.optimal_growth_mean = 15
		else:
			self.optimal_growth_mean = optimal_growth_mean

		# set the optimal growth standard deviation, with a default depending on Daisy color
		if optimal_growth_stddev == None:
			if color == 'white':
				self.optimal_growth_stddev = 3
			elif color == 'black':
				self.optimal_growth_stddev = 3
		else:
			self.optimal_growth_stddev = optimal_growth_stddev

		# row and column in the Daisyworld landscape
		self.location = [None, None]

		# Daisy health is a value between 0 and 1
		self.health = health

		self.age = 0

		self.max_age = 10 * np.random.randn() + 75 # max time steps daisy is alive for

		if self.local_temp_regulation:
			if self.color == "black":
				self.albedo = 0.15
			elif self.color == "white":
				self.albedo = 0.85
		else:
			self.albedo = SURFACE_ALBEDO

		# print("New Daisy with albedo {}".format(self.albedo))


	def adjust_health(self, local_temp):

		# print("Current health: {}".format(self.health))

		# Use a gaussian function to determine how much health should adjust based on distance from optimal value
		adjustment = gaussian_func(local_temp, 1.4, self.optimal_growth_mean, self.optimal_growth_stddev)

		# print("\nDaisy color: {}".format(self.color))
		# print("Local Temp: {}".format(local_temp))
		# print("Optimal Temp: {}".format(self.optimal_growth_mean))
		# print("Adjustment: {}".format(adjustment))

		# print("Old health: {}".format(self.health))

		new_health = self.health * adjustment


		if new_health > 1.0:
			# max health value at 1
			self.health = 1.0
		elif new_health < 0.01:
			self.health = 0.0
		else:
			self.health = new_health

		# print("New health: {}".format(self.health))

		return self.health


	def set_optimal_growth_mean(self, mean):
		self.optimal_growth_mean = mean

	def set_optimal_growth_stddev(self, stddev):
		self.optimal_growth_stddev = stddev

	def __str__(self):
		if self.color == "black":
			return "b"
		else:
			return "w"

	def __float__(self):
		if self.color == "black":
			return -float(self.health) / 2 - 0.5
		else:
			return float(self.health) / 2 + 0.5

	def __repr__(self):
		if self.color == "black":
			return "b"
		else:
			return "w"

def plot_heatmap(data):
	'''This is currently used for testing purposes.'''

	fig, ax = plt.subplots()
	img = ax.imshow(data, interpolation='nearest')
	plt.show()



def update(frameNum, images, lines, lines_axs, daisyworld):
	'''This is basically a wrapper function that advances the simulation, and gets image data to visualize'''

	# advance the model
	daisyworld.update()

	# update image data
	images[0].set_data(daisyworld.image_data)
	images[1].set_data(daisyworld.temp_map)

	# update line data: population
	lines[0].set_data(list(range(daisyworld.epoch+1)), daisyworld.black_daisy_history['num'])
	lines[1].set_data(list(range(daisyworld.epoch+1)), daisyworld.white_daisy_history['num'])

	# update line data: average temp
	lines[2].set_data(list(range(daisyworld.epoch+1)), daisyworld.average_temp_history)

	# update line data: average mean
	lines[3].set_data(list(range(daisyworld.epoch+1)), daisyworld.black_daisy_history['average_mean'])
	lines[4].set_data(list(range(daisyworld.epoch+1)), daisyworld.white_daisy_history['average_mean'])

	# update line data: average std dev
	lines[6].set_data(list(range(daisyworld.epoch+1)), daisyworld.black_daisy_history['average_stddev'])
	lines[7].set_data(list(range(daisyworld.epoch+1)), daisyworld.white_daisy_history['average_stddev'])

	# update line data: health
	lines[8].set_data(list(range(daisyworld.epoch+1)), daisyworld.black_daisy_history['average_health'])
	lines[9].set_data(list(range(daisyworld.epoch+1)), daisyworld.white_daisy_history['average_health'])

	# set the x axis limits
	for ax in lines_axs:
		if daisyworld.epoch < 100:
			ax.set_xlim([0, daisyworld.epoch+5])
		else:
			ax.set_xlim([daisyworld.epoch-105, daisyworld.epoch+5])

	# set the y limits; we might want these to be more custom, so let's do each individually
	if daisyworld.epoch < 100:
		lines_axs[0].set_ylim([0, max(daisyworld.black_daisy_history['num'] + daisyworld.white_daisy_history['num'])+5])

		lines_axs[1].set_ylim([min(daisyworld.average_temp_history + daisyworld.black_daisy_history['average_mean'] + daisyworld.white_daisy_history['average_mean'])-5,
						max(daisyworld.average_temp_history + daisyworld.black_daisy_history['average_mean'] + daisyworld.white_daisy_history['average_mean'])+10])

		lines_axs[3].set_ylim([min(daisyworld.black_daisy_history['average_stddev'] + daisyworld.white_daisy_history['average_stddev'])-5,
						max(daisyworld.black_daisy_history['average_stddev'] + daisyworld.white_daisy_history['average_stddev'])+5])

	else:
		lines_axs[0].set_ylim([0, max(daisyworld.black_daisy_history['num'][-100:] + daisyworld.white_daisy_history['num'][-100:])+5])

		lines_axs[1].set_ylim([min(daisyworld.average_temp_history[-100:] + daisyworld.black_daisy_history['average_mean'][-100:] + daisyworld.white_daisy_history['average_mean'][-100:])-5,
						max(daisyworld.average_temp_history[-100:] + daisyworld.black_daisy_history['average_mean'][-100:] + daisyworld.white_daisy_history['average_mean'][-100:])+10])

		lines_axs[3].set_ylim([min(daisyworld.black_daisy_history['average_stddev'][-100:] + daisyworld.white_daisy_history['average_stddev'][-100:])-5,
						max(daisyworld.black_daisy_history['average_stddev'][-100:] + daisyworld.white_daisy_history['average_stddev'][-100:])+5])


	return lines


def build_static_plots():

	size = 20 	# landscape size
	N = 50 		# num daisies
	evolution = True
	local_temp_regulation = False
	climate_change = True
	num_epochs = 5000
	num_sims = 20

	T = np.arange(0, num_epochs, 1)

	fig, axs = plt.subplots(2, 2, figsize=(10, 7), dpi=130)

	# init dicts for each plot category
	categories = ['num', 'average_mean', 'average_stddev', 'average_health']

	black_results = {}
	white_results = {}
	for cat in categories:
		black_results[cat] = np.zeros((num_sims, num_epochs))
		white_results[cat] = np.zeros((num_sims, num_epochs))
	temp_results = np.zeros((num_sims, num_epochs))

	for sim_num in range(num_sims):
		print("Sim number: {}".format(sim_num))

		# init Daisyworld
		daisyworld = Daisyworld(size=size, N=N, local_temp_regulation=local_temp_regulation, evolution=evolution, climate_change=climate_change)

		# run simulation
		for i in range(num_epochs-1):
			daisyworld.update()


		for cat in categories:
			black_results[cat][sim_num,] = daisyworld.black_daisy_history[cat]
			white_results[cat][sim_num,] = daisyworld.white_daisy_history[cat]

		temp_results[sim_num,] = daisyworld.average_temp_history

		# plot individual sim results
		axs[0, 0].plot(T, daisyworld.black_daisy_history['average_mean'], color="lightsteelblue", alpha=1, linewidth=0.7)
		axs[0, 0].plot(T, daisyworld.white_daisy_history['average_mean'], color="navajowhite", alpha=1, linewidth=0.7)

		axs[1, 0].plot(T, daisyworld.black_daisy_history['average_stddev'], color="lightsteelblue", alpha=1, linewidth=0.7)
		axs[1, 0].plot(T, daisyworld.white_daisy_history['average_stddev'], color="navajowhite", alpha=1, linewidth=0.7)

		axs[0, 1].plot(T, daisyworld.black_daisy_history['num'], color="lightsteelblue", alpha=1, linewidth=0.7)
		axs[0, 1].plot(T, daisyworld.white_daisy_history['num'], color="navajowhite", alpha=1, linewidth=0.7)

		axs[1, 1].plot(T, daisyworld.black_daisy_history['average_health'], color="lightsteelblue", alpha=1, linewidth=0.7)
		axs[1, 1].plot(T, daisyworld.white_daisy_history['average_health'], color="navajowhite", alpha=1, linewidth=0.7)


	# filter out the zeros
	black_results['average_mean'][black_results['average_mean'] == 0] = np.nan
	white_results['average_mean'][white_results['average_mean'] == 0] = np.nan
	black_results['average_stddev'][black_results['average_stddev'] == 0] = np.nan
	white_results['average_stddev'][white_results['average_stddev'] == 0] = np.nan


	# get mean and std deviation, ignoring zeros
	black_mean_of_mean = np.nanmean(black_results['average_mean'], axis=0)
	black_std_error_of_mean = np.nanstd(black_results['average_mean'], ddof=1, axis=0)
	white_mean_of_mean = np.nanmean(white_results['average_mean'], axis=0)
	white_std_error_of_mean = np.nanstd(white_results['average_mean'], ddof=1, axis=0)
	temp_mean = np.nanmean(temp_results, axis=0)
	temp_std_error = np.nanstd(temp_results, ddof=1, axis=0)

	axs[0, 0].plot(T, black_mean_of_mean, color="blue", alpha=1, linewidth=2, label="Black daisies")
	axs[0, 0].plot(T, white_mean_of_mean, color="orange", alpha=1, linewidth=2, label="White daisies")
	axs[0, 0].errorbar(T[10:], black_mean_of_mean[10:], yerr=black_std_error_of_mean[10:], errorevery=80, capsize=5, fmt='none', elinewidth=1, ecolor="midnightblue", alpha=1, label="Black standard error", zorder=100)
	axs[0, 0].errorbar(T[10:], white_mean_of_mean[10:], yerr=white_std_error_of_mean[10:], errorevery=88, capsize=5, fmt='none', elinewidth=1, ecolor="darkorange", alpha=1, label="White standard error", zorder=100)

	axs[0, 0].plot(T, temp_mean, color="purple", alpha=1, linewidth=2, label="Average landscape temp")

	black_mean_of_stddev = np.nanmean(black_results['average_stddev'], axis=0)
	black_std_error_of_stddev = np.nanstd(black_results['average_stddev'], ddof=1, axis=0)
	white_mean_of_stddev = np.nanmean(white_results['average_stddev'], axis=0)
	white_std_error_of_stddev = np.nanstd(white_results['average_stddev'], ddof=1, axis=0)

	axs[1, 0].plot(T, black_mean_of_stddev, color="blue", alpha=1, linewidth=2, label="Black daisies")
	axs[1, 0].plot(T, white_mean_of_stddev, color="orange", alpha=1, linewidth=2, label="White daisies")
	axs[1, 0].errorbar(T[10:], black_mean_of_stddev[10:], yerr=black_std_error_of_stddev[10:], errorevery=80, capsize=5, fmt='none', elinewidth=1, ecolor="midnightblue", alpha=1, label="Black standard error", zorder=100)
	axs[1, 0].errorbar(T[10:], white_mean_of_stddev[10:], yerr=white_std_error_of_stddev[10:], errorevery=88, capsize=5, fmt='none', elinewidth=1, ecolor="darkorange", alpha=1, label="White standard error", zorder=100)

	black_mean_of_num = np.nanmean(black_results['num'], axis=0)
	black_std_error_of_num = np.nanstd(black_results['num'], ddof=1, axis=0)
	white_mean_of_num = np.nanmean(white_results['num'], axis=0)
	white_std_error_of_num = np.nanstd(white_results['num'], ddof=1, axis=0)

	axs[0, 1].plot(T, black_mean_of_num, color="blue", alpha=1, linewidth=2, label="Black daisies")
	axs[0, 1].plot(T, white_mean_of_num, color="orange", alpha=1, linewidth=2, label="White daisies")
	axs[0, 1].errorbar(T[10:], black_mean_of_num[10:], yerr=black_std_error_of_num[10:], errorevery=80, capsize=5, fmt='none', elinewidth=1, ecolor="midnightblue", alpha=1, label="Black standard error", zorder=100)
	axs[0, 1].errorbar(T[10:], white_mean_of_num[10:], yerr=white_std_error_of_num[10:], errorevery=88, capsize=5, fmt='none', elinewidth=1, ecolor="darkorange", alpha=1, label="White standard error", zorder=100)

	black_mean_of_health = np.nanmean(black_results['average_health'], axis=0)
	black_std_error_of_health = np.nanstd(black_results['average_health'], ddof=1, axis=0)
	white_mean_of_health = np.nanmean(white_results['average_health'], axis=0)
	white_std_error_of_health = np.nanstd(white_results['average_health'], ddof=1, axis=0)

	axs[1, 1].plot(T, black_mean_of_health, color="blue", alpha=1, linewidth=2, label="Black daisies")
	axs[1, 1].plot(T, white_mean_of_health, color="orange", alpha=1, linewidth=2, label="White daisies")
	axs[1, 1].errorbar(T[10:], black_mean_of_health[10:], yerr=black_std_error_of_health[10:], errorevery=80, capsize=5, fmt='none', elinewidth=1, ecolor="midnightblue", alpha=1, label="Black standard error", zorder=100)
	axs[1, 1].errorbar(T[10:], white_mean_of_health[10:], yerr=white_std_error_of_health[10:], errorevery=88, capsize=5, fmt='none', elinewidth=1, ecolor="darkorange", alpha=1, label="White standard error", zorder=100)

	# set the labels. WOW this is getting confusing.
	# axs[0, 0].set_title("Average optimal growth curve mean by color")
	axs[0, 0].set_xlabel("t")
	axs[0, 0].set_ylabel("Optimal growth curve mean")

	# axs[1, 0].set_title("Average optimal growth curve standard deviation by color")
	axs[1, 0].set_xlabel("t")
	axs[1, 0].set_ylabel("Optimal growth curve\nstandard deviation")

	# axs[0, 1].set_title("Number of daisies by color")
	axs[0, 1].set_xlabel("t")
	axs[0, 1].set_ylabel("Number of daisies")

	# axs[1, 1].set_title("Average daisy health by color")
	axs[1, 1].set_xlabel("t")
	axs[1, 1].set_ylabel("Average daisy health")

	axs[0, 0].set_ylim(MUTATION_MEAN_RANGE)
	axs[1, 0].set_ylim(MUTATION_STDDEV_RANGE)

	axs[0, 0].set_xlim([4500, 5000])
	axs[1, 0].set_xlim([4500, 5000])
	axs[0, 1].set_xlim([4500, 5000])
	axs[1, 1].set_xlim([4500, 5000])

	axs[0, 0].grid()
	axs[1, 0].grid()
	axs[0, 1].grid()
	axs[1, 1].grid()
	axs[1, 0].legend()

	plt.show()

def run_animation():

	# parse arguments
	parser = argparse.ArgumentParser(description="Runs a 2D Daisyworld simulation")
 
	# add arguments
	parser.add_argument('--num-daisies', dest='N', required=True)
	parser.add_argument('--grid-size', dest='size', required=False)
	parser.add_argument('--mov-file', dest='movfile', required=False)
	parser.add_argument('--interval', dest='interval', required=False)
	parser.add_argument('--evolution', dest='evolution', required=True)
	parser.add_argument('--epochs', dest='epochs', required=False)
	parser.add_argument('--local-temp-reg', dest='local_temp_regulation', required=True)
	parser.add_argument('--climate-change', dest='climate_change', required=True)
	args = parser.parse_args()

	# number of starting daisies
	N = int(args.N)
	 
	# set landscape size
	size = 20
	if args.size and int(args.size) > 8:
		size = int(args.size)

	# number of epochs
	frames = 1000
	if args.epochs:
		frames = int(args.epochs)

	if args.evolution == 'True':
		evolution = True
	elif args.evolution == 'False':
		evolution = False
	else:
		raise ValueError("'--evolution' command line argument must be True or False")

	if args.local_temp_regulation == 'True':
		local_temp_regulation = True
	elif args.local_temp_regulation == 'False':
		local_temp_regulation = False
	else:
		raise ValueError("'--local-temp-reg' command line argument must be True or False")

	if args.climate_change == 'True':
		climate_change = True
	elif args.climate_change == 'False':
		climate_change = False
	else:
		raise ValueError("'--climate-change' command line argument must be True or False")

	# init Daisyworld
	daisyworld = Daisyworld(size=size, N=N, local_temp_regulation=local_temp_regulation, evolution=evolution, climate_change=climate_change)
	daisyworld.update()

	# set animation update interval (in milliseconds)
	updateInterval = 50
	if args.interval:
		updateInterval = int(args.interval)

	# set up animation
	fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=129)

	plt.title("Evolution = {}\tLocal temp regulation = {}\tClimate change = {}".format(evolution, local_temp_regulation, climate_change))

	images = []
	images.append(axs[0, 0].imshow(daisyworld.image_data, cmap='gray', interpolation='nearest'))
	axs[0, 0].title.set_text("Daisy location")
	axs[0, 0].axis('off')

	images.append(axs[1, 0].imshow(daisyworld.temp_map, vmin=12, vmax=28, cmap='plasma', interpolation='nearest'))
	axs[1, 0].title.set_text("Temperature")
	axs[1, 0].axis('off')
	fig.colorbar(images[1], ax=axs[1, 0])

	lines = []
	# population of each daisy type
	lobj_num_black, = axs[0, 1].plot([], [], label="Black Daisies", color='tab:blue', lw=2)
	lobj_num_white, = axs[0, 1].plot([], [], label="White Daisies", color='tab:orange', lw=2)
	lines.append(lobj_num_black)
	lines.append(lobj_num_white)
	axs[0, 1].legend(loc="upper left")
	axs[0, 1].set_ylabel('Population')
	axs[0, 1].title.set_text("Population by Daisy Type")

	# average temperature landscape
	lobj_avg_temp, = axs[1, 1].plot([], [], label="Average Landscape Temperature", color='tab:purple', lw=2, linestyle='dotted')
	# average optimal growth curve mean by daisy type
	lobj_mean_black, = axs[1, 1].plot([], [], label="Black Daisy Avg Optimal Temp", color='tab:blue', lw=2)
	lobj_mean_white, = axs[1, 1].plot([], [], label="White Daisy Avg Optimal Temp", color='tab:orange', lw=2)

	lines.append(lobj_avg_temp)
	lines.append(lobj_mean_black)
	lines.append(lobj_mean_white)

	axs[1, 1].set_xlabel('Epoch')
	axs[1, 1].set_ylabel('Temperature')
	axs[1, 1].title.set_text("Temperature")
	axs[1, 1].set_ylim([12, 28])
	axs[1, 1].legend(loc="upper left")

	lobj, = axs[0, 2].plot([], [], label="", color='black', lw=2)
	lines.append(lobj)

	# average optimal growth std deviation by daisy type
	lobj_stddev_black, = axs[1, 2].plot([], [], label="Black Daisies", color='tab:blue', lw=2)
	lobj_stddev_white, = axs[1, 2].plot([], [], label="White Daisies", color='tab:orange', lw=2, linestyle='dotted')
	lines.append(lobj_stddev_black)
	lines.append(lobj_stddev_white)
	axs[1, 2].legend(loc="upper left")
	axs[1, 2].set_ylabel('Average OGC S.D.')
	axs[1, 2].title.set_text("Avg optimal growth curve standard deviation")


	# health
	lobj_health_black, = axs[0, 2].plot([], [], label="Black Daisies", color='tab:blue', lw=2)
	lobj_health_white, = axs[0, 2].plot([], [], label="White Daisies", color='tab:orange', lw=2)
	lines.append(lobj_health_black)
	lines.append(lobj_health_white)
	axs[0, 2].legend(loc="upper left")
	axs[0, 2].set_ylabel('Health')
	axs[0, 2].title.set_text("Avg health")

	axs[0, 2].set_ylim([-0.1, 1.1])


	lines_axs = [axs[0, 1], axs[1, 1], axs[0, 2], axs[1, 2], axs[0, 2]]

	# lines_axs = {'num' : axs[0, 1],
	# 			'average_temp' : axs[1, 1],
	# 			'average_mean' : axs[0, 2],
	# 			'average_stddev' : axs[1, 2]}
	

	# fig.tight_layout()


	# This "makes an animation by repeatedly calling a function func."
	# For func: the first argument will be the next value in frames. Any additional positional arguments can be supplied via the fargs parameter.
	ani = animation.FuncAnimation(fig, func=update,
								fargs=(images, lines, lines_axs, daisyworld),
								frames = frames,
								interval=updateInterval,
								repeat=False)
 
	# set output file
	if args.movfile:
		# FFwriter = animation.FFMpegWriter(fps=30)
		ani.save(args.movfile, fps=15, extra_args=['-vcodec', 'libx264'])

 
	plt.show()

def main():

	run_animation()

	# build_static_plots()

if __name__ == '__main__':
	main()





