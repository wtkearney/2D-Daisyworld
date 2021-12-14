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
SIGMA_TEMP = 2

# standard deviation of offspring probability matrix
SIGMA_OFFSPRING = 2

# heat transfer coefficient
Q = 47

BASELINE_SOLAR_LUMINOSITY = 1.0
SOLAR_LUMINOSITY_ROC = 0.05 # rate of change

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
	def __init__(self, size, N):
		super(Daisyworld, self).__init__()
		self.size = size

		self.solar_luminosity = BASELINE_SOLAR_LUMINOSITY

		self.surface_albedo = 0.4

		# this holds Daisy objects, if present on a given tile
		self.landscape = np.empty((size, size), dtype=Daisy)
		# self.populate_landscape()
		self.populate_landscape_random(N)

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
		self.daisy_history = {'black' : [], 'white' : []}
		self.update_daisy_history()

		self.average_temp_history = []
		self.solar_luminosity_history = []
		self.update_temp_and_luminosity_history()


	def increase_global_temp(self, amount):
		self.global_temp = self.global_temp + amount

	def decrease_global_temp(self, amount):
		self.global_temp = self.global_temp - amount

	def update_temp_and_luminosity_history(self):

		# get average of temperature map
		self.average_temp_history.append(np.mean(self.temp_map))
		self.solar_luminosity_history.append(self.solar_luminosity)
		

	def update_daisy_history(self):

		# initialize new counter
		self.daisy_history['black'].append(0)
		self.daisy_history['white'].append(0)

		# count the daisies
		for row_idx in range(self.size):
			for col_idx in range(self.size):
				if type(self.landscape[row_idx, col_idx]) is Daisy:
					if self.landscape[row_idx, col_idx].color == 'black':
						self.daisy_history['black'][-1] = self.daisy_history['black'][-1] + 1
					if self.landscape[row_idx, col_idx].color == 'white':
						self.daisy_history['white'][-1] = self.daisy_history['white'][-1] + 1


	def populate_landscape(self):

		self.landscape[2,2] = Daisy("black", unique_id=0, health=1)
		self.landscape[2,4] = Daisy("black", unique_id=1, health=1)
		self.landscape[3,3] = Daisy("black", unique_id=2, health=0.75)
		self.landscape[3,5] = Daisy("black", unique_id=3, health=0.5)

		self.landscape[10,10] = Daisy("white", unique_id=4, health=0.25)
		self.landscape[11,12] = Daisy("white", unique_id=5, health=1)
		self.landscape[12,11] = Daisy("white", unique_id=6, health=0.5)
		self.landscape[5,7] = Daisy("white", unique_id=7, health=0.5)

		self.next_daisy_id = 8

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
			self.landscape[item[0], item[1]] = Daisy(color, unique_id=index, health=health)


	def update_albedo_map(self):
		'''Generate a 2D numpy array from the landscape albedo values. Use the surface albedo value if a daisy isn't present in a given tile.'''
		# print("Updating albedo map...")
		self.albedo_map = np.zeros((self.size, self.size))
		for row_idx in range(self.size):
			for col_idx in range(self.size):
				if type(self.landscape[row_idx, col_idx]) is Daisy:
					# if a daisy is present, use the albedo value of that particular daisy
					self.albedo_map[row_idx, col_idx] = self.landscape[row_idx, col_idx].albedo
				else:
					# otherwise use the surface albedo level
					self.albedo_map[row_idx, col_idx] = self.surface_albedo

		# plot_heatmap(self.albedo_map)

	def update_temp_map(self):
		'''Updates the 2D array that represents the temperature at each landscape tile.
		This is calculated by convolving a Gaussian kernel over the albedo map.'''

		# we'll calculate local albedo using a 5x5 Gaussian smoothing kernel
		# see: https://matthew-brett.github.io/teaching/smoothing_intro.html
		# print("Updating temp map...")
		x = np.arange(-2, 3, 1)
		y = np.arange(-2, 3, 1)
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
		print("Updating (epoch = {})...".format(self.epoch))
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
					# print("{}, {}: No daisy".format(row_idx, col_idx))
					# calculate the odds that a new daisy of each color would be populated here
					# remember that this takes into account daisy health
					black_daisy_prob = self.black_offspring_probability[row_idx, col_idx]
					white_daisy_prob = self.white_offspring_probability[row_idx, col_idx]
					leftover_prob = 1 - black_daisy_prob - white_daisy_prob

					choices = ['black', 'white', 'none']
					probability_vector = [black_daisy_prob, white_daisy_prob, leftover_prob]
					# print("\t", probability_vector)

					outcome = np.random.choice(a=choices, p=probability_vector)
					if outcome in ['black', 'white']:
						# create a new daisy with default health
						# TODO: mutate growth curve
						new_landscape[row_idx, col_idx] = Daisy(color=outcome, unique_id=self.next_daisy_id)
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

		# cycle the solar luminosity
		self.solar_luminosity = np.sin(self.epoch*SOLAR_LUMINOSITY_ROC)*0.15 + BASELINE_SOLAR_LUMINOSITY

		self.epoch = self.epoch + 1


class Daisy(object):
	"""docstring for Daisy"""
	def __init__(self, color, unique_id, optimal_growth_mean=None, optimal_growth_stddev=None, health=1.0):
		super(Daisy, self).__init__()
		self.color = color

		self.unique_id = unique_id

		if color == 'white':
			# black d
			self.optimal_growth_mean = 24
			self.optimal_growth_stddev = 3
		elif color == 'black':
			self.optimal_growth_mean = 17
			self.optimal_growth_stddev = 3
		else:
			raise ValueError('Daisy color must be black or white.')

		# row and column in the Daisyworld landscape
		self.location = [None, None]

		# Daisy health is a value between 0 and 1
		self.health = health

		self.age = 0

		self.max_age = health = 10 * np.random.randn() + 50 # max time steps daisy is alive for

		if self.color == "black":
			self.albedo = 0.25
		if self.color == "white":
			self.albedo = 0.75

	def adjust_health(self, local_temp):

		# print("Current health: {}".format(self.health))

		# Use a gaussian function to determine how much health should adjust based on distance from optimal value
		adjustment = gaussian_func(local_temp, 1.2, self.optimal_growth_mean, self.optimal_growth_stddev)

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



def update(frameNum, images, lines, population_ax, solar_temp_ax, daisyworld):
	'''This is basically a wrapper function that advances the simulation, and gets image data to visualize'''

	# advance the model
	daisyworld.update()

	# update image data
	images[0].set_data(daisyworld.image_data)
	images[1].set_data(daisyworld.temp_map)

	# update line data
	lines[0].set_data(list(range(daisyworld.epoch+1)), daisyworld.daisy_history['black'])
	lines[1].set_data(list(range(daisyworld.epoch+1)), daisyworld.daisy_history['white'])

	# update lines axis
	if daisyworld.epoch < 100:
		population_ax.set_xlim([0, daisyworld.epoch+5])
		population_ax.set_ylim([0, max(daisyworld.daisy_history['black'] + daisyworld.daisy_history['white'])+5])
	else:
		population_ax.set_xlim([daisyworld.epoch-105, daisyworld.epoch+5])
		population_ax.set_ylim([0, max(daisyworld.daisy_history['black'][-100:] + daisyworld.daisy_history['white'][-100:])+5])

	

	lines[2].set_data(list(range(daisyworld.epoch+1)), daisyworld.average_temp_history)
	# lines[3].set_data(list(range(daisyworld.epoch+1)), daisyworld.solar_luminosity_history)
	
	# update lines axis
	if daisyworld.epoch < 100:
		solar_temp_ax.set_xlim([0, daisyworld.epoch+5])
	else:
		solar_temp_ax.set_xlim([daisyworld.epoch-105, daisyworld.epoch+5])

	# lum_ax.set_ylim([0, max(daisyworld.solar_luminosity_history)+1])


	return lines


def main():

	# parse arguments
	parser = argparse.ArgumentParser(description="Runs Conway's Game of Life simulation.")
 
	# add arguments
	parser.add_argument('--num-daisies', dest='N', required=True)
	parser.add_argument('--grid-size', dest='size', required=False)
	parser.add_argument('--mov-file', dest='movfile', required=False)
	parser.add_argument('--interval', dest='interval', required=False)
	args = parser.parse_args()

	# number of starting daisies
	N = int(args.N)
	 
	# set landscape size
	size = 20
	if args.size and int(args.size) > 8:
		size = int(args.size)

	# init Daisyworld
	daisyworld = Daisyworld(size=size, N=N)
	daisyworld.update()

	# set animation update interval (in milliseconds)
	updateInterval = 100
	if args.interval:
		updateInterval = int(args.interval)

	# set up animation
	fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=129)

	images = []
	images.append(axs[0, 0].imshow(daisyworld.image_data, cmap='gray', interpolation='nearest'))
	axs[0, 0].title.set_text("Daisy location")
	axs[0, 0].axis('off')

	images.append(axs[1, 0].imshow(daisyworld.temp_map, vmin=12, vmax=28, cmap='plasma', interpolation='nearest'))
	axs[1, 0].title.set_text("Temperature")
	axs[1, 0].axis('off')
	fig.colorbar(images[1], ax=axs[1, 0])

	lines = []
	lobj_black, = axs[0, 1].plot([], [], label="Black Daisies", lw=2)
	lobj_white, = axs[0, 1].plot([], [], label="White Daisies", lw=2)
	lines.append(lobj_black)
	lines.append(lobj_white)
	axs[0, 1].legend(loc="upper left")
	axs[0, 1].set_ylabel('Population')
	axs[0, 1].title.set_text("Population by Daisy Type")


	lobj_avg_temp, = axs[1, 1].plot([], [], label="Average Temperature", color='tab:purple', lw=2)
	lines.append(lobj_avg_temp)
	axs[1, 1].set_xlabel('Epoch')
	axs[1, 1].set_ylabel('Temperature')
	axs[1, 1].title.set_text("Average Landscape Temperature")
	axs[1, 1].set_ylim([12, 28])


	# lum_ax = axs[1, 1].twinx()
	# lobj_solar, = lum_ax.plot([], [], label="Solar Luminosity")
	# lines.append(lobj_solar)
	# lum_ax.set_ylabel('Luminosity', color='tab:red')
	# axs[1, 1].legend(loc="upper left")
	

	fig.tight_layout()


	# This "makes an animation by repeatedly calling a function func."
	# For func: the first argument will be the next value in frames. Any additional positional arguments can be supplied via the fargs parameter.
	ani = animation.FuncAnimation(fig, func=update,
								fargs=(images, lines, axs[0, 1], axs[1, 1], daisyworld),
								frames = 400,
								interval=updateInterval,
								repeat=False)
 
	# set output file
	if args.movfile:
		# FFwriter = animation.FFMpegWriter(fps=30)
		ani.save(args.movfile, fps=15, extra_args=['-vcodec', 'libx264'])

 
	plt.show()



if __name__ == '__main__':
	main()





