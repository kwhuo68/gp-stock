import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

class Kernel():

	__metaclass__ = ABCMeta

	@abstractmethod
	def type(self):
		return 

	@abstractmethod
	def dot_prod(self, x, y):
		return

	#Get kernel matrix 
	def construct_kernel_matrix(self, X, X_prime):
		matrix = np.zeros((len(X), len(X_prime)))
		for i in range(0, len(X)):
			for j in range(0, len(X_prime)):
				# Dot product
				matrix[i, j] = self.dot_prod(X[i], X_prime[j])
		return matrix

class SquareExpKernel(Kernel):
	def __init__(self, sigma, length_scale):
		self.sigma = sigma
		self.length_scale = length_scale

	def type(self):
		return "SquareExpKernel"

	def dot_prod(self, x, y):
		# Using norm for vector magnitude
		return self.sigma**2 * (np.exp(-( np.linalg.norm((x - y))**2) / (2 * self.length_scale**2)))

class OrnsteinKernel(Kernel):
	def __init__(self, sigma, length_scale):
		self.sigma = sigma
		self.length_scale = length_scale

	def type(self):
		return "OrnsteinKernel"

	def dot_prod(self, x, y):
		# Using norm for vector magnitude
		return self.sigma**2 * (np.exp(-( np.linalg.norm((x - y))) / (self.length_scale)))


