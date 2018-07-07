import numpy as np
import kernels 
import matplotlib.pyplot as plt


class GaussianProcess():
	def __init__(self, kernel):
		self.x_obs = None
		self.y_obs = None
		self.x_pred = None
		self.kernel = kernel

	#Obtain conditional mean and variance
	def derive_conditional(self, x_obs, y_obs, x_pred):
		self.x_obs = x_obs
		self.y_obs = y_obs
		self.x_pred = x_pred

		#Set up kernel matrices 
		K_x_obs_obs = self.kernel.construct_kernel_matrix(self.x_obs, self.x_obs)
		K_x_obs_pred = self.kernel.construct_kernel_matrix(self.x_obs, self.x_pred)
		K_x_pred_obs = self.kernel.construct_kernel_matrix(self.x_pred, self.x_obs)
		K_x_pred_pred = self.kernel.construct_kernel_matrix(self.x_pred, self.x_pred)

		# print("K(X, X): ")
		# print(K_x_obs_obs)
		# print("K(X, X'): ")
		# print(K_x_obs_pred)
		# print("K(X', X): ")
		# print(K_x_pred_obs)
		# print("K(X', X')")
		# print(K_x_pred_pred)

		# Equations for conditional mean and variance
		K_x_obs_obs_inv = np.linalg.inv(K_x_obs_obs)
		conditional_mean = np.dot(np.dot(K_x_pred_obs, K_x_obs_obs_inv), y_obs)
		condition_var = K_x_pred_pred - np.dot(np.dot(K_x_pred_obs, K_x_obs_obs_inv), K_x_obs_pred)
		return conditional_mean, condition_var



