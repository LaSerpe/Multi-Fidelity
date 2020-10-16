from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import scipy as sp
from scipy.linalg import cholesky, cho_solve, solve_triangular

import math
import time
import copy

import sys

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct, ConstantKernel, WhiteKernel


import GPy
from IPython.display import display


from math import pi

import database as database


class GP:
	def __init__(self, Kernel, Basis= None):
		self.kernel = copy.deepcopy(Kernel);
		if Basis is None: 
			self.basis_function = None;
			self.Nbasis= 0;
		else:
			self.basis_function = copy.deepcopy(Basis);
			self.Nbasis= len(Basis);


	def loglikelihood(self):
		return np.array(0.5*(self.Training_values - self.basis.T.dot(self.regression_param) ).T.dot(self.alpha) + np.log(np.diag(self.L)).sum()).flatten();


	def cost_function(self, theta):
		self.kernel.theta = theta[0:len(self.kernel.theta)];
		if (len(self.kernel.theta) != len(theta)):
			self.regression_param = np.array(theta[len(self.kernel.theta)::]).reshape(-1, 1);
		K = self.kernel(self.Training_points);
		K[np.diag_indices_from(K)] += self.noise_level
		self.L = cholesky(K, lower=True);
		self.alpha = cho_solve((self.L, True), (self.Training_values - self.basis.T.dot(self.regression_param) ) )
		return self.loglikelihood();


	def fit(self, Training_points, Training_values, noise_level):

		self.noise_level = copy.deepcopy(noise_level);
		self.Training_points = copy.deepcopy(Training_points);
		self.Training_values = copy.deepcopy(Training_values);

		MIN = 1e16;
		bounds = self.kernel.bounds

		if self.basis_function is None: 
			self.Nbasis = 1;
			self.basis = np.ones((self.Nbasis, len(self.Training_points)))
			self.regression_param = np.zeros((1, 1));
		else:
			self.basis = [];
			for i in range(self.Nbasis): self.basis.append( self.basis_function[i](self.Training_points).flatten() );
			self.basis = np.array(self.basis);
			self.regression_param = np.ones((np.shape(self.basis)[0], 1));
			for i in self.regression_param: bounds = np.append(bounds, [[-10.0, 10.0]], axis=0)


		for int_it in range(10):
			InternalRandomGenerator = np.random.RandomState();
			x0 = InternalRandomGenerator.uniform(bounds[:, 0], bounds[:, 1]);
			res = sp.optimize.minimize(self.cost_function, x0, method="L-BFGS-B", bounds=bounds)
			if (self.cost_function(res.x)[0] < MIN):
				MIN = self.cost_function(res.x)
				MIN_x = res.x;

		if self.basis_function is None:
			self.kernel.theta = MIN_x;
		else:
			self.kernel.theta = MIN_x[0:len(self.kernel.theta)];
			self.regression_param = MIN_x[len(self.kernel.theta)::].reshape(-1, 1);

		Ktt = self.kernel(self.Training_points);
		Ktt[np.diag_indices_from(Ktt)] += self.noise_level;
		self.L = cholesky(Ktt, lower=True);
		self.alpha = cho_solve((self.L, True), (self.Training_values - np.array(self.basis.T.dot(self.regression_param)) ))



	def predict(self, x, return_variance= False):
		if self.basis_function is None: 
			Basis = np.zeros((self.Nbasis, len(x)));
		else:
			Basis = [];
			Basis_v = [];
			for i in range(self.Nbasis): 
				a = self.basis_function[i](x, True)
				#print(np.shape(a[0]))
				Basis.append(   a[0].flatten() );
				Basis_v.append( np.diag(a[1]).flatten() );
			Basis   = np.array(Basis);
			Basis_v = np.array(Basis_v);

		k = self.kernel(x, self.Training_points);
		mean = np.array(Basis.T.dot(np.array(self.regression_param))) + k.dot(self.alpha);
		v = cho_solve((self.L, True), k.T);
		variance = self.kernel(x) - k.dot(v) + np.square(self.regression_param.T).dot(np.square(Basis_v));

		if return_variance is True:
			return mean.T, variance;
		else:
			return mean.T;








plt.rc('font',family='Times New Roman')

RandomDataGenerator = np.random.RandomState();
RandomDataGenerator.seed(1);


col = ['r', 'b', 'm'];
FONTSIZE = 22


x_min = 0.0;
x_max = 1.0;

Np = 1000;
xx = np.linspace(x_min, x_max, Np);

# Define the models
Nmod = 3; #thruth doesn't count

def truth(x):
	return np.sin(x/x_max*pi*8)*x + x;

def model_1(x):
	return [x, 1e-8*np.random.normal(0.0, 1.0)];

def model_2(x):
	return [0.7*(np.sin(x/x_max*pi*8)*x + 0.0), 1e-8*np.random.normal(0.0, 1.0)];

def model_3(x):
	return [np.sin(x/x_max*pi*8)*x + x, 2e-8*np.random.normal(0.0, 1.0)];



models = [model_1, model_2, model_3];





Nobs_array = [ 2, 4, 6 ];
Nobs_array = [ 5, 15, 20 ];

fig_frame = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec( len(Nobs_array), 2, wspace= 0.1, hspace= 0.2 )

for nn in range(len(Nobs_array)):
	Nobs = Nobs_array[nn];

	print("Number of observations " + str(Nobs));
	print("Generating synthetic data")

	Nobs_model   = [Nobs for i in range(Nmod)];
	Train_points = [RandomDataGenerator.uniform(x_min, x_max, Nobs) for i in range(Nmod)];

	#Nobs_model   = [(Nmod - i)*Nobs for i in range(Nmod)];
	#Train_points = [np.random.uniform(x_min, x_max, Nobs_model[i]) for i in range(Nmod)];
	

	observations = [];
	for i in range(Nmod):
		observations.append([]);
		for j in range(Nobs_model[i]):
			observations[i].append(models[i](Train_points[i][j]));

	for i in range(Nmod):
		observations[i] = np.array(observations[i]);


	gp_restart = 10;
	kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) \
	+ WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-6, 1.0e-0));

	Mfs = [];


	def basis_function(x, return_variance= False):
		# basis = np.ones((2, len(x)));
		# for i in range(len(x)):
		# 	basis[1, i] = np.sin(x[i]);
		# return basis
		
		if return_variance is True:
			return np.ones((1, len(x) )), np.zeros((len(x), len(x) ));
		else:
			return np.ones((1, len(x) ));



	inner = gridspec.GridSpecFromSubplotSpec(Nmod, 1, subplot_spec= outer[2*nn], wspace=0.1, hspace=0.1)
	

	for Nm in range(Nmod):

		if Nm == 0: 
			Mfs.append(GP(kernel, [basis_function]));
			Mfs[Nm].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), 1e-2);
		else:
			Mfs.append( GP(kernel, [Mfs[i].predict for i in range(Nm)]) );
			#Mfs.append( GP(kernel, [Mfs[Nm-1].predict for i in range(1)]) );
			Mfs[Nm].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), 1e-2);
			
		print(Mfs[Nm].kernel)
		print(Mfs[Nm].regression_param)

		yy, vv = Mfs[Nm].predict(xx.reshape(-1, 1), return_variance= True  ) 
		yy = yy.flatten();
		ss = np.sqrt(np.diag(vv))

		ax = plt.Subplot(fig_frame, inner[Nm])

		ax.scatter(Train_points[Nm], observations[Nm][:, 0])

		ax.plot(xx, yy, color='r', label='GP')
		ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)


		ax.plot(xx, truth(xx), color='k', label='Truth')

		ax.yaxis.set_major_formatter(plt.NullFormatter())
		ax.set_ylabel('M ' + str(Nm+1))
		fig_frame.add_subplot(ax)

	ax.set_xlabel('x', fontsize=FONTSIZE)
	#ax.tight_layout()
	plt.savefig('FIGURES/mdl_table_' + str(Nobs) + '.pdf')


	gp_ref = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=gp_restart, alpha=1e-2, normalize_y=False);
	gp_ref.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1));
	oy, os = gp_ref.predict(xx.reshape(-1, 1), return_std=True)
	oy = oy.flatten();
	os = os.flatten();

	yy, vv = Mfs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
	yy = yy.flatten();
	ss = np.sqrt(np.diag(vv))


	inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[2*nn+1], wspace=0.1, hspace=0.1)
	ax = plt.Subplot(fig_frame, inner[0])

	ax.scatter(Train_points[-1], observations[-1][:, 0])

	ax.plot(xx, truth(xx), color='k', label='Truth')

	ax.plot(xx, yy, color='r', label='M GP')
	ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)
	ax.plot(xx, oy, color='b', label='GP')
	ax.fill_between(xx, oy-os, oy+os, facecolor='b', alpha=0.3, interpolate=True)

	ax.legend(prop={'size': FONTSIZE}, frameon=False)
	ax.legend(frameon=False)
	fig_frame.add_subplot(ax)

	#ax.xlabel('x', fontsize=FONTSIZE)
	#ax.ylabel('y [-]', fontsize=FONTSIZE)
	#ax.tight_layout()
	plt.savefig('FIGURES/general_cmp_' + str(Nobs) + '.pdf')

	print()


fig_frame.tight_layout()
plt.savefig('FIGURES/cmp.pdf')
plt.show()
exit()





