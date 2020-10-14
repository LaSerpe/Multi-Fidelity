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


# N = 50000;
# a = np.random.uniform(0,  1, N)
# b = np.random.uniform(0, 10, N)

# stda = 1/12;
# stdb = 10/12;

# mean_a = np.zeros((N, 1));
# mean_b = np.zeros((N, 1));
# std_a = np.zeros((N, 1));
# std_b = np.zeros((N, 1));
# for i in range(N):
# 	mean_a[i] = np.sum(a[0:i])/i;
# 	mean_b[i] = np.sum(b[0:i])/i;
# 	std_a[i] = np.sqrt((mean_a[i]-0.5)**2);
# 	std_b[i] = np.sqrt((mean_b[i]-5.0)**2);


# plt.figure()
# plt.plot(mean_a, color='r', label='0.5')
# plt.plot(mean_b, color='b', label='5.0')
# plt.tight_layout()

# plt.figure()
# plt.semilogy(std_a, color='r', label='0.5')
# plt.semilogy(stda/np.sqrt(np.arange(N)+1), ':', color='r')

# plt.semilogy(std_b, color='b', label='5.0')
# plt.semilogy(stdb/np.sqrt(np.arange(N)+1), ':', color='b')
# plt.tight_layout()

# plt.show()
# exit()


class GP:
	def __init__(self, Kernel, Basis= None):
		self.kernel = copy.deepcopy(Kernel);
		if Basis is None: 
			self.basis_function = None;
		else:
			self.basis_function = copy.deepcopy(Basis);


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

		self.noise_level = noise_level;
		self.Training_points = Training_points[:];
		self.Training_values = Training_values[:];

		MIN = 1e16;
		bounds = self.kernel.bounds

		if self.basis_function is None: 
			self.basis = np.ones((1, len(self.Training_points)))
			self.regression_param = np.zeros((1, 1));
		else:
			self.basis = self.basis_function(self.Training_points);
			self.regression_param = np.ones((np.shape(self.basis)[0], 1));
			for i in self.regression_param: bounds = np.append(bounds, [[-10.0, 10.0]], axis=0)
			# print(np.shape(bounds))
			# print(type(bounds))
			# print(bounds)

		for int_it in range(10):
			x0 = np.random.uniform(bounds[:, 0], bounds[:, 1]);
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
		self.alpha = cho_solve((self.L, True), (Training_values - np.array(self.basis.T.dot(self.regression_param)) ))



	def predict(self, x, return_variance= False):
		if self.basis_function is None: 
			Basis = np.zeros((1, len(x)));
		else:
			Basis = self.basis_function(x);

		k = self.kernel(x, self.Training_points);
		mean = np.array(Basis.T.dot(np.array(self.regression_param))) + k.dot(self.alpha);
		v = cho_solve((self.L, True), k.T);
		variance = self.kernel(x) - k.dot(v);
		if return_variance is True:
			return mean.T, variance;
		else:
			return mean.T;








plt.rc('font',family='Times New Roman')


np.random.seed(1)

col = ['r', 'b', 'm'];
FONTSIZE = 22


x_min = 0.0;
x_max = 1.0;

Np = 1000;
xx = np.linspace(x_min, x_max, Np);

# Define the models
Nmod = 3; #thruth doesn't count

# def truth(x):
# 	return np.sin(x/x_max*pi/2);

# def model_1(x):
# 	return [x, 1e-8*np.random.normal(0.0, 1.0)];

# def model_2(x):
# 	return [-x**2 + 2.0*x, 1e-8*np.random.normal(0.0, 1.0)];

# def model_3(x):
# 	return [np.sin(x/x_max*pi/2) + 0.02, 1e-8*np.random.normal(0.0, 1.0)];


def truth(x):
	return np.sin(x/x_max*pi*8)*x + x;

def model_1(x):
	return [x, 1e-8*np.random.normal(0.0, 1.0)];

def model_2(x):
	return [0.7*(np.sin(x/x_max*pi*8)*x + 0.0), 1e-8*np.random.normal(0.0, 1.0)];

def model_3(x):
	return [np.sin(x/x_max*pi*8)*x + x, 2e-8*np.random.normal(0.0, 1.0)];



models = [model_1, model_2, model_3];





# Nobs_array = [ 5 , 15, 20 ];

# fig_frame = plt.figure(figsize=(10, 8))
# outer = gridspec.GridSpec( len(Nobs_array), 2, wspace= 0.1, hspace= 0.2 )

# for nn in range(len(Nobs_array)):
# 	Nobs = Nobs_array[nn];

# 	print("Number of observations " + str(Nobs));

# 	# observations
# 	fx = np.random.uniform(x_min, x_max, Nobs) # this can change for datasets of different size

# 	Train_points = [np.random.uniform(x_min, x_max, Nobs) for i in range(Nmod)];
# 	Nobs_model   = [Nobs for i in range(Nmod)];

# 	observations = [];
# 	for i in range(Nmod):
# 		observations.append([]);
# 		for j in range(Nobs_model[i]):
# 			observations[i].append(models[i](Train_points[i][j]));


# 	Train_points = np.array(Train_points);
# 	observations = np.array(observations);

# 	print(np.shape(Train_points))
# 	print(np.shape(observations))


	# plt.figure()
	# plt.plot(xx, truth(xx), color='k', label='Truth')
	# for i in range(Nmod):
	# 	plt.plot(xx, models[i](xx)[0], color=col[i], label='Model ' + str(i+1))
	# 	plt.errorbar(Train_points[i], observations[i][:, 0], yerr=observations[i][:, 1], color=col[i], fmt='.r')
	# plt.legend(prop={'size': FONTSIZE-10}, frameon=False, loc='lower right')
	# plt.xlabel('x', fontsize=FONTSIZE)
	# plt.ylabel('y [-]', fontsize=FONTSIZE)
	# plt.tight_layout()



# Here is for the dataase
x_min = 0.0;
x_max = 24.0;

Np = 1000;
xx = np.linspace(x_min, x_max, Np);
fig_frame = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec( 1, 2, wspace= 0.1, hspace= 0.2 )

data_numbers = [0, 2, 4, 5, 6];
for nn in range(1):
	
	observations = [];
	Train_points = [];
	Truth = [];
	for i in data_numbers:
		Train_points.append(database.access_data(i)[:, 0]);
		observations.append([]);
		for j in range( len(Train_points[-1]) ):
			observations[-1].append( [database.access_data(i)[j][1], database.access_data(i)[j][2] ] );

	Nmod = len(data_numbers)		
	Nobs = len(Train_points[0][:]);
	Nobs_model   = [Nobs for i in range(Nmod)];
	Train_points = np.array(Train_points);
	observations = np.array(observations);

	print(np.shape(Train_points))
	print(np.shape(observations))



	gp_restart = 10;
	kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) \
	+ WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-6, 1.0e-0));

	Mfs = [];


	def basis_function(x):
		# basis = np.ones((2, len(x)));
		# for i in range(len(x)):
		# 	basis[1, i] = np.sin(x[i]);
		# return basis
		return np.ones((1, len(x) ));

	inner = gridspec.GridSpecFromSubplotSpec(Nmod, 1, subplot_spec= outer[2*nn], wspace=0.1, hspace=0.1)
	

	for Nm in range(Nmod):
		if Nm == 0: 
			Mfs.append(GP(kernel, basis_function));
			Mfs[Nm].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), 1e-2);

		# 	J.append();
		# 	B.append();
		# else: 
		# 	J.append();

		else:
			Mfs.append( GP(kernel, Mfs[Nm-1].predict) );
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


		#ax.plot(xx, truth(xx), color='k', label='Truth')

		ax.yaxis.set_major_formatter(plt.NullFormatter())
		ax.set_ylabel('M ' + str(Nm+1))
		fig_frame.add_subplot(ax)

	#ax.set_xlabel('x', fontsize=FONTSIZE)
	#ax.tight_layout()
	#plt.savefig('FIGURES/mdl_table_' + str(Nobs) + '.pdf')


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

	#ax.plot(xx, truth(xx), color='k', label='Truth')

	ax.plot(xx, yy, color='r', label='M GP')
	ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)
	ax.plot(xx, oy, color='b', label='GP')
	ax.fill_between(xx, oy-os, oy+os, facecolor='b', alpha=0.3, interpolate=True)

	#ax.legend(prop={'size': FONTSIZE}, frameon=False)
	ax.legend(frameon=False)
	fig_frame.add_subplot(ax)

	#ax.xlabel('x', fontsize=FONTSIZE)
	#ax.ylabel('y [-]', fontsize=FONTSIZE)
	#ax.tight_layout()
	#plt.savefig('FIGURES/general_cmp_' + str(Nobs) + '.pdf')



fig_frame.tight_layout()
plt.savefig('FIGURES/cmp.pdf')
plt.show()
exit()













































































# fy_1 = [model_1(i) + j for (i, j) in zip(fx, st_1)];
# fy_2 = [model_2(i) + j for (i, j) in zip(fx, st_2)];
# fy_3 = [model_3(i) + j for (i, j) in zip(fx, st_3)];

# Fx = [fx, fx, fx];
# Fy = [np.array(fy_1), np.array(fy_2), np.array(fy_3)];
# St = [st_1, st_2, st_3];

# fx = np.concatenate((fx, fx, fx), axis=None)
# fy = np.concatenate((fy_1, fy_2, fy_3), axis=None)
# st = np.concatenate((st_1, st_2, st_3), axis=None)






gp_restart = 100;
kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) \
+ WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-6, 1.0e-0));
gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=gp_restart, alpha=st, normalize_y=True)
yy_mean_prior, yy_std_prior = gp.predict(np.array(fx.reshape(-1, 1)), return_std=True)
gp.fit(fx.reshape(-1, 1), fy.reshape(-1, 1))
yy_mean_post, yy_std_post = gp.predict(np.array(xx.reshape(-1, 1)), return_std=True)
yy_mean_post = yy_mean_post.flatten();


plt.figure()
plt.plot(xx, truth(xx), color='k', label='Truth')
plt.plot(xx, model_1(xx), color='r', label='Model 1')
plt.plot(xx, model_2(xx), color='b', label='Model 2')
plt.plot(xx, model_3(xx), color='m', label='Model 3')
plt.plot(xx, yy_mean_post, color='y', label='GP')
plt.fill_between(xx, yy_mean_post-yy_std_post, yy_mean_post+yy_std_post, facecolor='y', alpha=0.3, interpolate=True)
plt.axis([x_min, x_max, 0, 1.1])
plt.legend(prop={'size': FONTSIZE}, frameon=False, loc='lower right')
plt.xlabel('x', fontsize=FONTSIZE)
plt.ylabel('y [-]', fontsize=FONTSIZE)
plt.tight_layout()



# kernel = RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) 
# print(kernel.__call__([[1]], Y= [[10]] ))



kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) + GPy.kern.White(1)
m = GPy.models.GPRegression(fx.reshape(-1, 1), fy.reshape(-1, 1),kernel)
m.optimize_restarts(num_restarts = 20)
display(m)


m.plot(plot_density=True)
plt.plot(xx, truth(xx), color='k', label='Truth')
plt.plot(xx, model_1(xx), color='r', label='Model 1')
plt.plot(xx, model_2(xx), color='b', label='Model 2')
plt.plot(xx, model_3(xx), color='m', label='Model 3')
plt.axis([x_min, x_max, 0, 1.1])
plt.legend(prop={'size': FONTSIZE-10}, frameon=False, loc='lower right')
plt.xlabel('x', fontsize=FONTSIZE)
plt.ylabel('y [-]', fontsize=FONTSIZE)
plt.tight_layout()


# kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) + GPy.kern.White(1)
s_kernel = [GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) + GPy.kern.White(1) for i in range(Nmod)];

s_m = [GPy.models.GPRegression(i.reshape(-1, 1), j.reshape(-1, 1), k, noise_var=1e-2) for i, j, k in zip(Fx, Fy, s_kernel)];

for i in s_m: 
	i.optimize_restarts(num_restarts = 20);
	display(i);


for i in range(Nmod): 
	s_m[i].plot(plot_density=True)
	plt.plot(xx, truth(xx), color='k', label='Truth')
	plt.plot(xx, model_1(xx), color='r', label='Model 1')
	plt.plot(xx, model_2(xx), color='b', label='Model 2')
	plt.plot(xx, model_3(xx), color='m', label='Model 3')
	plt.axis([x_min, x_max, 0, 1.1])
	plt.legend(prop={'size': FONTSIZE-10}, frameon=False, loc='lower right')
	plt.xlabel('x', fontsize=FONTSIZE)
	plt.ylabel('y [-]', fontsize=FONTSIZE)
	plt.tight_layout()


plt.show()
exit()




