from matplotlib import pyplot as plt

import numpy as np
import scipy as sp
from scipy.linalg import cholesky, cho_solve, solve_triangular

import math
import time

import sys

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct, ConstantKernel, WhiteKernel


import GPy
from IPython.display import display


from math import pi

plt.rc('font',family='Times New Roman')


np.random.seed(0)

col = ['r', 'b', 'm'];
FONTSIZE = 22

Nobs = 50;

x_min = 0.0;
x_max = 1.0;

Np = 1000;
xx = np.linspace(x_min, x_max, Np);

# Define the models
Nmod = 3; #thruth doesn't count

def truth(x):
	return np.sin(x/x_max*pi/2);

def model_1(x):
	#return [np.sin(x/x_max*pi*8)*x, 1e-2*np.random.normal(0.0, 1.0)];
	return [x, 1e-2*np.random.normal(0.0, 1.0)];

def model_2(x):
	return [-x**2 + 2.0*x, 1e-2*np.random.normal(0.0, 1.0)];

def model_3(x):
	return [np.sin(x/x_max*pi/2) + 0.02, 1e-2*np.random.normal(0.0, 1.0)];

models = [model_1, model_2, model_3];


# observations
fx = np.random.uniform(x_min, x_max, Nobs) # this can change for datasets of different size

Train_points = [fx for i in range(Nmod)];
Nobs_model   = [Nobs for i in range(Nmod)];



observations = [];
for i in range(Nmod):
	observations.append([]);
	for j in range(Nobs_model[i]):
		observations[i].append(models[i](Train_points[i][j]));


Train_points = np.array(Train_points);
observations = np.array(observations);


# plt.figure()
# plt.plot(xx, truth(xx), color='k', label='Truth')
# for i in range(Nmod):
# 	plt.plot(xx, models[i](xx)[0], color=col[i], label='Model ' + str(i+1))
# 	plt.errorbar(Train_points[i], observations[i][:, 0], yerr=observations[i][:, 1], color=col[i], fmt='.r')
# plt.legend(prop={'size': FONTSIZE-10}, frameon=False, loc='lower right')
# plt.xlabel('x', fontsize=FONTSIZE)
# plt.ylabel('y [-]', fontsize=FONTSIZE)
# plt.tight_layout()




class GP:
	def __init__(self, Kernel):
		self.kernel = Kernel;


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


	def fit(self, Training_points, Training_values, noise_level, Basis=None):

		self.noise_level = noise_level;
		self.Training_points = Training_points;
		self.Training_values = Training_values;

		MIN = 1e16;
		bounds = self.kernel.bounds

		if Basis is None: 
			self.basis = np.ones((1, len(self.Training_points)))
			self.regression_param = np.zeros((1, 1));
		else:
			self.basis = Basis;
			self.regression_param = np.ones((np.shape(Basis)[0], 1));
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

		if Basis is None:
			self.kernel.theta = MIN_x;
		else:
			self.kernel.theta = MIN_x[0:len(self.kernel.theta)];
			self.regression_param = MIN_x[len(self.kernel.theta)::].reshape(-1, 1);

		Ktt = self.kernel(self.Training_points);
		Ktt[np.diag_indices_from(Ktt)] += self.noise_level;
		self.L = cholesky(Ktt, lower=True);
		self.alpha = cho_solve((self.L, True), (Training_values - np.array(self.basis.T.dot(self.regression_param)) ))




	def predict(self, x, Basis= None):
		if Basis is None: 
			Basis = np.zeros((1, len(x)))

		k = self.kernel(x, self.Training_points);

		print(np.shape(k.dot(self.alpha)))
		print(np.shape(Basis.T.dot(np.array(self.regression_param))))

		mean = np.array(Basis.T.dot(np.array(self.regression_param))) + k.dot(self.alpha);
		#mean = k.dot(self.alpha);
		v = cho_solve((self.L, True), k.T);
		variance = self.kernel(x) - k.dot(v);
		return mean, variance;



gp_restart = 10;
kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) \
+ WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-6, 1.0e-0));

Mfs = [];

J = [];
H = [];
K = [];
B = [];
Br= [];
V = [];

def basis_function(x):
	# basis = np.ones((2, len(x)));
	# for i in range(len(x)):
	# 	basis[1, i] = np.sin(x[i]);
	# return basis
	return np.zeros((1, len(x) ));


for Nm in range(Nmod):
	if Nm == 0: 
		H.append(basis_function(Train_points[Nm]));
		
		start = time.time();
		Mfs.append(GP(kernel));
		Mfs[0].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), 1e-2, H[Nm]);
		print(time.time()-start)

		start = time.time();
		gp_ref = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=gp_restart, alpha=1e-2, normalize_y=False);
		gp_ref.fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1));
		print(time.time()-start)

	# 	J.append();
	# 	B.append();
	# else: 
	# 	J.append();


print(gp_ref.kernel_)
print(Mfs[0].kernel)
print(Mfs[0].regression_param)
print(np.mean(observations[0][:, 0]))
yy, vv = Mfs[0].predict(xx.reshape(-1, 1), basis_function(xx)  ) 
yy = yy.flatten();
ss = np.sqrt(np.diag(vv))

oy, os = gp_ref.predict(xx.reshape(-1, 1), return_std=True)
oy = oy.flatten();
os = os.flatten();


plt.figure()
plt.scatter(Train_points[0], observations[0][:, 0])

plt.plot(xx, yy, color='r', label='GP')
plt.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)


plt.plot(xx, oy, color='b', label='GP_old')
plt.fill_between(xx, oy-os, oy+os, facecolor='b', alpha=0.3, interpolate=True)

plt.xlabel('x', fontsize=FONTSIZE)
plt.ylabel('y [-]', fontsize=FONTSIZE)
plt.tight_layout()

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




