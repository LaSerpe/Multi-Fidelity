from matplotlib import pyplot as plt

import numpy as np
import scipy as sp
from scipy.linalg import cholesky, cho_solve, solve_triangular

import math

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

Nobs = 5;

x_min = 0.0;
x_max = 1.0;

Np = 100;
xx = np.linspace(x_min, x_max, Np);

# Define the models
Nmod = 3; #thruth doesn't count

def truth(x):
	return np.sin(x/x_max*pi/2);

def model_1(x):
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
	def __init__(self, Training_points, Training_values, Kernel, noise_level):
		self.Training_points = Training_points;
		self.Training_values = Training_values;
		self.kernel = Kernel;
		self.noise_level = noise_level;

		self.Ktt = self.correlation(self.Training_points, self.Training_points) + self.noise_level * np.eye(len(self.Training_points));
		self.L = cholesky(self.Ktt, lower=True);
		self.alpha = cho_solve((self.L, True), self.Training_values)

	def correlation(self, x, y):
		K = np.zeros((len(x), len(y)));
		for i in range(len(x)):
			for j in range(len(y)):
				K[i, j] = self.kernel.__call__([[ x[i] ]], Y= [[ y[j] ]]);
		return K;

	def loglikelihood(self):
		return 0.5*self.Training_values.T.dot(self.alpha) + np.log(np.diag(self.L)).sum();

	def cost_function(self, theta):
		self.kernel.theta = theta;
		K = self.correlation(self.Training_points, self.Training_points) + self.noise_level * np.eye(len(self.Training_points));
		self.L = cholesky(K, lower=True);
		self.alpha = cho_solve((self.L, True), self.Training_values)
		return self.loglikelihood(),

	def fit(self):
		MIN = 1e16;
		constraints = ({'type': 'ineq', "fun": lambda x: x - 1.0e-2}, {'type': 'ineq', "fun": lambda x: - x + 1.0e1},)

		Np = len(self.kernel.theta);
		for int_it in range(20):
			x0 = np.random.uniform(1.0e-2, 1.0e1, Np).reshape(1, Np); 

			res = sp.optimize.minimize( self.cost_function, x0, method='SLSQP', constraints=constraints, tol=1e-8, options={'maxiter': 150,'ftol': 1e-06, 'disp': False});
			if (self.cost_function(res.x)[0] < MIN):
				MIN = self.cost_function(res.x)
				MIN_x = res.x;

		self.kernel.theta = MIN_x;


	def predict(self, x):
		k = self.correlation(self.Training_points, x);
		mean = k.T.dot(self.alpha);
		variance = cho_solve((self.L, True), k)
		return mean, variance;



gp_restart = 100;
kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) \
+ WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-6, 1.0e-0));

Mfs = [];

J = [];
H = [];
K = [];
B = [];
Br= [];
V = [];


kernel = RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1));
#gp = GP(Train_points[0].reshape(-1,1), observations[0][:, 0].reshape(-1,1), kernel, 1e-2);
gp = GP(Train_points[0], observations[0][:, 0], kernel, 1e-2);
gp.fit()


exit()




def correlation(x, y, kernel):
	K = np.zeros((len(x), len(y)));
	for i in range(len(x)):
		for j in range(len(y)):
			K[i, j] = kernel.__call__([[ x[i] ]], Y= [[ y[j] ]]);
	return K;

def loglikelihood(K, y_observations):
	#y_observations.reshape(-1, 1);
	return  - ( - y_observations.T.dot(np.linalg.inv(K)).dot(y_observations) - np.log(np.linalg.det(K)) )

def cost_function(theta, x, y_observations):
	K = correlation(x, x, RBF(length_scale=theta));
	K += 1.0e-2 * np.eye(len(x));
	L = cholesky(K, lower=True);
	alpha = cho_solve((L, True), y_observations)
	return 0.5*y_observations.T.dot(alpha) + np.log(np.diag(L)).sum()


for Nm in range(Nmod):
	if Nm == 0: 
		H.append(np.zeros((1, Nobs_model[Nm])));

		kernel = RBF(length_scale=1.01, length_scale_bounds=(1.0e-2, 1.0e1)) 
		sub_K = correlation(Train_points[0], Train_points[0], kernel)

		MIN = 1e16;
		constraints = ({'type': 'ineq', "fun": lambda x: x - 1.0e-2}, {'type': 'ineq', "fun": lambda x: - x + 1.0e1},)

		print(observations[Nm][:, 0])
		obs_minus_mean = observations[Nm][:, 0] 
		print(obs_minus_mean)

		for int_it in range(gp_restart):
			x0 = np.random.uniform(1.0e-2, 1.0e1); 

			res = sp.optimize.minimize( cost_function, x0, args=(Train_points[0], obs_minus_mean), method='SLSQP', constraints=constraints, tol=1e-8, options={'maxiter': 150,'ftol': 1e-06, 'disp': False});
			if (cost_function(res.x, Train_points[0], obs_minus_mean ) < MIN):
				MIN = cost_function(res.x, Train_points[0], obs_minus_mean )
				MIN_x = res.x;

		kernel = RBF(length_scale=1.01, length_scale_bounds=(1.0e-2, 1.0e1)) 
		gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=gp_restart, alpha=1e-2, normalize_y=False)
		gp.fit(Train_points[0].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1))

		#sub_K += np.diag(1.0/observations[Nm][:, 1]**2)
		K.append(sub_K);

	# 	J.append();
	# 	B.append();
	# else: 
	# 	J.append();


print(gp.kernel_)
print(np.exp(gp.kernel_.theta[0]))
print(MIN_x[0])
yy_mean_post, yy_std_post = gp.predict(np.array(xx.reshape(-1, 1)), return_std=True);
yy_mean_post = yy_mean_post.flatten();

# yy = correlation(xx, observations[Nm][:, 0]), RBF(length_scale=np.exp(gp.kernel_.theta[0]))).dot(sub_K).dot(observations[Nm][:, 0])

Kss = correlation(xx, xx, RBF(length_scale=np.exp(gp.kernel_.theta[0])));
Kst = correlation(xx, Train_points[0], RBF(length_scale=MIN_x));
Ktt = correlation(Train_points[0], Train_points[0], RBF(length_scale=np.exp(gp.kernel_.theta[0]))) + 1.0e-4 * np.eye(Nobs_model[0]);

#yy = Kst.dot(np.linalg.inv(Ktt)).dot(observations[0][:, 0]-np.mean(observations[0][:, 0])) + np.mean(observations[0][:, 0])
yy = Kst.dot(np.linalg.inv(Ktt)).dot(observations[0][:, 0]) 
yy_cov = Kss - Kst.dot(np.linalg.inv(Ktt)).dot(Kst.T)

# print(yy)
# print(yy_cov_post)
# print(correlation([0.0], [0.0], RBF(length_scale=np.exp(gp.kernel_.theta[0]))))



tt = np.linspace(1.0e-2, 1.0e1, 1000)
pp = [];
for i in tt: pp.append( cost_function(i, Train_points[0], observations[0][:, 0]- np.mean(observations[0][:, 0])) );
plt.figure()
plt.plot(tt, pp, color='r', label='GP')
plt.scatter(MIN_x[0], cost_function(MIN_x[0], Train_points[0], observations[0][:, 0]- np.mean(observations[0][:, 0])) )
plt.xlabel('x', fontsize=FONTSIZE)
plt.ylabel('y [-]', fontsize=FONTSIZE)
plt.tight_layout()





plt.figure()
plt.plot(xx, yy, color='r', label='GP')
plt.plot(xx, yy_mean_post, color='y', label='GP')
plt.fill_between(xx, yy_mean_post-yy_std_post, yy_mean_post+yy_std_post, facecolor='y', alpha=0.3, interpolate=True)
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




