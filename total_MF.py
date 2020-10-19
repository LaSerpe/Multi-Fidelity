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
from GP_module import GP
from models_module import *



def basis_function(x, return_variance= False):
	# basis = np.ones((1, len(x)));
	# for i in range(len(x)):
	# 	basis[0][i] = x[i] + x[i]*np.sin(x[i]);
	# if return_variance is True:
	# 	return basis, np.zeros(( len(x), len(x) ));
	# else:
	# 	return basis;
	
	if return_variance is True:
		return np.ones((1, len(x) )), np.zeros((len(x), len(x) ));
	else:
		return np.ones((1, len(x) ));

plt.rc('font',family='Times New Roman')

RandomDataGenerator = np.random.RandomState();
RandomDataGenerator.seed(1);


col = ['r', 'b', 'm'];
FONTSIZE = 22


x_min = 0.0;
x_max = 1.0;

Np = 1000;
xx = np.linspace(x_min, x_max, Np);


#models = [model_1, model_2, model_5, model_4];
models = [model_4, model_4, model_5, model_4];
Nmod = len(models);


gp_restart = 10;
kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) \
+ WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-6, 1.0e-0)) 

# kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) \
# + ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-03, 10.0))**2;

# kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) \
# * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-03, 10.0))**2;

#kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * DotProduct(sigma_0=5.0, sigma_0_bounds=(1e-03, 10.0))**2;

#kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) 

# Nobs = 3;
# Nobs_model   = [(Nmod - i)*Nobs for i in range(Nmod)];
Nobs = 10;
Nobs_model   = [Nobs for i in range(Nmod)];

print("Number of observations " + str(Nobs));
print("Generating synthetic data")

Train_points = [RandomDataGenerator.uniform(x_min, x_max, Nobs_model[i]) for i in range(Nmod)];

observations = [];
for i in range(Nmod):
	observations.append([]);
	for j in range(Nobs_model[i]):
		observations[i].append(models[i](Train_points[i][j]));

for i in range(Nmod):
	observations[i] = np.array(observations[i]);

Mfs = [];

for Nm in range(Nmod):	
	Mfs.append( GP(kernel));
	Mfs[Nm].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), 1e-2);
	print(Mfs[Nm].kernel)
	print(Mfs[Nm].regression_param)



#### This combines all GPs to predict all data
print()
print("Total")
Mfs_total = GP(kernel, [Mfs[i].predict for i in range(Nmod)]);

cc = []
for i in range(Nmod):
	yy, vv = Mfs[i].predict(Train_points[i].reshape(-1, 1), return_variance= True) 
	yy = yy.flatten();
	cc.append(np.sqrt(np.diag(vv)))

Total_Training_Points = np.hstack([i for i in Train_points]).reshape(-1, 1)
Total_Observations = np.hstack([i[:, 0] for i in observations]).reshape(-1, 1)
Total_Noise = np.hstack([i for i in cc]).reshape(-1, 1)
#Mfs_total.fit(Total_Training_Points, Total_Observations, Total_Noise);
Mfs_total.fit(Total_Training_Points, Total_Observations, 1e-2);

print(Mfs_total.kernel)
print(Mfs_total.regression_param)


#gp_ref = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=gp_restart, alpha=Total_Noise.flatten(), normalize_y=False);
gp_ref = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=gp_restart, alpha=1e-2, normalize_y=False);
gp_ref.fit(Total_Training_Points, Total_Observations);
print(gp_ref.kernel_)
oy, os = gp_ref.predict(xx.reshape(-1, 1), return_std=True)
oy = oy.flatten();
os = os.flatten();


fig_frame = plt.figure(figsize=(14, 8))
outer = gridspec.GridSpec( 1, 2, wspace= 0.2, hspace= 0.2 )


inner = gridspec.GridSpecFromSubplotSpec(Nmod, 1, subplot_spec= outer[0], wspace=0.1, hspace=0.2)

cmap = plt.get_cmap("tab10")
for i in range(Nmod):
	axs = plt.Subplot(fig_frame, inner[i])
	yy, vv = Mfs[i].predict(xx.reshape(-1, 1), return_variance= True) 
	yy = yy.flatten();
	ss = np.sqrt(np.diag(vv))
	axs.plot(xx, yy, color=cmap(i), label='M'+str(i+1))
	axs.fill_between(xx, yy-ss, yy+ss, facecolor=cmap(i), alpha=0.3, interpolate=True)
	axs.scatter(Train_points[i], observations[i][:, 0]);
	axs.legend(prop={'size': FONTSIZE-10}, frameon=False)
	fig_frame.add_subplot(axs)
fig_frame.tight_layout()




inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec= outer[1], wspace=0.1, hspace=0.1)
axs = plt.Subplot(fig_frame, inner[0])

yy, vv = Mfs_total.predict(xx.reshape(-1, 1), return_variance= True) 
yy = yy.flatten();
ss = np.sqrt(np.diag(vv))


axs.plot(xx, truth(xx), color='k', label='Truth')
axs.plot(xx, yy, color='r', label='M GP')
axs.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)
axs.plot(xx, oy, color='b', label='GP')
axs.fill_between(xx, oy-os, oy+os, facecolor='b', alpha=0.3, interpolate=True)
axs.scatter(Total_Training_Points, Total_Observations);
axs.legend(prop={'size': FONTSIZE-10}, frameon=False)
fig_frame.add_subplot(axs)

axs = plt.Subplot(fig_frame, inner[1])
axs.barh(np.arange(len(Mfs_total.regression_param)), Mfs_total.regression_param.flatten(), 0.2, tick_label=["M " + str(j+1) for j in range(len(Mfs_total.regression_param))])
fig_frame.add_subplot(axs)







plt.show()
exit()





