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


models = [model_1, model_2, model_3, model_4];
Nmod = len(models);

Tychonov_regularization_coeff= 1e-4;

gp_restart = 10;
kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) \
+ WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-6, 1.0e-0));



#Nobs_array = [ 2, 2, 2 ];
Nobs_array = [ 3, 6, 9 ];
#Nobs_array = [ 4, 8, 16 ];
Nobs_array = [ 5, 15, 20 ];
#Nobs_array = [ 3 ];

N_columns = 3;
fig_frame = plt.figure(figsize=(14, 8))
outer = gridspec.GridSpec( len(Nobs_array), N_columns, wspace= 0.2, hspace= 0.2 )
fig_frame2 = plt.figure(figsize=(14, 8))
outer2= gridspec.GridSpec( len(Nobs_array), N_columns, wspace= 0.2, hspace= 0.2 )

for nn in range(len(Nobs_array)):
	Nobs = Nobs_array[nn];

	print("Number of observations " + str(Nobs));
	print("Generating synthetic data")

	Nobs_model   = [Nobs for i in range(Nmod)];
	Train_points = [RandomDataGenerator.uniform(x_min, x_max, Nobs_model[i]) for i in range(Nmod)];
	
	observations = [];
	for i in range(Nmod):
		observations.append([]);
		for j in range(Nobs_model[i]):
			observations[i].append(models[i](Train_points[i][j]));

	for i in range(Nmod):
		observations[i] = np.array(observations[i]);



	# Here is arbitrary order

	Mfs = [];

	inner = gridspec.GridSpecFromSubplotSpec(Nmod, 1, subplot_spec= outer[N_columns*nn], wspace=0.1, hspace=0.1)
	
	for Nm in range(Nmod):
		if Nm == 0: 
			Mfs.append(GP(kernel, [basis_function]));
			Mfs[Nm].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff);
		else:
			Mfs.append( GP(kernel, [Mfs[i].predict for i in range(Nm)]) );
			Mfs[Nm].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff);


		yy, vv = Mfs[Nm].predict(xx.reshape(-1, 1), return_variance= True) 
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


	print("Score MF: ", Mfs[-1].score(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	print("Log L MF: ", Mfs[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	print("Qcrit MF: ", Mfs[-1].Qcriteria(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)).sum() )
	print(Mfs[-1].kernel)
	print(Mfs[-1].regression_param)

	yy, vv = Mfs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
	yy = yy.flatten();
	ss = np.sqrt(np.diag(vv))

	# GP_single = GP(kernel);
	# GP_single.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1), Tychonov_regularization_coeff);
	# yy_s, vv_s = GP_single.predict(xx.reshape(-1, 1), return_variance= True) 
	# yy_s = yy_s.flatten();
	# ss_s = np.sqrt(np.diag(vv_s))

	# print("Score SF: ", GP_single.score(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	# print("Log L SF: ", GP_single.compute_loglikelihood(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	# print("Qcrit SF: ", GP_single.Qcriteria(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)).sum() )
	# print(GP_single.kernel)
	# print(GP_single.regression_param)

	gp_ref = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=gp_restart, alpha=Tychonov_regularization_coeff, normalize_y=False);
	gp_ref.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1));
	oy, os = gp_ref.predict(xx.reshape(-1, 1), return_std=True)
	oy = oy.flatten();
	os = os.flatten();


	inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*nn+1], wspace=0.1, hspace=0.1)
	ax = plt.Subplot(fig_frame, inner[0])

	ax.scatter(Train_points[-1], observations[-1][:, 0])

	ax.plot(xx, truth(xx), color='k', label='Truth')

	ax.plot(xx, yy, color='r', label='MF GP')
	ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)
	# ax.plot(xx, yy_s, color='g', label='SF GP')
	# ax.fill_between(xx, yy_s-ss_s, yy_s+ss_s, facecolor='g', alpha=0.3, interpolate=True)
	ax.plot(xx, oy, color='b', label='SKL GP')
	ax.fill_between(xx, oy-os, oy+os, facecolor='b', alpha=0.3, interpolate=True)

	ax.legend(prop={'size': FONTSIZE}, frameon=False)
	ax.legend(frameon=False)
	fig_frame.add_subplot(ax)

	inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*nn+2], wspace=0.1, hspace=0.3)
	ax = plt.Subplot(fig_frame, inner[0])

	#ax.barh(np.arange(len(Mfs[Nm].regression_param)), Mfs[Nm].regression_param.flatten(), 0.2, tick_label=["M " + str(j+1) for j in range(len(Mfs[Nm].regression_param))])

	for i in range(Nmod):
		ax.bar(i+1, np.absolute( Mfs[i].regression_param ).max() +0.1, 0.95, color='gainsboro', edgecolor='k');
		l = len(Mfs[i].regression_param.flatten());
		w = 1.0/(l);
		bar_chart_width= 0.7/(Nmod-1);

		w = 1.0/(Nmod-1);
		ax.bar([(i+0.5)+w/2 +j*w for j in range(l)], np.absolute( Mfs[i].regression_param ).flatten(), bar_chart_width, color=[ 'r' if j else 'k' for j in (Mfs[i].regression_param > 0) ] )
	ax.set_xticklabels(["M " + str(j) for j in range(Nmod+1)]);

	ax.legend(frameon=False)
	fig_frame.add_subplot(ax)


	print()




	# Here is ordered order
	
	Mfs_ordered = [];

	inner = gridspec.GridSpecFromSubplotSpec(Nmod, 1, subplot_spec= outer2[N_columns*nn], wspace=0.1, hspace=0.1)
	
	model_order = np.append(np.argsort( np.absolute(Mfs[-1].regression_param.flatten()) ), Nmod-1).flatten();
	print(model_order)
	for Nm in model_order:
		if not Mfs_ordered: 
			Mfs_ordered.append(GP(kernel, [basis_function]));
			Mfs_ordered[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff);
		else:
			Mfs_ordered.append( GP(kernel, [Mfs_ordered[i].predict for i in range( len(Mfs_ordered) )]) );
			Mfs_ordered[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff);


		yy, vv = Mfs_ordered[-1].predict(xx.reshape(-1, 1), return_variance= True) 
		yy = yy.flatten();
		ss = np.sqrt(np.diag(vv))

		ax = plt.Subplot(fig_frame2, inner[Nm])

		ax.scatter(Train_points[Nm], observations[Nm][:, 0])

		ax.plot(xx, yy, color='r', label='GP')
		ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)


		ax.plot(xx, truth(xx), color='k', label='Truth')

		ax.yaxis.set_major_formatter(plt.NullFormatter())
		ax.set_ylabel('M ' + str(Nm+1))
		fig_frame2.add_subplot(ax)

	ax.set_xlabel('x', fontsize=FONTSIZE)


	print("Score MF: ", Mfs_ordered[-1].score(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	print("Log L MF: ", Mfs_ordered[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	print("Qcrit MF: ", Mfs_ordered[-1].Qcriteria(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)).sum() )
	print(Mfs_ordered[-1].kernel)
	print(Mfs_ordered[-1].regression_param)

	yy, vv = Mfs_ordered[-1].predict(xx.reshape(-1, 1), return_variance= True) 
	yy = yy.flatten();
	ss = np.sqrt(np.diag(vv))

	# GP_single = GP(kernel);
	# GP_single.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1), Tychonov_regularization_coeff);
	# yy_s, vv_s = GP_single.predict(xx.reshape(-1, 1), return_variance= True) 
	# yy_s = yy_s.flatten();
	# ss_s = np.sqrt(np.diag(vv_s))

	# print("Score SF: ", GP_single.score(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	# print("Log L SF: ", GP_single.compute_loglikelihood(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	# print("Qcrit SF: ", GP_single.Qcriteria(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)).sum() )
	# print(GP_single.kernel)
	# print(GP_single.regression_param)

	gp_ref = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=gp_restart, alpha=Tychonov_regularization_coeff, normalize_y=False);
	gp_ref.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1));
	oy, os = gp_ref.predict(xx.reshape(-1, 1), return_std=True)
	oy = oy.flatten();
	os = os.flatten();


	inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer2[N_columns*nn+1], wspace=0.1, hspace=0.1)
	ax = plt.Subplot(fig_frame2, inner[0])

	ax.scatter(Train_points[-1], observations[-1][:, 0])

	ax.plot(xx, truth(xx), color='k', label='Truth')

	ax.plot(xx, yy, color='r', label='MF GP')
	ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)
	# ax.plot(xx, yy_s, color='g', label='SF GP')
	# ax.fill_between(xx, yy_s-ss_s, yy_s+ss_s, facecolor='g', alpha=0.3, interpolate=True)
	ax.plot(xx, oy, color='b', label='SKL GP')
	ax.fill_between(xx, oy-os, oy+os, facecolor='b', alpha=0.3, interpolate=True)

	ax.legend(prop={'size': FONTSIZE}, frameon=False)
	ax.legend(frameon=False)
	fig_frame2.add_subplot(ax)

	inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer2[N_columns*nn+2], wspace=0.1, hspace=0.3)
	ax = plt.Subplot(fig_frame2, inner[0])

	print(["M " + str(j+1) for j in model_order[0:-1] ])
	#ax.barh(model_order[0:-1], Mfs_ordered[-1].regression_param.flatten(), 0.2, tick_label=["M " + str(j+1) for j in model_order[0:-1] ], color='r')

	for i in range(Nmod):
		ax.bar(i+1, np.absolute( Mfs_ordered[i].regression_param ).max() +0.1, 0.95, color='gainsboro', edgecolor='k');
		l = len(Mfs_ordered[i].regression_param.flatten());
		w = 1.0/(l);
		bar_chart_width= 0.7/(Nmod-1);

		w = 1.0/(Nmod-1);
		ax.bar([(i+0.5)+w/2 +j*w for j in range(l)], np.absolute( Mfs_ordered[i].regression_param ).flatten(), bar_chart_width, color=[ 'r' if j else 'k' for j in (Mfs_ordered[i].regression_param > 0) ] )
	ax.set_xticklabels(["M " + str(j) for j in range(Nmod+1)]);

	ax.legend(frameon=False)
	fig_frame2.add_subplot(ax)


	print()


fig_frame.tight_layout()
plt.savefig('FIGURES/cmp.pdf')

fig_frame2.tight_layout()
plt.savefig('FIGURES/cmp2.pdf')

plt.show()
exit()





