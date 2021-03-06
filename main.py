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

N_columns = 3;
fig_frame = plt.figure(figsize=(14, 8))
outer = gridspec.GridSpec( len(Nobs_array), N_columns, wspace= 0.2, hspace= 0.2 )

for nn in range(len(Nobs_array)):
	Nobs = Nobs_array[nn];

	print("Number of observations " + str(Nobs));
	print("Generating synthetic data")

	Nobs_model   = [(Nmod - i)*Nobs for i in range(Nmod)];
	Train_points = [RandomDataGenerator.uniform(x_min, x_max, Nobs_model[i]) for i in range(Nmod)];
	
	observations = [];
	for i in range(Nmod):
		observations.append([]);
		for j in range(Nobs_model[i]):
			observations[i].append(models[i](Train_points[i][j]));

	for i in range(Nmod):
		observations[i] = np.array(observations[i]);

	Mfs = [];

	inner = gridspec.GridSpecFromSubplotSpec(Nmod, 1, subplot_spec= outer[N_columns*nn], wspace=0.1, hspace=0.1)
	

	for Nm in range(Nmod):

		#if Nm == 2: Mfs.append( Mfs[Nm-1] ); continue;

		if Nm == 0: 
			Mfs.append(GP(kernel, [basis_function]));
			Mfs[Nm].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff);
		else:
			# if Nm == 3:
			# 	Mfs.append( GP(kernel, [Mfs[i].predict for i in range(Nm-1)]) );
			# else:
			# 	Mfs.append( GP(kernel, [Mfs[i].predict for i in range(Nm)]) );
			Mfs.append( GP(kernel, [Mfs[i].predict for i in range(Nm)]) );
			#Mfs.append( GP(kernel, [Mfs[Nm-1].predict for i in range(1)]) );
			Mfs[Nm].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff);


		#if Nm == 2: continue;
		# print(Mfs[Nm].kernel)
		# print(Mfs[Nm].regression_param)
		# print("Score MF: ", Mfs[Nm].score(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
		# print("Log L MF: ", Mfs[Nm].compute_loglikelihood(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))

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
	#ax.tight_layout()
	#plt.savefig('FIGURES/mdl_table_' + str(Nobs) + '.pdf')

	print("Score MF: ", Mfs[-1].score(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	print("Log L MF: ", Mfs[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	print("Qcrit MF: ", Mfs[-1].Qcriteria(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)).sum() )
	print(Mfs[-1].kernel)
	print(Mfs[-1].regression_param)

	yy, vv = Mfs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
	yy = yy.flatten();
	ss = np.sqrt(np.diag(vv))

	GP_single = GP(kernel);
	GP_single.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1), Tychonov_regularization_coeff);

	yy_s, vv_s = GP_single.predict(xx.reshape(-1, 1), return_variance= True) 
	yy_s = yy_s.flatten();
	ss_s = np.sqrt(np.diag(vv_s))

	print("Score SF: ", GP_single.score(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	print("Log L SF: ", GP_single.compute_loglikelihood(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
	print("Qcrit SF: ", GP_single.Qcriteria(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)).sum() )
	print(GP_single.kernel)
	print(GP_single.regression_param)

	gp_ref = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=gp_restart, alpha=Tychonov_regularization_coeff, normalize_y=False);
	gp_ref.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1));
	oy, os = gp_ref.predict(xx.reshape(-1, 1), return_std=True)
	oy = oy.flatten();
	os = os.flatten();


	inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*nn+1], wspace=0.1, hspace=0.1)
	ax = plt.Subplot(fig_frame, inner[0])

	ax.scatter(Train_points[-1], observations[-1][:, 0])

	ax.plot(xx, truth(xx), color='k', label='Truth')

	ax.plot(xx, yy, color='r', label='M GP')
	ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)
	ax.plot(xx, yy_s, color='g', label='M GP')
	ax.fill_between(xx, yy_s-ss_s, yy_s+ss_s, facecolor='g', alpha=0.3, interpolate=True)
	ax.plot(xx, oy, color='b', label='GP')
	ax.fill_between(xx, oy-os, oy+os, facecolor='b', alpha=0.3, interpolate=True)

	ax.legend(prop={'size': FONTSIZE}, frameon=False)
	ax.legend(frameon=False)
	fig_frame.add_subplot(ax)

	#ax.xlabel('x', fontsize=FONTSIZE)
	#ax.ylabel('y [-]', fontsize=FONTSIZE)
	#ax.tight_layout()
	#plt.savefig('FIGURES/general_cmp_' + str(Nobs) + '.pdf')


	inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*nn+2], wspace=0.1, hspace=0.3)
	ax = plt.Subplot(fig_frame, inner[0])

	cell_text = []
	for i in range(Nmod):
		ax.barh(np.arange(len(Mfs[Nm].regression_param)), Mfs[Nm].regression_param.flatten(), 0.2, tick_label=["M " + str(j+1) for j in range(len(Mfs[Nm].regression_param))])
		cell_text.append(['{:.2e}'.format(j) for j in np.exp(Mfs[Nm].kernel.theta)])

	#the_table = ax.table(cellText=cell_text, rowLabels=["std", "$\mathcal{l}$", "$\sigma_n$"], colLabels=["M " + str(j+1) for j in range(Nmod)], loc='bottom')
		
	ax.legend(frameon=False)
	fig_frame.add_subplot(ax)


	print()


fig_frame.tight_layout()
plt.savefig('FIGURES/cmp.pdf')





print("Score MF: ", Mfs[-1].score(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
print("Log L MF: ", Mfs[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
print("Qcrit MbF: ", Mfs[-1].Qcriteria(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)).sum() )

Mfs_Nno_Basis = GP(kernel);
Mfs_Nno_Basis.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1), Tychonov_regularization_coeff);
print("Score SF: ", Mfs_Nno_Basis.score(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
print("Log L SF: ", Mfs_Nno_Basis.compute_loglikelihood(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)))
print("Qcrit SF: ", Mfs_Nno_Basis.Qcriteria(xx.reshape(-1, 1), truth(xx).reshape(-1, 1)).sum() )

plt.figure()
plt.plot(xx, truth(xx), color='k', label='Truth')

yy, vv = Mfs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
yy = yy.flatten();
ss = np.sqrt(np.diag(vv))
plt.plot(xx, yy, color='r', label='M GP')
plt.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)
a = Mfs[-1].Qcriteria(xx.reshape(-1, 1), truth(xx).reshape(-1, 1))
plt.plot(xx, a/a.max(), ':', color='r', label='M GP Q')

yy, vv = Mfs_Nno_Basis.predict(xx.reshape(-1, 1), return_variance= True) 
yy = yy.flatten();
ss = np.sqrt(np.diag(vv))
plt.plot(xx, yy, color='b', label='STD GP')
plt.fill_between(xx, yy-ss, yy+ss, facecolor='b', alpha=0.3, interpolate=True)
b = Mfs_Nno_Basis.Qcriteria(xx.reshape(-1, 1), truth(xx).reshape(-1, 1))
plt.plot(xx, b/a.max(), ':', color='b', label='STD GP Q')

plt.legend(prop={'size': FONTSIZE}, frameon=False)
plt.legend(frameon=False)

plt.show()
exit()





