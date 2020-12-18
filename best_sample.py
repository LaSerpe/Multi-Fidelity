from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import scipy as sp
from scipy.linalg import cholesky, cho_solve, solve_triangular

import math
import time
import copy

import sys
import os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct, ConstantKernel, WhiteKernel



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

Rnd_seed = 49;
RandomDataGenerator = np.random.RandomState();
RandomDataGenerator.seed( Rnd_seed );


col = ['r', 'b', 'm'];
FONTSIZE = 22

Mode='G'
Mode_Opt = 'MLLW';
LASSO= False;

Nested= False;
Matching = False;
Equal_size= False;
Deterministic= True;

x_min = 0.0;
x_max = 1.0;

Np = 1000;
xx = np.linspace(x_min, x_max, Np);


# Complex Function
models = [model_1, model_2, model_3, model_4];
models = [model_1, model_2, model_3, model_6, model_4];
#models = [model_1, model_2, model_3, model_4, model_4];
#models = [model_3, model_6, model_4];
#models = [model_4, model_4, model_4, model_4, model_4];
truth = model_4


# Harmonics
# models = [model_9, model_10, model_11, model_12]
# #models = [model_9, model_10, model_12]
# #models = [model_9, model_10, model_11, model_9, model_10, model_11, model_12]
# truth = model_12

# Sacher
# models = [model_Sacher_1, model_Sacher_2]
# truth = model_Sacher_2

# models = [U_1, U_2, U_3, U_4s, U_4s, U_4];
# truth = U_4;

models = [U_5, U_3, U_1, U_6, U_2, U_4, U_7];
truth = U_7;





Nmod = len(models);

Tychonov_regularization_coeff= 1e-4;

gp_restart = 10;
kernel = ConstantKernel(1.0**2, (1.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-1, 1.0e1)) \
+ WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-8, 1.0e-0));



Nobs_array = [ 9 ];
Nobs_array = [ 3 ];
Ns = 60;


nOrdering = 1;
N_columns = 3;
fig_frame = [];
for iOrdering in range( Ns ):
	fig_frame.append(plt.figure(figsize=(14, 8)));
outer = gridspec.GridSpec( nOrdering, N_columns, wspace= 0.2, hspace= 0.2 );


for nn in range(len(Nobs_array)):
	Nobs = Nobs_array[nn];
	model_order = [];
	model_order.append(np.arange(Nmod).flatten());

	print("Number of observations " + str(Nobs));
	print("Generating synthetic data")

	if Equal_size:
		Nobs_model   = [Nobs for i in range(Nmod)];
	else:
		if not Deterministic:
			Nobs_model = [(Nmod - i)*Nobs for i in range(Nmod)];
		else:
			Nobs_model   = [2*Nobs for i in range(Nmod)];
			Nobs_model[-1] = int(Nobs_model[-1]/2);

	if Matching:
		if not Equal_size: print("Matching must have equal sized data sets!"); exit();
		Train_points = [];

		if Deterministic:
			for i in range(Nmod):
				Train_points.append( np.linspace(x_min, x_max, Nobs_model[i]) );
		else:
			for i in range(Nmod):
				RandomDataGenerator.seed( Rnd_seed );
				Train_points.append( RandomDataGenerator.uniform(x_min, x_max, Nobs_model[i]) );

	elif Nested and nn != 0:
		for i in range(Nmod):
			if Deterministic:
				Train_points[i] = np.concatenate( (Train_points[i], np.linspace(x_min, x_max, Nobs_model[i]-len(Train_points[i])) ), axis=None)
			else:
				Train_points[i] = np.concatenate( (Train_points[i], RandomDataGenerator.uniform(x_min, x_max, Nobs_model[i]-len(Train_points[i])) ), axis=None)

	else: 
		Train_points = [];
		for i in range(Nmod):
			if Deterministic:
				Train_points.append( np.linspace(x_min, x_max, Nobs_model[i]) );
			else:
				Train_points.append( RandomDataGenerator.uniform(x_min, x_max, Nobs_model[i]) );

	Train_points[-1] = np.insert(Train_points[-1], [0], [0.0] );
	Nobs_model[-1] += 1;

	image_counter = 0;
	regression_param_history = [];
	for x_new in np.linspace(x_min, x_max, Ns):
		print(x_new);

		Train_points[-1][0] = x_new;
		#print(Train_points[-1])

		observations = [];
		for i in range(Nmod):
			observations.append([]);
			for j in range(Nobs_model[i]):
				observations[i].append(models[i](Train_points[i][j]));

		for i in range(Nmod):
			observations[i] = np.array(observations[i]);


		it_frame = fig_frame[image_counter];
		it_frame.suptitle('N points ' + str(Nobs) );

		for iOrdering in range(nOrdering):	

			
			inner = gridspec.GridSpecFromSubplotSpec(Nmod, 1, subplot_spec= outer[N_columns*iOrdering], wspace=0.1, hspace=0.1)
			
			print(model_order[iOrdering])
			Mfs = [];
			for Nm in model_order[iOrdering]:
				if not Mfs: 
					#Mfs.append(GP(kernel, [basis_function]));
					Mfs.append(GP(kernel, mode=Mode));
					Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);
				else:
					#Mfs.append( GP(kernel, [Mfs[-1].predict]) );
					Mfs.append( GP(kernel, [Mfs[i].predict for i in range( len(Mfs) )], mode=Mode) );
					Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);


				yy, vv = Mfs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
				yy = yy.flatten();
				ss = np.sqrt(np.diag(vv))


				ax = plt.Subplot(it_frame, inner[ np.where(np.array(model_order[iOrdering]) == Nm )[0][0] ])

				ax.scatter(Train_points[Nm], observations[Nm][:, 0])

				ax.plot(xx, yy, color='r', label='GP')
				ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)


				#ax.plot(xx, truth(xx), color='k', label='Truth')
				ax.plot(xx, models[Nm](xx)[:][0], color='k', label='Truth')

				ax.yaxis.set_major_formatter(plt.NullFormatter())
				ax.set_ylabel('M ' + str(Nm+1))
				it_frame.add_subplot(ax)

				print('Level ' + str(Nm))
				print("Score MF: ", Mfs[-1].score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)))
				print("Log L MF: ", Mfs[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)))
				print("Qcrit MF: ", Mfs[-1].Qcriteria(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)).sum() )
				print('GPM')
				print(Mfs[-1].kernel)
				print(Mfs[-1].regression_param.flatten())
				print()
				print()

			regression_param_history.append(Mfs[-1].regression_param.flatten())

			print()
			print()

			print("Score MF: ", Mfs[-1].score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)))
			print("Log L MF: ", Mfs[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)))
			print("Qcrit MF: ", Mfs[-1].Qcriteria(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)).sum() )
			print('GPM')
			print(Mfs[-1].kernel)
			print(Mfs[-1].regression_param.flatten())


			yy, vv = Mfs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
			yy = yy.flatten();
			ss = np.sqrt(np.diag(vv))


			inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*iOrdering+1], wspace=0.1, hspace=0.1)
			ax = plt.Subplot(it_frame, inner[0])

			ax.scatter(Train_points[-1], observations[-1][:, 0])
			ax.scatter(Train_points[-1][0], observations[-1][0, 0], color='r')

			ax.plot(xx, truth(xx)[0], color='k', label='Truth')

			ax.plot(xx, yy, color='r', label='MF GP')
			ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)

			ax.legend(prop={'size': 6}, frameon=False)
			#ax.legend(frameon=False)
			it_frame.add_subplot(ax)



			inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*iOrdering+2], wspace=0.1, hspace=0.3)
			ax = plt.Subplot(it_frame, inner[0])

			print(["M " + str(j+1) for j in model_order[iOrdering] ])
			for i in range( len(Mfs) ):
				if (len(Mfs[i].regression_param) == 0): continue;
				ax.bar(i, np.absolute( Mfs[i].regression_param ).max() +0.1, 0.95, color='gainsboro', edgecolor='k');
				l = len(Mfs[i].regression_param.flatten());
				w = 1.0/(l);
				bar_chart_width= 0.7/(Nmod-1);

				w = 1.0/(Nmod-1);
				#ax.bar([(i-0.5)+w/2 +j*w for j in range(l)], np.absolute( Mfs[i].regression_param ).flatten(), bar_chart_width, color=[ 'r' if j else 'k' for j in (Mfs[i].regression_param > 0) ] )
				ax.bar([(i-0.5)+w/2 +j*w for j in model_order[iOrdering][0:l]], np.absolute( Mfs[i].regression_param ).flatten(), bar_chart_width, color=[ 'r' if j else 'k' for j in (Mfs[i].regression_param > 0) ] )
			# ax.set_xticks([j for j in range(Nmod)]);
			# ax.set_xticklabels(["M " + str(j+1) for j in range(Nmod)]);

			ax.set_xticks([j for j in range( len(model_order[iOrdering]) )])
			ax.set_xticklabels(["M " + str(j+1) for j in model_order[iOrdering]]);


			ax.legend(frameon=False)
			it_frame.add_subplot(ax)

		image_counter += 1;







		

	


for nn in range(Ns):
	fig_frame[nn].tight_layout()
	fig_frame[nn].savefig( 'FIGURES/movie/img' + str(nn) + '.png', dpi=500)
os.system("ffmpeg -r 1 -i ./FIGURES/movie/img%01d.png -vcodec mpeg4 -y ./FIGURES/movie/movie.mp4")




plt.figure()
for i in range( Nmod - 1 ):
	plt.plot(np.linspace(x_min, x_max, Ns), np.array(regression_param_history).T[i], label='Model ' + str(i+1));
plt.legend()
plt.tight_layout()
plt.savefig( 'FIGURES/correlations.pdf')

#plt.show()
exit()








