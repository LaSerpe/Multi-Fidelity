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



from math import pi

import database as database
#from GP_module import GP
#from GP_module_M import GP
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
Nested= False;
Matching = False;
Equal_size= True;
Deterministic= False;

x_min = 0.0;
x_max = 1.0;

Np = 1000;
xx = np.linspace(x_min, x_max, Np);


# Complex Function
models = [model_1, model_2, model_3, model_4];
models = [model_1, model_2, model_3, model_6, model_4];
#models = [model_1, model_2, model_3, model_4, model_4];
#models = [model_3, model_6, model_4];
truth = model_4


# Harmonics
# models = [model_9, model_10, model_11, model_12]
# #models = [model_9, model_10, model_11, model_9, model_10, model_11, model_12]
# truth = model_12

# Sacher
# models = [model_Sacher_1, model_Sacher_2]
# truth = model_Sacher_2

models = [U_1, U_2, U_3, U_4s, U_4s, U_4];
truth = U_4;

# models = [U_5, U_3, U_1, U_6, U_2, U_4, U_7];
# truth = U_7;

plt.figure()
for i in range( len(models) ):
	plt.plot(xx, models[i](xx)[0], label=str(i+1));
plt.legend()
# plt.show();
# 






Nmod = len(models);

Tychonov_regularization_coeff= 1e-4;

gp_restart = 10;
kernel = ConstantKernel(1.0**2, (1.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-1, 1.0e1)) \
+ WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-8, 1.0e-0));




Nobs_array = [ 3, 5, 10, 15, 20 ];
Nobs_array = [ 5, 9, 17 ];
#Nobs_array = [ 3, 6, 9 ];



nOrdering = 4;

N_columns = 4;
fig_frame = [];
for iOrdering in range( len(Nobs_array) ):
	fig_frame.append(plt.figure(figsize=(14, 8)));
outer = gridspec.GridSpec( nOrdering, N_columns, wspace= 0.2, hspace= 0.2 );
Train_points = [];


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
			Nobs_model = [ (Nobs - 1) *2**(Nmod - i - 1) + 1   for i in range(Nmod)];

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
	

	observations = [];
	for i in range(Nmod):
		observations.append([]);
		for j in range(Nobs_model[i]):
			observations[i].append(models[i](Train_points[i][j]));

	for i in range(Nmod):
		observations[i] = np.array(observations[i]);



	Mfs_store = [];
	it_frame = fig_frame[nn];
	it_frame.suptitle('N points ' + str(Nobs) );

	for iOrdering in range(nOrdering):	

		
		inner = gridspec.GridSpecFromSubplotSpec(Nmod, 1, subplot_spec= outer[N_columns*iOrdering], wspace=0.1, hspace=0.1)
		
		print(model_order[iOrdering])
		Mfs = [];
		for Nm in model_order[iOrdering]:
			if not Mfs: 
				#Mfs.append(GP(kernel, [basis_function]));
				Mfs.append(GP(kernel, mode=Mode));
				Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt);
			else:
				#Mfs.append( GP(kernel, [Mfs[-1].predict]) );
				Mfs.append( GP(kernel, [Mfs[i].predict for i in range( len(Mfs) )], mode=Mode) );
				Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt);


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

		GP_single = GP(kernel, mode=Mode);
		GP_single.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt);
		yy_s, vv_s = GP_single.predict(xx.reshape(-1, 1), return_variance= True) 
		yy_s = yy_s.flatten();
		ss_s = np.sqrt(np.diag(vv_s))

		print("Score SF: ", GP_single.score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)))
		print("Log L SF: ", GP_single.compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)))
		print("Qcrit SF: ", GP_single.Qcriteria(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)).sum() )
		print('GPS')
		print(GP_single.kernel)

		gp_ref = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=gp_restart, alpha=Tychonov_regularization_coeff, normalize_y=False);
		gp_ref.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1));
		print('SKCP')
		print(gp_ref.kernel_)
		oy, os = gp_ref.predict(xx.reshape(-1, 1), return_std=True)
		oy = oy.flatten();
		os = os.flatten();


		inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*iOrdering+1], wspace=0.1, hspace=0.1)
		ax = plt.Subplot(it_frame, inner[0])

		ax.scatter(Train_points[-1], observations[-1][:, 0])

		ax.plot(xx, truth(xx)[0], color='k', label='Truth')

		ax.plot(xx, yy, color='r', label='MF GP')
		ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)
		ax.plot(xx, yy_s, color='g', label='SF GP')
		ax.fill_between(xx, yy_s-ss_s, yy_s+ss_s, facecolor='g', alpha=0.3, interpolate=True)
		ax.plot(xx, oy, color='b', label='SKL GP')
		ax.fill_between(xx, oy-os, oy+os, facecolor='b', alpha=0.3, interpolate=True)

		ax.legend(prop={'size': 6}, frameon=False)
		#ax.legend(frameon=False)
		it_frame.add_subplot(ax)


		inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*iOrdering+3], wspace=0.1, hspace=0.1)
		tmp = [ 0.0 for i in model_order[iOrdering][0:-1] ];
		#a = np.argsort( model_order[iOrdering] );
		for i in range(len(tmp)): tmp[ np.argsort( model_order[iOrdering] )[i] ] = round( Mfs[-1].regression_param.flatten()[i] , 3)
		ax = plt.Subplot(it_frame, inner[0])

		# txt = "Score MF: " + str( Mfs[-1].score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1))) + '\n' + \
		# "L2 er MF: " + str( Mfs[-1].L2normCreteria(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)).sum() ) + '\n' + \
		# "Log L MF: " + str( Mfs[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1))[0] ) + '\n\n' + \
		# str( Mfs[-1].kernel) + '\n' + \
		# str( tmp ) 

		txt = "Score MF: " + str( round( Mfs[-1].score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)), 3) ) + "  SF: " + \
		str( round( GP_single.score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)), 3) ) + '\n' + \
		"L2err MF: " + str( round( Mfs[-1].L2normCreteria(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)).sum(), 3) ) + "  SF: " + \
		str( round( GP_single.L2normCreteria(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)).sum(), 3) ) + '\n' + \
		"Log L MF: " + str( round( Mfs[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1))[0], 3 ) ) + "  SF: " + \
		str( round( GP_single.compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1))[0], 3 ) ) + '\n\n' + \
		str( Mfs[-1].kernel ) + '\n' + \
		str( tmp ) 

		ax.text(0, 0, txt, fontsize=10, ha='left', wrap=True)
		ax.set_axis_off()
		ax.set_frame_on(False)
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

		Mfs_store.append(Mfs);

		if(iOrdering == 0): model_order.append( np.append(np.argsort( np.absolute(Mfs_store[0][-1].regression_param.flatten()) )      , Nmod-1).flatten() );
		if(iOrdering == 1): model_order.append( np.append(np.argsort( np.absolute(Mfs_store[0][-1].regression_param.flatten()) )[::-1], Nmod-1).flatten() );
		if(iOrdering == 2): 
			sub_model_ordering = [];
			tmp = -1;
			while True:
				if( len(Mfs_store[0][tmp].regression_param.flatten()) == 0): break;
				tmp = np.argsort( np.absolute(Mfs_store[0][tmp].regression_param.flatten()) )[-1];
				sub_model_ordering.append( tmp );
				if (tmp == 0):
					break;

			sub_model_ordering.reverse();
			sub_model_ordering.append( Nmod-1 );
			sub_model_ordering = np.array(sub_model_ordering).copy();
			model_order.append( sub_model_ordering.flatten() );


	print()
	print()
	print()
	print()
	print()


string_save = 'FIGURES/' + Mode + '_' + Mode_Opt + '_';
if Matching:      string_save+= 'matching_';
if Nested:        string_save+= 'nested_';
if Equal_size:    string_save+= 'equal_';

for nn in range(len(Nobs_array)):
	fig_frame[nn].tight_layout()
	fig_frame[nn].savefig( string_save + str(nn) + '.pdf')

plt.show()
exit()








