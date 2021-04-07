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
FONTSIZE = 14

Mode_Opt_labels= ['ML',  'LOO', 'DL', 'WA', 'AWLL', 'LS'];
Mode_Opt_list  = ['MLL', 'LOO', 'MLLD', 'MLLW', 'MLLA', 'MLLS'];
LASSO_list     = [False, False, True, False, False, False];
Mode='G'#Don't touch this for the paper

Nobs_array = [ 6, 9, 12, 15 ];
Nobs_array = [ 15 ];
NdataRandomization= 100;
Nested= False;
Matching = False;
Equal_size= False;
Deterministic= False;

Activate_histogram_plot=False

x_min = 0.0;
x_max = 1.0;

Np = 1000;
xx = np.linspace(x_min, x_max, Np);

# Complex Function
models = [model_1, model_2, model_6, model_3, model_4];
truth = model_4
Nmod = len(models);

Tychonov_regularization_coeff= 1e-4;

gp_restart = 10;
kernel = ConstantKernel(1.0**2, (1.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-1, 1.0e1)) \
+ WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-8, 1.0e-0));



Mode_Opt = Mode_Opt_list[ 0 ];
LASSO= LASSO_list[0];



nOrdering = 1;


for iDataRandomization in range(NdataRandomization):

	Train_points = [];

	for nn in range(len(Nobs_array)):
	
		Nobs = Nobs_array[nn];
		model_order = [];
		model_order.append(np.arange(Nmod).flatten());
		
		print("Number of observations " + str(Nobs) + " iRand " + str(iDataRandomization));
		print("Generating synthetic data")

		if Equal_size:
			Nobs_model   = [Nobs for i in range(Nmod)];
		else:
			if not Deterministic:
				Nobs_model = [(Nmod - i)*Nobs for i in range(Nmod)];
				#Nobs_model   = [10*Nobs for i in range(Nmod)];
				#Nobs_model[-1] = Nobs;
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


		for iOrdering in range(nOrdering):	

		
			print(model_order[iOrdering])

			Mfs = [];
			print('IR regression param');
			for Nm in model_order[iOrdering]:
				if not Mfs: 
					Mfs.append(GP(kernel, mode=Mode));
					Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);
				else:
					Mfs.append( GP(kernel, [Mfs[i].predict for i in range( len(Mfs) )], mode=Mode) );
					Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);

			# MGs = [];
			# print('LeGratiet regression param');
			# for Nm in model_order[iOrdering]:
			# 	if not MGs: 
			# 		MGs.append(GP(kernel, mode=Mode));
			# 		MGs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);
			# 	else:
			# 		MGs.append( GP(kernel, [MGs[-1].predict], mode=Mode) );
			# 		MGs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);

			# GP_single = GP(kernel, mode=Mode);
			# GP_single.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);





exit()








