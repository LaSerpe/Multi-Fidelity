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

import subprocess
command = ['rm PRELIMINARY_PAPER_B/*']
subprocess.call(command, shell=True)

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

Mode_Opt_labels= ['ML',  'LOO', 'DL', 'WA']#, 'AWLL', 'LS'];
Mode_Opt_list  = ['MLL', 'LOO', 'MLLD', 'MLLW']#, 'MLLA', 'MLLS'];
LASSO_list     = [False, False, True, False]#, False, False];
Mode='G'#Don't touch this for the paper

Nobs_array = [ 6, 9, 12, 15 ];
Nobs_array = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ];
#Nobs_array = [ 6  ];
NdataRandomization= 1;
Nested= True;
Matching = False;
Equal_size= False;
Deterministic= False;

Activate_histogram_plot=True

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




nOrdering = len(Mode_Opt_list);
N_columns = 2;



class PerformanceRecord:
	regression_param 	= [];
	kernel_param 		= [];
	LOGL 				= [];
	score 				= [];


MF_performance = [];

for iDataRandomization in range(NdataRandomization):

	Train_points = [];

	for iOrdering in range(nOrdering):
		MF_performance.append([PerformanceRecord() for i in range(len(Nobs_array))]);


	fig_frame = [];
	for iOrdering in range( len(Nobs_array) ):
		fig_frame.append(plt.figure(figsize=(14, 8)));
	outer = gridspec.GridSpec( nOrdering, N_columns, wspace= 0.2, hspace= 0.3 );

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


		it_frame = fig_frame[nn];
		it_frame.suptitle('N points ' + str(Nobs) );

		for iOrdering in range(nOrdering):	

			Mode_Opt = iOrdering;
	
			print(model_order[iOrdering])
			Mfs = [];
			print('IR regression param');
			for Nm in model_order[iOrdering]:
				if not Mfs: 
					Mfs.append(GP(kernel, mode=Mode));
					Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt_list[Mode_Opt], LASSO=LASSO_list[Mode_Opt]);
				else:
					Mfs.append( GP(kernel, [Mfs[i].predict for i in range( len(Mfs) )], mode=Mode) );
					Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt_list[Mode_Opt], LASSO=LASSO_list[Mode_Opt]);
				print(len(Train_points[Nm]), Mfs[-1].regression_param)


			MF_performance[-(nOrdering-iOrdering)][nn].regression_param = np.copy( Mfs[-1].regression_param.flatten() );
			MF_performance[-(nOrdering-iOrdering)][nn].kernel_param		= np.copy( np.exp(Mfs[-1].kernel.theta) );
			MF_performance[-(nOrdering-iOrdering)][nn].LOGL 			= np.copy( Mfs[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )
			MF_performance[-(nOrdering-iOrdering)][nn].score 			= np.copy( Mfs[-1].score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )


			yy, vv = Mfs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
			yy = yy.flatten();
			ss = np.sqrt(np.diag(vv))



			inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*iOrdering], wspace=0.1, hspace=0.1)
			ax = plt.Subplot(it_frame, inner[0])
			ax.scatter(Train_points[-1], observations[-1][:, 0])
			ax.plot(xx, truth(xx)[0], color='k', label='T')
			ax.plot(xx, yy, color='r', label=Mode_Opt_labels[Mode_Opt])
			ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)
			ax.legend(prop={'size': FONTSIZE}, frameon=False)
			ax.tick_params(axis='both', labelsize=FONTSIZE)
			it_frame.add_subplot(ax)



			inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*iOrdering+1], wspace=0.1, hspace=0.3)
			ax = plt.Subplot(it_frame, inner[0])




			# print(["M " + str(j+1) for j in model_order[iOrdering] ])
			# for i in range( len(Mfs) ):
			# 	if (len(Mfs[i].regression_param) == 0): continue;
			# 	ax.bar(i, np.absolute( Mfs[i].regression_param ).max() +0.1, 0.95, color='gainsboro', edgecolor='k');
			# 	l = len(Mfs[i].regression_param.flatten());
			# 	w = 1.0/(l);
			# 	bar_chart_width= 0.7/(Nmod-1);

			# 	w = 1.0/(Nmod-1);
			# 	ax.bar([(i-0.5)+w/2 +j*w for j in model_order[iOrdering][0:l]], np.absolute( Mfs[i].regression_param ).flatten(), bar_chart_width, color=[ 'r' if j else 'k' for j in (Mfs[i].regression_param > 0) ] )


			# # if   Mode_Opt_list[Mode_Opt] == 'MLLW': ax.set_ylim([0, 1]);
			# # else Mode_Opt_list[Mode_Opt] == 'MLLW': ax.set_ylim([0,10]);

			# ax.tick_params(axis='both', labelsize=FONTSIZE)
			# ax.set_xticks([j for j in range( len(model_order[iOrdering]) )])
			# ax.set_xticklabels(["M " + str(j+1) for j in model_order[iOrdering]]);
			# ax.legend(prop={'size': FONTSIZE}, frameon=False)
			# it_frame.add_subplot(ax)

			#########################################################################################################################################################################################

			print(["M " + str(j+1) for j in model_order[iOrdering] ])
			l = len(Mfs[-1].regression_param.flatten());
			w = 1.0/(l);
			bar_chart_width= 0.7;

			w = 1.0;
			ax.bar([(1-0.5)+w/2 +j*w for j in model_order[iOrdering][0:l]], np.absolute( Mfs[-1].regression_param ).flatten(), bar_chart_width, color=[ 'r' if j else 'k' for j in (Mfs[-1].regression_param > 0) ] )

			ax.tick_params(axis='both', labelsize=FONTSIZE)
			ax.set_xticks([j+1 for j in range( len(model_order[iOrdering][0:l]) )])
			ax.set_xticklabels(["M " + str(j+1) for j in model_order[iOrdering][0:l] ]);
			ax.legend(prop={'size': FONTSIZE}, frameon=False)
			it_frame.add_subplot(ax)

			ax.set_ylim([0,2]);

			model_order.append(np.arange(Nmod).flatten());



	string_save = 'PRELIMINARY_PAPER_B/' + str(iDataRandomization) + '_' + Mode + '_' + Mode_Opt_list[Mode_Opt] + '_';
	if LASSO_list[Mode_Opt]:      string_save+= 'LASSO_';
	if Matching:      string_save+= 'matching_';
	if Nested:        string_save+= 'nested_';
	if Equal_size:    string_save+= 'equal_';

	for nn in range(len(Nobs_array)):
		fig_frame[nn].tight_layout()
		fig_frame[nn].savefig( string_save + str(nn) + '.pdf')


	
av_score_IR = np.zeros((nOrdering, len(Nobs_array)));
st_score_IR = np.zeros((nOrdering, len(Nobs_array)));


for j in range(len(Nobs_array)):
	for iOrdering in range(nOrdering):
		f_IR = open('PRELIMINARY_PAPER_B/regression_params_IR_' + Mode + '_' + Mode_Opt_list[iOrdering] + '_n'+ str(Nobs_array[j]) + '.dat', 'w')

		reg_IR = []; 
		sco_IR = []; 
		kp_IR  = []; 

		for i in MF_performance[iOrdering::nOrdering]: reg_IR.append(i[j].regression_param); sco_IR.append(i[j].score); kp_IR.append(i[j].kernel_param);

		f_IR.write('Mean: ' + str( np.array(reg_IR).mean(axis=0) ) + '\n');
		f_IR.write('Std:  ' + str( np.array(reg_IR).std(axis=0) ) + '\n');
		f_IR.write('Mean score:  ' + str( np.array(sco_IR).mean(axis=0) ) + '    Std score:  ' +  str( np.array(sco_IR).std(axis=0) ) + '\n');
		f_IR.write('Mean ker p:  ' + str( np.array(kp_IR ).mean(axis=0) ) + '    Std reg p:  ' +  str( np.array(kp_IR ).std(axis=0) ) + '\n');
		av_score_IR[iOrdering, j] = np.array(sco_IR ).mean(axis=0); st_score_IR[iOrdering, j] = np.array(sco_IR ).std(axis=0);


		f_IR.write('N reg param, score, kern params\n');



		for i in MF_performance[iOrdering::nOrdering]:
			reg_IR.append(i[j].regression_param);
			for k in i[j].regression_param:
				f_IR.write( str(k) + '    ');
			f_IR.write( str(i[j].score) + '    ');
			for k in i[j].kernel_param:
				f_IR.write( str(k) + '    ');
			f_IR.write('\n')


f_IR.close()




string_save = 'PRELIMINARY_PAPER_B/' + Mode + '_';
if Matching:      string_save+= 'matching_';
if Nested:        string_save+= 'nested_';
if Equal_size:    string_save+= 'equal_';

plt.figure()
for iOrdering in range(nOrdering):
	plt.plot(av_score_IR[iOrdering, :], label=Mode_Opt_labels[iOrdering])

plt.xlabel(r'$N^l$', fontsize=FONTSIZE);
#plt.ylabel(r'$score_{AVG}$', fontsize=FONTSIZE);
plt.ylabel(r'$score$', fontsize=FONTSIZE);
plt.xticks(np.arange(0, len(Nobs_array)), [str(j) for j in Nobs_array], fontsize=FONTSIZE);
plt.yticks(fontsize=FONTSIZE);

plt.legend(prop={'size': FONTSIZE}, frameon=False)
plt.tight_layout()
plt.savefig( string_save + 'nPoints_behavior_AVG.pdf');

plt.figure()
for iOrdering in range(nOrdering):
	plt.plot(st_score_IR[iOrdering, :], label=Mode_Opt_labels[iOrdering])

plt.xlabel(r'$N^l$', fontsize=FONTSIZE);
#plt.ylabel(r'$score_{STD}$', fontsize=FONTSIZE);
plt.ylabel(r'$score$', fontsize=FONTSIZE);
plt.xticks(np.arange(0, len(Nobs_array)), [str(j) for j in Nobs_array], fontsize=FONTSIZE);
plt.yticks(fontsize=FONTSIZE);

plt.legend(prop={'size': FONTSIZE}, frameon=False)
plt.tight_layout()
plt.savefig( string_save + 'nPoints_behavior_STD.pdf');

exit()








