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
command = ['rm PRELIMINARY_PAPER_MCMC/*']
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
plt.rc('figure', max_open_warning = 0)

Rnd_seed = 49;
RandomDataGenerator = np.random.RandomState();
RandomDataGenerator.seed( Rnd_seed );


col = ['r', 'b', 'm'];
FONTSIZE = 14

Mode_Opt_labels= ['ML',  'LOO', 'DL', 'WA', 'AWLL', 'LS', 'MCMC'];
Mode_Opt_list  = ['MLL', 'LOO', 'MLLD', 'MLLW', 'MLLA', 'MLLS', 'MLL_MC'];
LASSO_list     = [False, False, True, False, False, False];
Mode='G'#Don't touch this for the paper

#Nobs_array = [ 6, 9, 12, 15 ];
Nobs_array = [ 15 ]#, 9, 12, 15 ];
NdataRandomization= 1;#100;
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




# for iMode in range(len(Mode_Opt_list)):

# 	Mode_Opt = Mode_Opt_list[ iMode ];
# 	LASSO= LASSO_list[iMode];

Mode_Opt = Mode_Opt_list[ -1 ];
LASSO= LASSO_list[ -1 ];



nOrdering = 1;
N_columns = 2;
if Activate_histogram_plot: N_columns += 1;



class PerformanceRecord:
	regression_param 	= [];
	kernel_param 		= [];
	LOGL 				= [];
	score 				= [];


MF_performance = [];
SF_performance = [];
LG_performance = [];

for iDataRandomization in range(NdataRandomization):

	Train_points = [];

	for iOrdering in range(nOrdering):
		MF_performance.append([PerformanceRecord() for i in range(len(Nobs_array))]);
		SF_performance.append([PerformanceRecord() for i in range(len(Nobs_array))]);
		LG_performance.append([PerformanceRecord() for i in range(len(Nobs_array))]);


	fig_frame = [];
	for iOrdering in range( len(Nobs_array) ):
		fig_frame.append(plt.figure(figsize=(14, 8)));
	outer = gridspec.GridSpec( nOrdering, N_columns, wspace= 0.2, hspace= 0.3 );

	for nn in range(len(Nobs_array)):

		mc_fig        = [];
		mc_fig_mixing = [];
		for qq in range( Nmod ):
			mc_fig.append(plt.figure(figsize=(14, 8)));
			mc_fig_mixing.append(plt.figure(figsize=(14, 8)));
	
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

			inner = gridspec.GridSpecFromSubplotSpec(Nmod, 1, subplot_spec= outer[N_columns*iOrdering], wspace=0.1, hspace=0.1)
			
			print(model_order[iOrdering])
			Mfs = [];
			print('IR regression param');
			for Nm in model_order[iOrdering]:
				if not Mfs: 
					Mfs.append(GP(kernel, mode=Mode));
					mc_fig[Nm], mc_fig_mixing[Nm] = Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);
				else:
					Mfs.append( GP(kernel, [Mfs[i].predict for i in range( len(Mfs) )], mode=Mode) );
					mc_fig[Nm], mc_fig_mixing[Nm] =  Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);
				print('N train points, ', len(Train_points[Nm]), ' Reg param, ', Mfs[-1].regression_param)
				print()

			# MGs = [];
			# print('LeGratiet regression param');
			# for Nm in model_order[iOrdering]:
			# 	if not MGs: 
			# 		MGs.append(GP(kernel, mode=Mode));
			# 		MGs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);
			# 	else:
			# 		MGs.append( GP(kernel, [MGs[-1].predict], mode=Mode) );
			# 		MGs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);
			# 	print(len(Train_points[Nm]), MGs[-1].regression_param)



			for Nm in model_order[iOrdering]:

				ax = plt.Subplot(it_frame, inner[ np.where(np.array(model_order[iOrdering]) == Nm )[0][0] ])

				# yy, vv = Mfs[ np.where(np.array(model_order[iOrdering]) == Nm )[0][0] ].predict(xx.reshape(-1, 1), return_variance= True) 
				# yy = yy.flatten();
				# ss = np.sqrt(np.diag(vv))				
				# ax.plot(xx, yy, color='r', label='GP')
				# ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)

				ax.scatter(Train_points[Nm], observations[Nm][:, 0])
				ax.plot(xx, models[Nm](xx)[:][0], color='k', label='T')

				if Nm == Nmod-1: 
					ax.tick_params(axis='both', labelsize=FONTSIZE)
					ax.yaxis.set_major_formatter(plt.NullFormatter())
					ax.set_ylabel('T', fontsize=FONTSIZE)
				else:
					ax.yaxis.set_major_formatter(plt.NullFormatter())
					ax.set_ylabel('M ' + str(Nm+1), fontsize=FONTSIZE)
				it_frame.add_subplot(ax)


			MF_performance[-(nOrdering-iOrdering)][nn].regression_param = np.copy( Mfs[-1].regression_param.flatten() );
			MF_performance[-(nOrdering-iOrdering)][nn].kernel_param		= np.copy( np.exp(Mfs[-1].kernel.theta) );
			MF_performance[-(nOrdering-iOrdering)][nn].LOGL 			= np.copy( Mfs[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )
			MF_performance[-(nOrdering-iOrdering)][nn].score 			= np.copy( Mfs[-1].score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )


			# LG_performance[-(nOrdering-iOrdering)][nn].regression_param = np.copy( MGs[-1].regression_param.flatten() );
			# LG_performance[-(nOrdering-iOrdering)][nn].kernel_param		= np.copy( np.copy( np.exp(MGs[-1].kernel.theta) ) );
			# LG_performance[-(nOrdering-iOrdering)][nn].LOGL 			= np.copy( MGs[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )
			# LG_performance[-(nOrdering-iOrdering)][nn].score 			= np.copy( MGs[-1].score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )


			yy, vv = Mfs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
			yy = yy.flatten();
			ss = np.sqrt(np.diag(vv))

			# yg, vg = MGs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
			# yg = yg.flatten();
			# sg = np.sqrt(np.diag(vg))

			# GP_single = GP(kernel, mode=Mode);
			# GP_single.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);

			# SF_performance[-(nOrdering-iOrdering)][nn].regression_param = np.copy( GP_single.regression_param.flatten() );
			# SF_performance[-(nOrdering-iOrdering)][nn].kernel_param		= np.copy( np.exp(GP_single.kernel.theta) );
			# SF_performance[-(nOrdering-iOrdering)][nn].LOGL 			= np.copy( GP_single.compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )
			# SF_performance[-(nOrdering-iOrdering)][nn].score 			= np.copy( GP_single.score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )

			# yy_s, vv_s = GP_single.predict(xx.reshape(-1, 1), return_variance= True) 
			# yy_s = yy_s.flatten();
			# ss_s = np.sqrt(np.diag(vv_s))

			inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*iOrdering+1], wspace=0.1, hspace=0.1)
			ax = plt.Subplot(it_frame, inner[0])
			ax.scatter(Train_points[-1], observations[-1][:, 0])
			ax.plot(xx, truth(xx)[0], color='k', label='T')
			ax.plot(xx, yy, color='r', label='IR')
			ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)
			# ax.plot(xx, yy_s, color='g', label='SF')
			# ax.fill_between(xx, yy_s-ss_s, yy_s+ss_s, facecolor='g', alpha=0.3, interpolate=True)
			# ax.plot(xx, yg, color='b', label='SR')
			# ax.fill_between(xx, yg-sg, yg+sg, facecolor='b', alpha=0.3, interpolate=True)
			ax.legend(prop={'size': FONTSIZE}, frameon=False)
			ax.tick_params(axis='both', labelsize=FONTSIZE)
			it_frame.add_subplot(ax)


			if Activate_histogram_plot:
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
					ax.bar([(i-0.5)+w/2 +j*w for j in model_order[iOrdering][0:l]], np.absolute( Mfs[i].regression_param ).flatten(), bar_chart_width, color=[ 'r' if j else 'k' for j in (Mfs[i].regression_param > 0) ] )


				ax.tick_params(axis='both', labelsize=FONTSIZE)
				ax.set_xticks([j for j in range( len(model_order[iOrdering]) )])
				ax.set_xticklabels(["M " + str(j+1) for j in model_order[iOrdering]]);
				ax.legend(prop={'size': FONTSIZE}, frameon=False)
				it_frame.add_subplot(ax)


			# model_order.append(np.arange(Nmod).flatten());
			# model_order[-1][2] = 3;
			# model_order[-1][3] = 2;



	string_save = 'PRELIMINARY_PAPER_MCMC/' + str(iDataRandomization) + '_' + Mode + '_' + Mode_Opt + '_';
	if LASSO:      string_save+= 'LASSO_';
	if Matching:      string_save+= 'matching_';
	if Nested:        string_save+= 'nested_';
	if Equal_size:    string_save+= 'equal_';

	for nn in range(len(Nobs_array)):
		fig_frame[nn].tight_layout()
		fig_frame[nn].savefig( string_save + str(nn) + '.pdf')


	for nn in range( Nmod ):
		mc_fig[nn][0].tight_layout()
		mc_fig[nn][0].savefig( string_save + 'burnMC_' + str(nn) + '.pdf')
		mc_fig_mixing[nn][0].tight_layout()
		mc_fig_mixing[nn][0].savefig( string_save + 'burnMC_mix_' + str(nn) + '.pdf')

	for nn in range( Nmod ):
		mc_fig[nn][1].tight_layout()
		mc_fig[nn][1].savefig( string_save + 'MC_' + str(nn) + '.pdf')
		mc_fig_mixing[nn][1].tight_layout()
		mc_fig_mixing[nn][1].savefig( string_save + 'MC_mix_' + str(nn) + '.pdf')


av_score_IR = np.zeros((nOrdering, len(Nobs_array)));
st_score_IR = np.zeros((nOrdering, len(Nobs_array)));

# av_score_SR = np.zeros((nOrdering, len(Nobs_array)));
# st_score_SR = np.zeros((nOrdering, len(Nobs_array)));

# av_score_SF = np.zeros((nOrdering, len(Nobs_array)));
# st_score_SF = np.zeros((nOrdering, len(Nobs_array)));

for j in range(len(Nobs_array)):
	for iOrdering in range(nOrdering):
		f_IR = open('PRELIMINARY_PAPER_MCMC/regression_params_IR_' + Mode + '_' + Mode_Opt + '_n'+ str(Nobs_array[j]) + '_o' + str(iOrdering) + '.dat', 'w')
		# f_SR = open('PRELIMINARY_PAPER_MCMC/regression_params_SR_' + Mode + '_' + Mode_Opt + '_n'+ str(Nobs_array[j]) + '_o' + str(iOrdering) + '.dat', 'w')
		# f_SF = open('PRELIMINARY_PAPER_MCMC/regression_params_SF_' + Mode + '_' + Mode_Opt + '_n'+ str(Nobs_array[j]) + '_o' + str(iOrdering) + '.dat', 'w')

		reg_IR = []; reg_SR = []; reg_SF = [];
		sco_IR = []; sco_SR = []; sco_SF = [];
		kp_IR  = []; kp_SR  = []; kp_SF  = [];


		for i in MF_performance[iOrdering::nOrdering]: reg_IR.append(i[j].regression_param); sco_IR.append(i[j].score); kp_IR.append(i[j].kernel_param);
		# for i in LG_performance[iOrdering::nOrdering]: reg_SR.append(i[j].regression_param); sco_SR.append(i[j].score); kp_SR.append(i[j].kernel_param);
		# for i in SF_performance[iOrdering::nOrdering]: reg_SF.append(i[j].regression_param); sco_SF.append(i[j].score); kp_SF.append(i[j].kernel_param);

		f_IR.write('Mean: ' + str( np.array(reg_IR).mean(axis=0) ) + '\n');
		f_IR.write('Std:  ' + str( np.array(reg_IR).std(axis=0) ) + '\n');
		f_IR.write('Mean score:  ' + str( np.array(sco_IR).mean(axis=0) ) + '    Std score:  ' +  str( np.array(sco_IR).std(axis=0) ) + '\n');
		f_IR.write('Mean ker p:  ' + str( np.array(kp_IR ).mean(axis=0) ) + '    Std reg p:  ' +  str( np.array(kp_IR ).std(axis=0) ) + '\n');
		av_score_IR[iOrdering, j] = np.array(sco_IR ).mean(axis=0); st_score_IR[iOrdering, j] = np.array(sco_IR ).std(axis=0);

		# f_SR.write('Mean: ' + str( np.array(reg_SR).mean(axis=0) ) + '\n');
		# f_SR.write('Std:  ' + str( np.array(reg_SR).std(axis=0) ) + '\n');
		# f_SR.write('Mean score:  ' + str( np.array(sco_SR).mean(axis=0) ) + '    Std score:  ' +  str( np.array(sco_SR).std(axis=0) ) + '\n');
		# f_SR.write('Mean ker p:  ' + str( np.array(kp_SR ).mean(axis=0) ) + '    Std reg p:  ' +  str( np.array(kp_SR ).std(axis=0) ) + '\n');
		# av_score_SR[iOrdering, j] = np.array(sco_SR ).mean(axis=0); st_score_SR[iOrdering, j] = np.array(sco_SR ).std(axis=0);

		# f_SF.write('Mean: ' + str( np.array(reg_SF).mean(axis=0) ) + '\n');
		# f_SF.write('Std:  ' + str( np.array(reg_SF).std(axis=0) ) + '\n');
		# f_SF.write('Mean score:  ' + str( np.array(sco_SF).mean(axis=0) ) + '    Std score:  ' +  str( np.array(sco_SF).std(axis=0) ) + '\n');
		# f_SF.write('Mean ker p:  ' + str( np.array(kp_SF ).mean(axis=0) ) + '    Std reg p:  ' +  str( np.array(kp_SF ).std(axis=0) ) + '\n');
		# av_score_SF[iOrdering, j] = np.array(sco_SF ).mean(axis=0); st_score_SF[iOrdering, j] = np.array(sco_SF ).std(axis=0);

		f_IR.write('N reg param, score, kern params\n');
		# f_SR.write('N reg param, score, kern params\n');
		# f_SF.write('N reg param, score, kern params\n');


		for i in MF_performance[iOrdering::nOrdering]:
			reg_IR.append(i[j].regression_param);
			for k in i[j].regression_param:
				f_IR.write( str(k) + '    ');
			f_IR.write( str(i[j].score) + '    ');
			for k in i[j].kernel_param:
				f_IR.write( str(k) + '    ');
			f_IR.write('\n')

		# for i in LG_performance[iOrdering::nOrdering]:
		# 	for k in i[j].regression_param:
		# 		f_SR.write( str(k) + '    ');
		# 	f_SR.write( str(i[j].score) + '    ');
		# 	for k in i[j].kernel_param:
		# 		f_SR.write( str(k) + '    ');
		# 	f_SR.write('\n')

		# for i in SF_performance[iOrdering::nOrdering]:
		# 	for k in i[j].regression_param:
		# 		f_SF.write( str(k) + '    ');
		# 	f_SF.write( str(i[j].score) + '    ');
		# 	for k in i[j].kernel_param:
		# 		f_SF.write( str(k) + '    ');
		# 	f_SF.write('\n')

f_IR.close()
# f_SR.close()
# f_SF.close()



for iOrdering in range(nOrdering):
	string_save = 'PRELIMINARY_PAPER_MCMC/' + Mode + '_' + Mode_Opt + '_o' + str(iOrdering) + '_';
	if LASSO:      string_save+= 'LASSO_';
	if Matching:      string_save+= 'matching_';
	if Nested:        string_save+= 'nested_';
	if Equal_size:    string_save+= 'equal_';

	plt.figure()
	plt.plot(av_score_IR[iOrdering, :], color='r', label='IR')
	# plt.plot(av_score_SR[iOrdering, :], color='b', label='SR')
	# plt.plot(av_score_SF[iOrdering, :], color='g', label='SF')

	plt.xlabel(r'$N^l$', fontsize=FONTSIZE);
	plt.ylabel(r'$score_{AVG}$', fontsize=FONTSIZE);
	plt.xticks(np.arange(0, len(Nobs_array)), [str(j) for j in Nobs_array], fontsize=FONTSIZE);
	plt.yticks(fontsize=FONTSIZE);

	plt.legend(prop={'size': FONTSIZE}, frameon=False)
	plt.tight_layout()
	plt.savefig( string_save + 'nPoints_behavior_AVG.pdf');

	plt.figure()
	plt.plot(st_score_IR[iOrdering, :], color='r', label='IR')
	# plt.plot(st_score_SR[iOrdering, :], color='b', label='SR')
	# plt.plot(st_score_SF[iOrdering, :], color='g', label='SF')

	plt.xlabel(r'$N^l$', fontsize=FONTSIZE);
	plt.ylabel(r'$score_{STD}$', fontsize=FONTSIZE);
	plt.xticks(np.arange(0, len(Nobs_array)), [str(j) for j in Nobs_array], fontsize=FONTSIZE);
	plt.yticks(fontsize=FONTSIZE);

	plt.legend(prop={'size': FONTSIZE}, frameon=False)
	plt.tight_layout()
	plt.savefig( string_save + 'nPoints_behavior_STD.pdf');

exit()








