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
command = ['rm PRELIMINARY_PAPER_D/*']
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

Mode_Opt_labels= ['ML',  'LOO', 'DL', 'WA', 'AWLL', 'LS'];
Mode_Opt_list  = ['MLL', 'LOO', 'MLLD', 'MLLW', 'MLLA', 'MLLS'];
LASSO_list     = [False, False, True, False, False, False];
Mode='G'#Don't touch this for the paper

BudgetMax = 2;
Nobs_array = [ 7 ];
NdataRandomization= 1;


Activate_histogram_plot=False
Activate_VarianceDec_plot=True

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



nOrdering = 1; # Don't touch or this!!!! Could cause problems with the re-sampling
N_columns = 2;
if Activate_histogram_plot or Activate_VarianceDec_plot: N_columns += 1;
if Activate_histogram_plot and Activate_VarianceDec_plot: print("Error, only histogram OR var dec"); exit();


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
		MF_performance.append([PerformanceRecord() for i in range(BudgetMax)]);
		SF_performance.append([PerformanceRecord() for i in range(BudgetMax)]);
		LG_performance.append([PerformanceRecord() for i in range(BudgetMax)]);


	fig_frame = [];
	for iOrdering in range( BudgetMax ):
		fig_frame.append(plt.figure(figsize=(14, 8)));
	outer = gridspec.GridSpec( nOrdering, N_columns, wspace= 0.2, hspace= 0.3 );


	Nobs_model     = [ Nobs_array[0] for i in range(Nmod)]; # They start with equal size !!!!!!!!!!!
	#Nobs_model[-1] = Nobs_array[0];

	for nn in range(BudgetMax):
	
		model_order = [];
		model_order.append(np.arange(Nmod).flatten());
		
		print("Budget allocated " + str(float(nn)/(BudgetMax-1)) + " iRand " + str(iDataRandomization));
		print("Re-sampling...")

		highlight = np.zeros((Nmod, 1));
		if nn != 0:
			# for i in range(Nmod):
			# 	Train_points[i] = np.append( Train_points[i], RandomDataGenerator.uniform(x_min, x_max, 1)[0] )
			# 	Nobs_model[i] += 1;

			TrainHat = [];
			for i in Train_points[-1]:
				tmp = [];
				for j in range(Nmod-1):
					tmp.append( Mfs[j].predict(np.array([i]).reshape(-1, 1), return_variance= False) );

				TrainHat.append( np.var(np.array(tmp)) );
			TrainHat = np.array(TrainHat);

			kernel_hat = ConstantKernel(1.0**2, (1.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-1, 1.0e1));
			GP_hat = GP(kernel_hat, mode=Mode);
			GP_hat.fit(Train_points[-1].reshape(-1, 1), TrainHat.reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);

			#mu_hat, sigma_hat = GP_hat.predict(xx.reshape(-1, 1), return_variance= True) 

			def obj1(x): a,b = Mfs[-1].predict(np.array([x]).reshape(-1, 1), return_variance= True); return b[0];
			def obj2(x): a,b =  GP_hat.predict(np.array([x]).reshape(-1, 1), return_variance= True); return 0;#b[0];
			def obj(x): a = 1.0; return -obj1(x);# - (a+obj1(x)) * (a+obj2(x));

			MIN = float("inf");
			bnd = ((x_min, x_max),);
			for int_it in range(10):
				x0 = np.random.uniform(x_min, x_max);
				res = sp.optimize.minimize(obj, x0, method="L-BFGS-B", bounds=bnd, tol= 1e-09, options={'disp': None, 'maxcor': 10, 'ftol': 1e-09, 'maxiter': 15000})
				if (obj(res.x) < MIN):
					MIN   = obj(res.x)
					MIN_x = np.copy(res.x);
			print(MIN_x)


			if obj2(MIN_x) > obj1(MIN_x) and 1==0: # remove second condition for original
				for i in range(Nmod):
					Train_points[i] = np.append( Train_points[i], MIN_x )
					Nobs_model[i] += 1;
					highlight[i]   = 1;
			else:
				delta_i = []; 
				for i in range(len(Mfs[-1].regression_param)):
					MfOtp = copy.deepcopy(Mfs[-1]);
					MfOtp.regression_param[i] = 0.0;
					print(MfOtp.regression_param.flatten());
					delta_i.append( (obj1(MIN_x) - MfOtp.predict(np.array( MIN_x ).reshape(-1, 1), return_variance= True)[1][0,0]) );

				# MfOtp = copy.deepcopy(Mfs[-1]);
				# for i in range(len(Mfs[-1].regression_param)): MfOtp.regression_param[i] = 0.0;
				# print(MfOtp.regression_param.flatten());	
				# delta_i.append( (MfOtp.predict(np.array( MIN_x ).reshape(-1, 1), return_variance= True)[1][0,0]) );	#sigma_L-(sigma_L-contributionfromL)		

				# print("Delta")
				# print(MfOtp.predict(np.array( MIN_x ).reshape(-1, 1), return_variance= True)[1][0,0])
				# print(np.shape(MfOtp.predict(np.array( MIN_x ).reshape(-1, 1), return_variance= True)[1][0,0]))
				# print(np.array(delta_i))
				#print(np.array(delta_i)/np.sum(np.array(delta_i)))
				Train_points[ delta_i.index(max(delta_i)) ]  = np.append( Train_points[ delta_i.index(max(delta_i)) ], MIN_x )
				Nobs_model[   delta_i.index(max(delta_i)) ] += 1;
				highlight[    delta_i.index(max(delta_i)) ]  = 1;



		else: 
			Train_points = [];
			for i in range(Nmod):
				Train_points.append( RandomDataGenerator.uniform(x_min, x_max, Nobs_model[i]) );


		

		observations = [];
		for i in range(Nmod):
			observations.append([]);
			for j in range(Nobs_model[i]):
				observations[i].append(models[i](Train_points[i][j]));

		for i in range(Nmod):
			observations[i] = np.array(observations[i]);


		it_frame = fig_frame[nn];
		it_frame.suptitle('Budget ' + str(float(nn)/(BudgetMax-1)) );

		for iOrdering in range(nOrdering):	

			inner = gridspec.GridSpecFromSubplotSpec(Nmod, 1, subplot_spec= outer[N_columns*iOrdering], wspace=0.1, hspace=0.1)
			
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
				print(len(Train_points[Nm]), Mfs[-1].regression_param)


			MGs = [];
			print('LeGratiet regression param');
			for Nm in model_order[iOrdering]:
				if not MGs: 
					MGs.append(GP(kernel, mode=Mode));
					MGs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);
				else:
					MGs.append( GP(kernel, [MGs[-1].predict], mode=Mode) );
					MGs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);
				print(len(Train_points[Nm]), MGs[-1].regression_param)



			for Nm in model_order[iOrdering]:

				ax = plt.Subplot(it_frame, inner[ np.where(np.array(model_order[iOrdering]) == Nm )[0][0] ])

				yy, vv = Mfs[ np.where(np.array(model_order[iOrdering]) == Nm )[0][0] ].predict(xx.reshape(-1, 1), return_variance= True) 
				yy = yy.flatten();
				ss = np.sqrt(np.diag(vv))				
				ax.plot(xx, yy, color='r', label='GP')
				ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)

				ax.scatter(Train_points[Nm], observations[Nm][:, 0])
				ax.plot(xx, models[Nm](xx)[:][0], color='k', label='T')

				if (highlight[Nm] == 1): ax.scatter(Train_points[Nm][-1], observations[Nm][-1, 0], marker='v', color='r')


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


			LG_performance[-(nOrdering-iOrdering)][nn].regression_param = np.copy( MGs[-1].regression_param.flatten() );
			LG_performance[-(nOrdering-iOrdering)][nn].kernel_param		= np.copy( np.copy( np.exp(MGs[-1].kernel.theta) ) );
			LG_performance[-(nOrdering-iOrdering)][nn].LOGL 			= np.copy( MGs[-1].compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )
			LG_performance[-(nOrdering-iOrdering)][nn].score 			= np.copy( MGs[-1].score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )


			yy, vv = Mfs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
			yy = yy.flatten();
			ss = np.sqrt(np.diag(vv))


			yg, vg = MGs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
			yg = yg.flatten();
			sg = np.sqrt(np.diag(vg))

			GP_single = GP(kernel, mode=Mode);
			GP_single.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt, LASSO=LASSO);

			SF_performance[-(nOrdering-iOrdering)][nn].regression_param = np.copy( GP_single.regression_param.flatten() );
			SF_performance[-(nOrdering-iOrdering)][nn].kernel_param		= np.copy( np.exp(GP_single.kernel.theta) );
			SF_performance[-(nOrdering-iOrdering)][nn].LOGL 			= np.copy( GP_single.compute_loglikelihood(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )
			SF_performance[-(nOrdering-iOrdering)][nn].score 			= np.copy( GP_single.score(xx.reshape(-1, 1), truth(xx)[0].reshape(-1, 1)) )

			yy_s, vv_s = GP_single.predict(xx.reshape(-1, 1), return_variance= True) 
			yy_s = yy_s.flatten();
			ss_s = np.sqrt(np.diag(vv_s))

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




			if Activate_VarianceDec_plot:
				inner = gridspec.GridSpecFromSubplotSpec(Nmod-1, 1, subplot_spec= outer[N_columns*iOrdering+2], wspace=0.1, hspace=0.1)

				for i in range(len(Mfs[-1].regression_param)):
					ax = plt.Subplot(it_frame, inner[i])
					MfOtp = copy.deepcopy(Mfs[-1]);
					MfOtp.regression_param[i] = 0.0;
					yyi, vvi = MfOtp.predict(xx.reshape(-1, 1), return_variance= True) 
					yyi = yyi.flatten();
					ssi = np.sqrt(np.diag(vvi))				
					ax.plot(xx, ss, color='k', label=r'$\sigma^2_{L}$')
					ax.plot(xx, ssi, color='r', label=r'$\sigma^2_{L/' + str(i+1) + '}$')
					ax.fill_between(xx, ssi-ssi, ssi, facecolor='r', alpha=0.3, interpolate=True)
					ax.fill_between(xx, ssi, ss, facecolor='b', alpha=0.3, interpolate=True)
					#ax.set_yticks([-ss.max(), ss.max()])
					ax.legend(prop={'size': FONTSIZE-6}, frameon=False)

					ax.tick_params(axis='both', labelsize=FONTSIZE)
					if i != len(Mfs[-1].regression_param)-1: 
						ax.xaxis.set_major_formatter(plt.NullFormatter())
					it_frame.add_subplot(ax)



			# model_order.append(np.arange(Nmod).flatten());
			# model_order[-1][2] = 3;
			# model_order[-1][3] = 2;



	string_save = 'PRELIMINARY_PAPER_D/' + str(iDataRandomization) + '_' + Mode + '_' + Mode_Opt + '_';
	if LASSO:      string_save+= 'LASSO_';

	for nn in range(BudgetMax):
		fig_frame[nn].tight_layout()
		fig_frame[nn].savefig( string_save + str(nn) + '.pdf')


	
av_score_IR = np.zeros((nOrdering, len(Nobs_array)));
st_score_IR = np.zeros((nOrdering, len(Nobs_array)));

av_score_SR = np.zeros((nOrdering, len(Nobs_array)));
st_score_SR = np.zeros((nOrdering, len(Nobs_array)));

av_score_SF = np.zeros((nOrdering, len(Nobs_array)));
st_score_SF = np.zeros((nOrdering, len(Nobs_array)));

for j in range(len(Nobs_array)):
	for iOrdering in range(nOrdering):
		f_IR = open('PRELIMINARY_PAPER_D/regression_params_IR_' + Mode + '_' + Mode_Opt + '_n'+ str(Nobs_array[j]) + '_o' + str(iOrdering) + '.dat', 'w')
		f_SR = open('PRELIMINARY_PAPER_D/regression_params_SR_' + Mode + '_' + Mode_Opt + '_n'+ str(Nobs_array[j]) + '_o' + str(iOrdering) + '.dat', 'w')
		f_SF = open('PRELIMINARY_PAPER_D/regression_params_SF_' + Mode + '_' + Mode_Opt + '_n'+ str(Nobs_array[j]) + '_o' + str(iOrdering) + '.dat', 'w')

		reg_IR = []; reg_SR = []; reg_SF = [];
		sco_IR = []; sco_SR = []; sco_SF = [];
		kp_IR  = []; kp_SR  = []; kp_SF  = [];


		for i in MF_performance[iOrdering::nOrdering]: reg_IR.append(i[j].regression_param); sco_IR.append(i[j].score); kp_IR.append(i[j].kernel_param);
		for i in LG_performance[iOrdering::nOrdering]: reg_SR.append(i[j].regression_param); sco_SR.append(i[j].score); kp_SR.append(i[j].kernel_param);
		for i in SF_performance[iOrdering::nOrdering]: reg_SF.append(i[j].regression_param); sco_SF.append(i[j].score); kp_SF.append(i[j].kernel_param);

		f_IR.write('Mean: ' + str( np.array(reg_IR).mean(axis=0) ) + '\n');
		f_IR.write('Std:  ' + str( np.array(reg_IR).std(axis=0) ) + '\n');
		f_IR.write('Mean score:  ' + str( np.array(sco_IR).mean(axis=0) ) + '    Std score:  ' +  str( np.array(sco_IR).std(axis=0) ) + '\n');
		f_IR.write('Mean ker p:  ' + str( np.array(kp_IR ).mean(axis=0) ) + '    Std reg p:  ' +  str( np.array(kp_IR ).std(axis=0) ) + '\n');
		av_score_IR[iOrdering, j] = np.array(sco_IR ).mean(axis=0); st_score_IR[iOrdering, j] = np.array(sco_IR ).std(axis=0);

		f_SR.write('Mean: ' + str( np.array(reg_SR).mean(axis=0) ) + '\n');
		f_SR.write('Std:  ' + str( np.array(reg_SR).std(axis=0) ) + '\n');
		f_SR.write('Mean score:  ' + str( np.array(sco_SR).mean(axis=0) ) + '    Std score:  ' +  str( np.array(sco_SR).std(axis=0) ) + '\n');
		f_SR.write('Mean ker p:  ' + str( np.array(kp_SR ).mean(axis=0) ) + '    Std reg p:  ' +  str( np.array(kp_SR ).std(axis=0) ) + '\n');
		av_score_SR[iOrdering, j] = np.array(sco_SR ).mean(axis=0); st_score_SR[iOrdering, j] = np.array(sco_SR ).std(axis=0);

		f_SF.write('Mean: ' + str( np.array(reg_SF).mean(axis=0) ) + '\n');
		f_SF.write('Std:  ' + str( np.array(reg_SF).std(axis=0) ) + '\n');
		f_SF.write('Mean score:  ' + str( np.array(sco_SF).mean(axis=0) ) + '    Std score:  ' +  str( np.array(sco_SF).std(axis=0) ) + '\n');
		f_SF.write('Mean ker p:  ' + str( np.array(kp_SF ).mean(axis=0) ) + '    Std reg p:  ' +  str( np.array(kp_SF ).std(axis=0) ) + '\n');
		av_score_SF[iOrdering, j] = np.array(sco_SF ).mean(axis=0); st_score_SF[iOrdering, j] = np.array(sco_SF ).std(axis=0);

		f_IR.write('N reg param, score, kern params\n');
		f_SR.write('N reg param, score, kern params\n');
		f_SF.write('N reg param, score, kern params\n');


		for i in MF_performance[iOrdering::nOrdering]:
			reg_IR.append(i[j].regression_param);
			for k in i[j].regression_param:
				f_IR.write( str(k) + '    ');
			f_IR.write( str(i[j].score) + '    ');
			for k in i[j].kernel_param:
				f_IR.write( str(k) + '    ');
			f_IR.write('\n')

		for i in LG_performance[iOrdering::nOrdering]:
			for k in i[j].regression_param:
				f_SR.write( str(k) + '    ');
			f_SR.write( str(i[j].score) + '    ');
			for k in i[j].kernel_param:
				f_SR.write( str(k) + '    ');
			f_SR.write('\n')

		for i in SF_performance[iOrdering::nOrdering]:
			for k in i[j].regression_param:
				f_SF.write( str(k) + '    ');
			f_SF.write( str(i[j].score) + '    ');
			for k in i[j].kernel_param:
				f_SF.write( str(k) + '    ');
			f_SF.write('\n')

f_IR.close()
f_SR.close()



for iOrdering in range(nOrdering):
	string_save = 'PRELIMINARY_PAPER_D/' + Mode + '_' + Mode_Opt + '_o' + str(iOrdering) + '_';
	if LASSO:      string_save+= 'LASSO_';

	plt.figure()
	plt.plot(av_score_IR[iOrdering, :], color='r', label='IR')
	plt.plot(av_score_SR[iOrdering, :], color='b', label='SR')
	plt.plot(av_score_SF[iOrdering, :], color='g', label='SF')

	plt.xlabel(r'$N^l$', fontsize=FONTSIZE);
	plt.ylabel(r'$score_{AVG}$', fontsize=FONTSIZE);
	plt.xticks(np.arange(0, len(Nobs_array)), [str(j) for j in Nobs_array], fontsize=FONTSIZE);
	plt.yticks(fontsize=FONTSIZE);

	plt.legend(prop={'size': FONTSIZE}, frameon=False)
	plt.tight_layout()
	plt.savefig( string_save + 'nPoints_behavior_AVG.pdf');

	plt.figure()
	plt.plot(st_score_IR[iOrdering, :], color='r', label='IR')
	plt.plot(st_score_SR[iOrdering, :], color='b', label='SR')
	plt.plot(st_score_SF[iOrdering, :], color='g', label='SF')

	plt.xlabel(r'$N^l$', fontsize=FONTSIZE);
	plt.ylabel(r'$score_{STD}$', fontsize=FONTSIZE);
	plt.xticks(np.arange(0, len(Nobs_array)), [str(j) for j in Nobs_array], fontsize=FONTSIZE);
	plt.yticks(fontsize=FONTSIZE);

	plt.legend(prop={'size': FONTSIZE}, frameon=False)
	plt.tight_layout()
	plt.savefig( string_save + 'nPoints_behavior_STD.pdf');

exit()








