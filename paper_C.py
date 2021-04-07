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

import csv

from math import pi

from GP_module import GP
from models_module import *

import subprocess
command = ['rm PRELIMINARY_PAPER_C/*']
subprocess.call(command, shell=True)


# FONTSIZE=22
# Nobs_array = [ 6, 9, 12, 15 ];

# m1='MLLW';
# m2='MLL';
# av_score_IR=[];
# st_score_IR=[];
# for i in Nobs_array:
# 	with open('./PRELIMINARY_PAPER_A/regression_params_IR_n' + str(i) + '_o1.dat', 'r') as fid:
# 		lines = fid.read().splitlines()
# 	fid.close()
# 	av_score_IR.append(float(lines[2].split()[2]))
# 	st_score_IR.append(float(lines[2].split()[5]))
# 	print(lines[2].split()[2], lines[2].split()[5])

# av_score_SR=[];
# st_score_SR=[];
# for i in Nobs_array:
# 	with open('./PRELIMINARY_PAPER_A/regression_params_SR_n' + str(i) + '_o1.dat', 'r') as fid:
# 		lines = fid.read().splitlines()
# 	fid.close()
# 	av_score_SR.append(float(lines[2].split()[2]))
# 	st_score_SR.append(float(lines[2].split()[5]))
# 	print(lines[2].split()[2], lines[2].split()[5])

# av_score_SF=[];
# st_score_SF=[];
# for i in Nobs_array:
# 	with open('./PRELIMINARY_PAPER_A/regression_params_SF_n' + str(i) + '_o1.dat', 'r') as fid:
# 		lines = fid.read().splitlines()
# 	fid.close()
# 	av_score_SF.append(float(lines[2].split()[2]))
# 	st_score_SF.append(float(lines[2].split()[5]))
# 	print(lines[2].split()[2], lines[2].split()[5])


# plt.figure()
# plt.plot(av_score_IR, color='r', label='IR')
# plt.plot(av_score_SR, color='b', label='SR')
# plt.plot(av_score_SF, color='g', label='SF')

# plt.xlabel(r'$N^l$', fontsize=FONTSIZE);
# plt.ylabel(r'$score_{AVG}$', fontsize=FONTSIZE);
# plt.xticks(np.arange(0, len(Nobs_array)), [str(j) for j in Nobs_array], fontsize=FONTSIZE-8);
# plt.yticks(fontsize=FONTSIZE-8);

# plt.legend(prop={'size': FONTSIZE}, frameon=False)
# plt.tight_layout()
# plt.savefig( 'G_' + m1 + '_nPoints_behavior_AVG');

# plt.figure()
# plt.plot(st_score_IR, color='r', label='IR')
# plt.plot(st_score_SR, color='b', label='SR')
# plt.plot(st_score_SF, color='g', label='SF')

# plt.xlabel(r'$N^l$', fontsize=FONTSIZE);
# plt.ylabel(r'$score_{STD}$', fontsize=FONTSIZE);
# plt.xticks(np.arange(0, len(Nobs_array)), [str(j) for j in Nobs_array], fontsize=FONTSIZE-8);
# plt.yticks(fontsize=FONTSIZE-8);

# plt.legend(prop={'size': FONTSIZE}, frameon=False)
# plt.tight_layout()
# plt.savefig( 'G_' + m1 + '_nPoints_behavior_STD');

# plt.show()
# exit()

#"PointID","x","y","Density","Momentum_x","Momentum_y","Energy","Pressure","Temperature","Mach","Pressure_Coefficient"

#"PointID","x","y","Density","Momentum_x","Momentum_y","Energy","Nu_Tilde","Pressure","Temperature","Mach",
#"Pressure_Coefficient","Laminar_Viscosity","Skin_Friction_Coefficient_x","Skin_Friction_Coefficient_y","Heat_Flux","Y_Plus","Eddy_Viscosity"


Pref=101325;
Uref=34.6;
Rhoref=1.185;
Cref=0.42;

FONTSIZE = 22

def sortFunction(val): 
    return val[0] 


x_min = 0.0;
x_max = 2.0;

models_eu = ['eu_0', 'eu_2'];
models_ns = ['ns_1', 'ns_2'];


data = [];
for model in models_eu:
	data.append([]);
	with open('./PAPER_C_DATA/surf_'+ model + '.csv', 'r') as fid:
		csv_reader = csv.reader(fid, delimiter=',')
		for row in csv_reader:
			data[-1].append(np.transpose([row[1], row[10] ]));
		data[-1] = data[-1][1::];
		data[-1] = np.array(data[-1]);
		data[-1] = data[-1].astype(np.float);
		data[-1] = data[-1][data[-1][:,0].argsort()];
		data[-1][:, 0] /= Cref;
		data[-1] = data[-1][ (data[-1][:, 0] <= x_max), : ];
		data[-1] = data[-1][ (data[-1][:, 0] >= x_min), : ];
	fid.close()


for model in models_ns:
	data.append([]);
	with open('./PAPER_C_DATA/surf_'+ model + '.csv', 'r') as fid:
		csv_reader = csv.reader(fid, delimiter=',')
		for row in csv_reader:
			data[-1].append(np.transpose([row[1], row[11] ]));
		data[-1] = data[-1][1::];
		data[-1] = np.array(data[-1]);
		data[-1] = data[-1].astype(np.float);
		data[-1] = data[-1][data[-1][:,0].argsort()];
		data[-1][:, 0] /= Cref;
		data[-1] = data[-1][ (data[-1][:, 0] <= x_max), : ];
		data[-1] = data[-1][ (data[-1][:, 0] >= x_min), : ];
	fid.close()



Cp_exp=[]
with open('./PAPER_C_DATA/noflow_cp.dat', 'r') as fid:
	lines = fid.read().splitlines()
fid.close()

data.append([]);
for line in lines[5::]:
	if ( float(line.split()[0])+0.080454 ) >= data[0][ 0, 0] and ( float(line.split()[0])+0.080454 ) <= data[0][-1, 0]:
		Cp_exp.append(  [(float(line.split()[0])+0.080454), float(line.split()[1])]);
	data[-1].append([(float(line.split()[0])+0.080454), float(line.split()[1])]);
data[-1] = np.array(data[-1]);
data[-1] = data[-1][ (data[-1][:, 0] <= x_max), : ];
data[-1] = data[-1][ (data[-1][:, 0] >= x_min), : ];









Cp_exp = np.array(Cp_exp);







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

Mode_Opt=2
Mode_Opt_labels_IR= ['ML',  'LOO',   'DL',   'WA', 'AWLL', '  LS'];
Mode_Opt_list_IR  = ['MLL', 'LOO', 'MLLD', 'MLLW', 'MLLA', 'MLLS'];
Mode_Opt_labels_SR= ['ML',  'LOO', '  DL',   'ML', 'AWLL',   'LS'];
Mode_Opt_list_SR  = ['MLL', 'LOO', 'MLLD',  'MLL', 'MLLA', 'MLLS'];

LASSO_list     = [False, False, True, False, False, False];
Mode='G'#Don't touch this for the paper

LASSO= LASSO_list[0];

Nobs_array = [ 6, 9, 12, 15 ];
NdataRandomization= 100;
Nested= False;
Matching = False;
Equal_size= False;
Deterministic= False;

Activate_histogram_plot=True


Np = 1000;
xx = np.linspace(x_min, x_max, Np);

Nmod = len(data);

Tychonov_regularization_coeff= 1e-4;

gp_restart = 10;
kernel = ConstantKernel(1.0**2, (1.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-1, 1.0e1)) \
+ WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-8, 1.0e-0));

#kernel = ConstantKernel(1.0**2, (1.0e-1**2, 1.0e1**2)) * Matern(length_scale=1.0, length_scale_bounds=(1.0e-1, 1.0e1), nu=2.5) \
#+ WhiteKernel(noise_level=1.0e-1, noise_level_bounds=(1.0e-8, 1.0e-0));





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
	
		Nobs = Nobs_array[nn];
		model_order = [];
		tmp = np.arange(Nmod-1);
		np.random.shuffle(tmp)


		tmp = np.append(tmp, [Nmod-1], axis=0);


		model_order.append(tmp);

		# print(model_order)
		# exit()
		
		print("Number of observations " + str(Nobs) + " iRand " + str(iDataRandomization));
		print("Generating synthetic data")

		
		Nobs_model = [ (Nobs - 1) *2**(Nmod - i - 1) + 1   for i in range(Nmod)];

		Train_points = [];
		observations = [];
		for i in range(Nmod):
			index = [];
			for j in RandomDataGenerator.uniform(0, len(data[i][:, 0]), Nobs_model[i]): index.append(int(j));
			# print(index)
			# print(np.shape(data[i][index[:], 0]))
			# print(data[i][index[:], 0])
			Train_points.append( data[i][index[:], 0] );

			observations.append([]);
			for j in range(Nobs_model[i]):
				observations[i].append( [ data[i][index[j], 1], 1.0]);


		for i in range(Nmod):
			Train_points[i] = np.array(Train_points[i]);
			observations[i] = np.array( observations[i] );
			

		#print(Train_points)
		# print(np.shape(Train_points[0]))
		# print(np.shape(observations[0]))
		#exit()

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
					Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt_list_IR[Mode_Opt], LASSO=LASSO);
				else:
					Mfs.append( GP(kernel, [Mfs[i].predict for i in range( len(Mfs) )], mode=Mode) );
					Mfs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt_list_IR[Mode_Opt], LASSO=LASSO);
				print(len(Train_points[Nm]), Mfs[-1].regression_param)

			MGs = [];
			print('LeGratiet regression param');
			for Nm in model_order[iOrdering]:
				if not MGs: 
					MGs.append(GP(kernel, mode=Mode));
					MGs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt_list_SR[Mode_Opt], LASSO=LASSO);
				else:
					MGs.append( GP(kernel, [MGs[-1].predict], mode=Mode) );
					MGs[-1].fit(Train_points[Nm].reshape(-1, 1), observations[Nm][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt_list_SR[Mode_Opt], LASSO=LASSO);
				print(len(Train_points[Nm]), MGs[-1].regression_param)



			for Nm in model_order[iOrdering]:

				ax = plt.Subplot(it_frame, inner[ np.where(np.array(model_order[iOrdering]) == Nm )[0][0] ])

				yy, vv = Mfs[ np.where(np.array(model_order[iOrdering]) == Nm )[0][0] ].predict(xx.reshape(-1, 1), return_variance= True) 
				yy = yy.flatten();
				ss = np.sqrt(np.diag(vv))				
				ax.plot(xx, yy, color='b', label='GP')
				ax.fill_between(xx, yy-ss, yy+ss, facecolor='b', alpha=0.3, interpolate=True)

				ax.scatter(Train_points[Nm], observations[Nm][:, 0])
				ax.plot(data[Nm][:, 0], data[Nm][:, 1], color='r', label='M')
				ax.plot(Cp_exp[:, 0], Cp_exp[:, 1], color='k', label='T')

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
			MF_performance[-(nOrdering-iOrdering)][nn].LOGL 			= np.copy( Mfs[-1].compute_loglikelihood(Cp_exp[:, 0].reshape(-1, 1), Cp_exp[:, 0].reshape(-1, 1)) )
			MF_performance[-(nOrdering-iOrdering)][nn].score 			= np.copy( Mfs[-1].score(Cp_exp[:, 0].reshape(-1, 1), Cp_exp[:, 1].reshape(-1, 1)) )


			LG_performance[-(nOrdering-iOrdering)][nn].regression_param = np.copy( MGs[-1].regression_param.flatten() );
			LG_performance[-(nOrdering-iOrdering)][nn].kernel_param		= np.copy( np.copy( np.exp(MGs[-1].kernel.theta) ) );
			LG_performance[-(nOrdering-iOrdering)][nn].LOGL 			= np.copy( MGs[-1].compute_loglikelihood(Cp_exp[:, 0].reshape(-1, 1), Cp_exp[:, 0].reshape(-1, 1)) )
			LG_performance[-(nOrdering-iOrdering)][nn].score 			= np.copy( MGs[-1].score(Cp_exp[:, 0].reshape(-1, 1), Cp_exp[:, 1].reshape(-1, 1)) )


			yy, vv = Mfs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
			yy = yy.flatten();
			ss = np.sqrt(np.diag(vv))

			yg, vg = MGs[-1].predict(xx.reshape(-1, 1), return_variance= True) 
			yg = yg.flatten();
			sg = np.sqrt(np.diag(vg))

			GP_single = GP(kernel, mode=Mode);
			GP_single.fit(Train_points[-1].reshape(-1, 1), observations[-1][:, 0].reshape(-1, 1), Tychonov_regularization_coeff, Opt_Mode= Mode_Opt_list_SR[Mode_Opt], LASSO=LASSO);

			SF_performance[-(nOrdering-iOrdering)][nn].regression_param = np.copy( GP_single.regression_param.flatten() );
			SF_performance[-(nOrdering-iOrdering)][nn].kernel_param		= np.copy( np.exp(GP_single.kernel.theta) );
			SF_performance[-(nOrdering-iOrdering)][nn].LOGL 			= np.copy( GP_single.compute_loglikelihood(Cp_exp[:, 0].reshape(-1, 1), Cp_exp[:, 0].reshape(-1, 1)) )
			SF_performance[-(nOrdering-iOrdering)][nn].score 			= np.copy( GP_single.score(Cp_exp[:, 0].reshape(-1, 1), Cp_exp[:, 1].reshape(-1, 1)) )

			yy_s, vv_s = GP_single.predict(xx.reshape(-1, 1), return_variance= True) 
			yy_s = yy_s.flatten();
			ss_s = np.sqrt(np.diag(vv_s))

			inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec= outer[N_columns*iOrdering+1], wspace=0.1, hspace=0.1)
			ax = plt.Subplot(it_frame, inner[0])
			ax.scatter(Train_points[-1], observations[-1][:, 0])
			ax.plot(Cp_exp[:, 0], Cp_exp[:, 1], color='k', label='T')
			ax.plot(xx, yy, color='r', label='IR')
			ax.fill_between(xx, yy-ss, yy+ss, facecolor='r', alpha=0.3, interpolate=True)
			ax.plot(xx, yy_s, color='g', label='SF')
			ax.fill_between(xx, yy_s-ss_s, yy_s+ss_s, facecolor='g', alpha=0.3, interpolate=True)
			ax.plot(xx, yg, color='b', label='SR')
			ax.fill_between(xx, yg-sg, yg+sg, facecolor='b', alpha=0.3, interpolate=True)
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




	string_save = 'PRELIMINARY_PAPER_C/' + str(iDataRandomization) + '_' + Mode + '_' + Mode_Opt_list_IR[Mode_Opt] + '_';
	if LASSO:      string_save+= 'LASSO_';
	if Matching:      string_save+= 'matching_';
	if Nested:        string_save+= 'nested_';
	if Equal_size:    string_save+= 'equal_';

	for nn in range(len(Nobs_array)):
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
		f_IR = open('PRELIMINARY_PAPER_C/regression_params_IR_' + Mode + '_' + Mode_Opt_list_IR[Mode_Opt] + '_n'+ str(Nobs_array[j]) + '_o' + str(iOrdering) + '.dat', 'w')
		f_SR = open('PRELIMINARY_PAPER_C/regression_params_SR_' + Mode + '_' + Mode_Opt_list_SR[Mode_Opt] + '_n'+ str(Nobs_array[j]) + '_o' + str(iOrdering) + '.dat', 'w')
		f_SF = open('PRELIMINARY_PAPER_C/regression_params_SF_' + Mode + '_' + Mode_Opt_list_SR[Mode_Opt] + '_n'+ str(Nobs_array[j]) + '_o' + str(iOrdering) + '.dat', 'w')

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
	string_save = 'PRELIMINARY_PAPER_C/' + Mode + '_' + Mode_Opt_list_IR[Mode_Opt] + '_o' + str(iOrdering) + '_';
	if LASSO:      string_save+= 'LASSO_';
	if Matching:      string_save+= 'matching_';
	if Nested:        string_save+= 'nested_';
	if Equal_size:    string_save+= 'equal_';

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









