from matplotlib import pyplot as plt
import numpy as np

import csv
import os




n_alpha = 25
alpha = np.linspace(0, n_alpha, n_alpha+1);

grids = ['A', 'B', 'C', 'D', 'E', 'S_1', 'S_2', 'S_3']


for G in grids:
	os.chdir('DATA_NACA0012/POLAR_' + G)
	exec(G + '_CL_epm = np.zeros((n_alpha+1, 5))')
	exec(G + '_CD_epm = np.zeros((n_alpha+1, 5))')
	exec(G + '_EF_epm = np.zeros((n_alpha+1, 5))')
	exec(G + '_CM_epm = np.zeros((n_alpha+1, 5))')

	exec(G + '_CL_baseline = np.zeros((n_alpha+1, 1))')
	exec(G + '_CD_baseline = np.zeros((n_alpha+1, 1))')
	exec(G + '_EF_baseline = np.zeros((n_alpha+1, 1))')
	exec(G + '_CM_baseline = np.zeros((n_alpha+1, 1))')

	exec(G + '_CL_trust = np.zeros((n_alpha+1, 1))')
	exec(G + '_CD_trust = np.zeros((n_alpha+1, 1))')
	exec(G + '_EF_trust = np.zeros((n_alpha+1, 1))')
	exec(G + '_CM_trust = np.zeros((n_alpha+1, 1))')


	for i in range(5):
		with open('EPM_' + str(i + 1) + '.txt', 'r') as f:
			lines = f.read().splitlines()

		for k in range(n_alpha + 1):
			#tmp = np.array( map(float, lines[k].split()) )
			tmp = np.array( lines[k].split()) 
			exec(G + '_CL_epm[k][i]= tmp[0]')
			exec(G + '_CD_epm[k][i]= tmp[1]')
			exec(G + '_EF_epm[k][i]= tmp[2]')
			exec(G + '_CM_epm[k][i]= tmp[3]')

		f.close()

	for k in range(n_alpha + 1):
		exec(G + '_CL_trust[k]= ' + G + '_CL_epm[k].max() - ' + G + '_CL_epm[k].min()')
		exec(G + '_CD_trust[k]= ' + G + '_CD_epm[k].max() - ' + G + '_CD_epm[k].min()')
		exec(G + '_EF_trust[k]= ' + G + '_EF_epm[k].max() - ' + G + '_EF_epm[k].min()')
		exec(G + '_CM_trust[k]= ' + G + '_CM_epm[k].max() - ' + G + '_CM_epm[k].min()')



	with open('baseline.txt', 'r') as f:
		lines = f.read().splitlines()

	for k in range(n_alpha + 1):
		#tmp = np.array( map(float, lines[k].split()) )
		tmp = np.array( lines[k].split()) 
		exec(G + '_CL_baseline[k] = tmp[0]')
		exec(G + '_CD_baseline[k] = tmp[1]')
		exec(G + '_EF_baseline[k] = tmp[2]')
		exec(G + '_EF_baseline[k] = tmp[3]')

	f.close();
	os.chdir('../../')





Ladson_transition80= [[2.05,    .2125,  .00816],
[4.04,    .4316,  .00823],
[6.09,    .6546,  .00885],
[8.30,    .8873,  .01050],
[10.12,  1.0707,  .01201],
[11.13,  1.1685,  .01239],
[12.12,  1.2605,  .01332],
[13.08,  1.3455,  .01503],
[14.22,  1.4365,  .01625],
[15.26,  1.5129,  .01900],
[16.30,  1.5739,  .02218],
[17.13,  1.6116,  .02560],
[18.02,   .9967,  .18785],
[19.08,  1.1358,  .27292]]

Ladson_transition120= [[.01,    -0.0122,  .00804],
[2.15,    .2236,  .00823],
[4.11,    .4397,  .00879],
[6.01,    .6487,  .00842],
[8.08,    .8701,  .00995],
[10.10,  1.0775,  .01175],
[11.23,  1.1849,  .01248],
[12.13,  1.2720,  .01282],
[13.26,  1.3699,  .01408],
[14.30,  1.4571,  .01628],
[15.27,  1.5280,  .01790],
[16.16,  1.5838,  .02093],
[17.24,  1.6347,  .02519],
[18.18,  1.1886,  .25194],
[19.25,  1.1888,  .28015]]

Ladson_transition180= [[.04,    -.0013,  .00811],
[2.00,    .2213,  .00814],
[4.06,    .4365,  .00814],
[6.09,    .6558,  .00851],
[8.09,    .8689,  .00985],
[10.18,  1.0809,  .01165],
[11.13,  1.1731,  .01247],
[12.10,  1.2644,  .01299],
[13.31,  1.3676,  .01408],
[14.08,  1.4316,  .01533],
[15.24,  1.5169,  .01870],
[16.33,  1.5855,  .02186],
[17.13,  1.6219,  .02513],
[18.21,  1.0104,  .25899],
[19.27,  1.0664,  .43446]]


Abbot_Van_Doenhoff_CL = [[0.0, 0.0],
[0.940006, 0.120611],
[1.96944, 0.215533],
[2.99515, 0.34477],
[3.85131, 0.439599],
[4.87888, 0.551678],
[5.90831, 0.6466],
[7.96346, 0.870758],
[10.1891, 1.12074],
[11.0471, 1.19842],
[13.1088, 1.36252],
[16.3759, 1.59591],
[16.5678, 1.42443],
[17.2971, 1.09024]]

cl_cd = [[0.0205051, 0.00593438],
[0.115637, 0.00592927],
[0.262707, 0.00600738],
[0.461945, 0.00659882],
[0.557543, 0.0074539],
[0.661509, 0.00779239],
[0.878463, 0.00915707],
[1.07854, 0.0112969],
[1.26127, 0.0133515]];

Abbot_Van_Doenhoff_CL = np.array(Abbot_Van_Doenhoff_CL)
cl_cd = np.array(cl_cd)
Ladson_transition80 = np.array(Ladson_transition80)
Ladson_transition120= np.array(Ladson_transition120)
Ladson_transition180= np.array(Ladson_transition180)



def access_data(fidelity_deg):
	if   fidelity_deg == 0: a = alpha[0:14]; b = A_CL_baseline[0:14];
	elif fidelity_deg == 1: a = alpha[0:14]; b = B_CL_baseline[0:14];
	elif fidelity_deg == 2: a = alpha[0:14]; b = C_CL_baseline[0:14];
	elif fidelity_deg == 3: a = alpha[0:14]; b = D_CL_baseline[0:14];
	elif fidelity_deg == 4: a = alpha[0:14]; b = E_CL_baseline[0:14];

	elif fidelity_deg == 5: a = Abbot_Van_Doenhoff_CL[0:14, 0]; b = Abbot_Van_Doenhoff_CL[0:24, 1];

	elif fidelity_deg == 6: a = np.array(Ladson_transition80[ 0:14, 0]).reshape(-1, 1); b = np.array(Ladson_transition80[ 0:14, 1]).reshape(-1, 1);
	elif fidelity_deg == 7: a = np.array(Ladson_transition120[0:14, 0]).reshape(-1, 1); b = np.array(Ladson_transition120[0:14, 1]).reshape(-1, 1);
	elif fidelity_deg == 8: a = np.array(Ladson_transition180[0:14, 0]).reshape(-1, 1); b = np.array(Ladson_transition180[0:14, 1]).reshape(-1, 1);

	else: print("Invalid model choice in database"); exit();

	r = np.zeros((14, 3));
	for i in range(14):
		r[i][0] = a[i];
		r[i][1] = b[i];
		r[i][2] = 0.0;
	# print(type(r))
	# print(np.shape(r))
	return r;















