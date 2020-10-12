from matplotlib import pyplot as plt

import numpy as np
import scipy as sp

import math

import sys

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct, ConstantKernel, WhiteKernel

from math import pi



np.random.seed(0)

FONTSIZE = 22

Nobs = 2;

x_min = 0.0;
x_max = 1.0;

Np = 1000;
xx = np.linspace(x_min, x_max, Np);

# Define the model
def truth(x):
	return np.sin(x/x_max*pi/2);

def model_1(x):
	return x;

def model_2(x):
	return -x**2 + 2.0*x;

def model_3(x):
	return np.sin(x/x_max*pi/2) + 0.02;



plt.figure()

plt.plot(xx, truth(xx), color='k', label='Truth')
plt.plot(xx, model_1(xx), color='r', label='Model 1')
plt.plot(xx, model_2(xx), color='b', label='Model 2')
plt.plot(xx, model_3(xx), color='m', label='Model 3')

plt.legend(prop={'size': FONTSIZE}, frameon=False, loc='lower right')
plt.xlabel('x', fontsize=FONTSIZE)
plt.ylabel('y [-]', fontsize=FONTSIZE)
plt.tight_layout()





# observations
fx = np.random.uniform(x_min, x_max, Nobs)
fy_1 = [model_1(i) for i in fx];
fy_2 = [model_2(i) for i in fx];
fy_3 = [model_3(i) for i in fx];

fx = np.concatenate((fx, fx, fx), axis=None)
fy = np.concatenate((fy_1, fy_2, fy_3), axis=None)





gp_restart = 100;
#kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) + WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-6, 1.0e-0));
kernel = ConstantKernel(1.0**2, (3.0e-1**2, 1.0e1**2)) * RBF(length_scale=1.0, length_scale_bounds=(1.0e-2, 1.0e1)) + WhiteKernel(noise_level=1.0e-2, noise_level_bounds=(1.0e-6, 1.0e-0));

gp = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=gp_restart, alpha=(1.0e-2), normalize_y=True)

yy_mean_prior, yy_cov_prior = gp.predict(np.array(fx.reshape(-1, 1)), return_cov=True)
yy_std_prior = np.sqrt(np.diag(yy_cov_prior))

gp.fit(fx.reshape(-1, 1), fy.reshape(-1, 1))
yy_mean_post, yy_cov_post = gp.predict(np.array(fx.reshape(-1, 1)), return_cov=True)
yy_std_post = np.sqrt(np.diag(yy_cov_post))





eigVals, eigVecs = np.linalg.eig(yy_cov_post)

# print(eigVals)
# print(eigVecs.dot(np.diag(eigVals)).dot(eigVecs.T))

#print(eigVecs)

def loglikelihood(x, y_observations):
	y_observations.reshape(-1, 1);
	K = eigVecs.dot(np.diag(x)).dot(eigVecs.T) #+ 1e-8*np.eye(len(x))
	return  - ( -0.5*y_observations.T.dot(np.linalg.inv(K)).dot(y_observations) - 0.5*np.log(np.linalg.det(K)) )


constraints = ({'type': 'ineq', "fun": lambda x: x - 1e-8}, {'type': 'ineq', "fun": lambda x: - np.sum(x) + len(x)},)


MIN = 1e16;
for int_it in range(100):
	x0 = np.random.uniform(0.0, 1.0, len(eigVals)); 

	res = sp.optimize.minimize(loglikelihood, x0, args=(fy), method='SLSQP', constraints=constraints, tol=1e-5, options={'maxiter': 100, 'disp': False});
	if (loglikelihood(res.x, fy) < MIN):
		MIN = loglikelihood(res.x, fy);
		MIN_x = res.x;


yy_std_new = np.sqrt(np.diag(eigVecs.dot(np.diag(MIN_x)).dot(eigVecs.T)))



plt.figure()

plt.plot(xx, truth(xx), color='k', label='Truth')
plt.plot(xx, model_1(xx), color='r', label='Model 1')
plt.plot(xx, model_2(xx), color='b', label='Model 2')
plt.plot(xx, model_3(xx), color='m', label='Model 3')

plt.errorbar(fx+0.001, fy, yerr=yy_std_post, fmt='.k')

plt.errorbar(fx[0*Nobs:1*Nobs]+0.01, fy[0*Nobs:1*Nobs], yerr=yy_std_new[0*Nobs:1*Nobs], fmt='.r')
plt.errorbar(fx[1*Nobs:2*Nobs]+0.02, fy[1*Nobs:2*Nobs], yerr=yy_std_new[1*Nobs:2*Nobs], fmt='.b')
plt.errorbar(fx[2*Nobs:3*Nobs]+0.03, fy[2*Nobs:3*Nobs], yerr=yy_std_new[2*Nobs:3*Nobs], fmt='.m')

plt.axis([x_min, x_max, 0, 1.1])
plt.legend(prop={'size': FONTSIZE}, frameon=False, loc='lower right')
plt.xlabel('x', fontsize=FONTSIZE)
plt.ylabel('y [-]', fontsize=FONTSIZE)
plt.tight_layout()



plt.figure()
plt.plot(yy_std_post, color='r', label='GP')
plt.plot(yy_std_new, color='b', label='NV')

plt.legend(prop={'size': FONTSIZE}, frameon=False, loc='lower right')
plt.xlabel('x_i', fontsize=FONTSIZE)
plt.ylabel('std [-]', fontsize=FONTSIZE)
plt.tight_layout()



print('Loglikelihood GP: ', loglikelihood(eigVals, fy))
print('Loglikelihood GP: ', loglikelihood(MIN_x, fy))


print(eigVals)
# print(eigVecs.dot(np.diag(eigVals)).dot(eigVecs.T))
print(MIN_x)
# print(eigVecs.dot(np.diag(MIN_x)).dot(eigVecs.T))


plt.show()
exit()




