import numpy as np
import scipy as sp
from scipy.linalg import cholesky, cho_solve, solve_triangular
import math
import copy

#from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct, ConstantKernel, WhiteKernel



class GP:
	def __init__(self, Kernel, Basis= None):
		self.kernel = copy.deepcopy(Kernel);
		if Basis is None: 
			self.basis_function = None;
			self.Nbasis= 0;
		else:
			self.basis_function = copy.deepcopy(Basis);
			self.Nbasis= len(Basis);


	def loglikelihood(self):
		return np.array(0.5*(self.Training_values - self.basis.T.dot(self.regression_param) ).T.dot(self.alpha) + np.log(np.diag(self.L)).sum()).flatten();

	def compute_loglikelihood(self, x, y):
		if self.basis_function is None: 
			Basis = np.zeros((self.Nbasis, len(x)));
		else:
			Basis = [];
			for i in range(self.Nbasis): Basis.append( self.basis_function[i](x).flatten() );
			Basis = np.array(Basis);
		K = self.kernel(x);
		if type(self.noise_level) is float: K[np.diag_indices_from(K)] += self.noise_level;
		L = cholesky(K, lower=True); 
		alpha = cho_solve((L, True), (y - Basis.T.dot(self.regression_param) ) )
		return - np.array(0.5*(y - Basis.T.dot(self.regression_param) ).T.dot(alpha) - np.log(np.diag(L)).sum()).flatten() - 0.5*len(x)*np.log(2*math.pi);

	def cost_function(self, theta):
		self.kernel.theta = theta[0:len(self.kernel.theta)];
		if (len(self.kernel.theta) != len(theta)):
			self.regression_param = np.array(theta[len(self.kernel.theta)::]).reshape(-1, 1);
		K = self.kernel(self.Training_points);
		if type(self.noise_level) is float:
			K[np.diag_indices_from(K)] += self.noise_level;
		else:
			for i in range(len(self.noise_level)): K[i, i] += self.noise_level[i];
		
		self.L = cholesky(K, lower=True); 
		self.alpha = cho_solve((self.L, True), (self.Training_values - self.basis.T.dot(self.regression_param) ) )
		return self.loglikelihood();


	def fit(self, Training_points, Training_values, noise_level):

		self.noise_level = copy.deepcopy(noise_level);
		self.Training_points = copy.deepcopy(Training_points);
		self.Training_values = copy.deepcopy(Training_values);

		MIN = 1e16;
		bounds = self.kernel.bounds

		if self.basis_function is None: 
			self.Nbasis = 1;
			self.basis = np.ones((self.Nbasis, len(self.Training_points)))
			self.regression_param = np.zeros((1, 1));
		else:
			self.basis = [];
			for i in range(self.Nbasis): self.basis.append( self.basis_function[i](self.Training_points).flatten() );
			self.basis = np.array(self.basis);
			self.regression_param = np.ones((np.shape(self.basis)[0], 1));
			for i in self.regression_param: bounds = np.append(bounds, [[-10.0, 10.0]], axis=0)


		for int_it in range(10):
			InternalRandomGenerator = np.random.RandomState();
			x0 = InternalRandomGenerator.uniform(bounds[:, 0], bounds[:, 1]);
			res = sp.optimize.minimize(self.cost_function, x0, method="L-BFGS-B", bounds=bounds)
			#res = sp.optimize.minimize(self.cost_function, x0, method="L-BFGS-B", bounds=bounds, tol= 1e-09, options={'disp': None, 'maxcor': 10, 'ftol': 1e-09, 'maxiter': 15000})
			if (self.cost_function(res.x)[0] < MIN):
				MIN = self.cost_function(res.x)
				MIN_x = res.x;

		if self.basis_function is None:
			self.kernel.theta = MIN_x;
		else:
			self.kernel.theta = MIN_x[0:len(self.kernel.theta)];
			self.regression_param = MIN_x[len(self.kernel.theta)::].reshape(-1, 1);

		Ktt = self.kernel(self.Training_points);
		if type(self.noise_level) is float:
			Ktt[np.diag_indices_from(Ktt)] += self.noise_level;
		else:
			for i in range(len(self.noise_level)): Ktt[i, i] += self.noise_level[i];
		self.L = cholesky(Ktt, lower=True);
		self.alpha = cho_solve((self.L, True), (self.Training_values - np.array(self.basis.T.dot(self.regression_param)) ))



	def predict(self, x, return_variance= False):
		if self.basis_function is None: 
			Basis = np.zeros((self.Nbasis, len(x)));
			Basis_v = np.zeros((self.Nbasis, len(x)));
		else:
			Basis = [];
			Basis_v = [];
			for i in range(self.Nbasis): 
				a = self.basis_function[i](x, True)
				#print(np.shape(a[0]))
				Basis.append(   a[0].flatten() );
				Basis_v.append( np.diag(a[1]).flatten() );
			Basis   = np.array(Basis);
			Basis_v = np.array(Basis_v);

		k = self.kernel(x, self.Training_points);
		mean = np.array(Basis.T.dot(np.array(self.regression_param))) + k.dot(self.alpha);
		v = cho_solve((self.L, True), k.T);
		variance = self.kernel(x) - k.dot(v) + np.square(self.regression_param.T).dot(np.square(Basis_v));

		if return_variance is True:
			return mean.T, variance;
		else:
			return mean.T;


	def score(self, x, y, sample_weight=None):
		return 1 - ((y - self.predict(x).T)**2).sum() / ((y - y.mean())**2).sum()


	def Qcriteria(self, x, y, eps=0.00):
		y_gp, sigma_gp = self.predict(x, return_variance= True);
		y_gp = y_gp.reshape(-1, 1);
		sigma_gp = np.sqrt(np.diag(sigma_gp)).reshape(-1, 1)

		return np.absolute( 0.5*sp.special.erf( np.divide((y - y_gp) , (np.sqrt(2)*(sigma_gp+eps) )) ) ) * ( np.absolute(y - y_gp) * sigma_gp)
		#return np.absolute( 0.5*sp.special.erf( np.divide((y - y_gp) , (np.sqrt(2)*(sigma_gp+eps) )) ) ) * np.absolute(y - y_gp) / sigma_gp ;


	def L2normCreteria(self, x, y, eps=0.00):
		y_gp= self.predict(x);
		y_gp = y_gp.reshape(-1, 1);

		return np.absolute( y - y_gp ).sum();
		


