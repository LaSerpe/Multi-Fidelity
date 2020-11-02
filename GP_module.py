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

	def compute_Gramm_matrix(self, x1, x2):
		K = self.kernel(x1, x2);
		for i in range(self.Nbasis): 
			K += self.basis_function[i](x1, x2, return_variance=True)[1]*self.regression_param[i]**2;
		if type(self.noise_level) is float: K[np.diag_indices_from(K)] += self.noise_level;
		return K;


	def compute_loglikelihood(self, x1, y): #to check
		Basis = np.zeros((len(x1), 1));
		if self.basis_function is not None: 
			for i in range(self.Nbasis): 
				Basis += self.basis_function[i](x1)*self.regression_param[i];

		L = cholesky( self.compute_Gramm_matrix(x1, x1), lower=True); 
		alpha = cho_solve((L, True), (y - Basis ) )
		return - np.array(0.5*( y - Basis ).T.dot(alpha) - np.log(np.diag(L)).sum()).flatten() - 0.5*len(x1)*np.log(2*math.pi);


	def cost_function(self, Theta):
		theta = Theta[0:len(self.kernel.theta)];
		if (len(theta) != len(Theta)): 
			regression_param = Theta[len(theta)::];


		b = np.copy(self.Training_values);
		K = self.kernel(self.Training_points);

		for i in range(self.Nbasis): 
			b -= self.basis[ i ]*regression_param[i];
			K += self.basis_v[i]*regression_param[i]**2;

		if type(self.noise_level) is float: K[np.diag_indices_from(K)] += self.noise_level;	

		L = cholesky(K, lower=True); 
		alpha = cho_solve((L, True), b )

		return np.array(0.5*b.T.dot(alpha) + np.log(np.diag(L)).sum()).flatten();


	def fit(self, Training_points, Training_values, noise_level):

		self.noise_level     = copy.deepcopy(noise_level);
		self.Training_points = copy.deepcopy(Training_points);
		self.Training_values = copy.deepcopy(Training_values);

		MIN = float("inf");
		bounds = self.kernel.bounds

		self.basis   = [];
		self.basis_v = [];
		for i in range(self.Nbasis): 
			a = self.basis_function[i](self.Training_points, return_variance=True);
			self.basis.append(   a[0] );
			self.basis_v.append( a[1] );
		self.regression_param = np.ones((np.shape(self.basis)[0], 1));
		for i in self.regression_param: 
			bounds = np.append(bounds, [[-10.0, 10.0]], axis=0)

		for int_it in range(10):
			InternalRandomGenerator = np.random.RandomState();
			x0 = InternalRandomGenerator.uniform(bounds[:, 0], bounds[:, 1]);
			res = sp.optimize.minimize(self.cost_function, x0, method="L-BFGS-B", bounds=bounds)
			#res = sp.optimize.minimize(self.cost_function, x0, method="L-BFGS-B", bounds=bounds, tol= 1e-09, options={'disp': None, 'maxcor': 10, 'ftol': 1e-09, 'maxiter': 15000})
			if (self.cost_function(res.x)[0] < MIN):
				MIN = self.cost_function(res.x)
				MIN_x = np.copy(res.x);


		self.kernel.theta     = np.copy(MIN_x[0:len(self.kernel.theta)]);
		self.regression_param = np.copy(MIN_x[len(self.kernel.theta)::]);

		b = np.copy(self.Training_values);
		for i in range(self.Nbasis): 
			b -= self.basis[i]*self.regression_param[i];

		self.L = cholesky(self.compute_Gramm_matrix(self.Training_points, self.Training_points), lower=True);
		self.alpha = cho_solve((self.L, True), b)



	def predict(self, x, return_variance= False):
		Basis = np.zeros((len(x), 1));
		Basis_v = np.zeros((len(x), len(x)));

		k = self.kernel(self.Training_points, x)

		if self.basis_function is not None: 
			for i in range(self.Nbasis): 
				a = self.basis_function[i](x, True)
				Basis   += a[0]*self.regression_param[i];
				Basis_v += a[1]*self.regression_param[i]**2;

		mean = Basis + k.T.dot(self.alpha);
		v = cho_solve((self.L, True), k);
		variance = Basis_v + self.kernel(x) - k.T.dot(v);

		if return_variance is True:
			return mean, variance;
		else:
			return mean;


	# def predict(self, x, y=None, return_variance= False):
	# 	if y is None: y = x.copy();
	# 	Basis   = np.zeros((len(x), 1));
	# 	Basis_v = np.zeros((len(x), len(y)));

	# 	k_l = self.kernel(x, self.Training_points)
	# 	k_r = self.kernel(self.Training_points, y)

	# 	for i in range(self.Nbasis): 
	# 		a = self.basis_function[i](x, y, True)
	# 		Basis   += a[0]*self.regression_param[i];
	# 		Basis_v += a[1]*self.regression_param[i]**2;

	# 	for i in range(self.Nbasis): 	
	# 		a = self.basis_function[i](x, self.Training_points, True);
	# 		k_l += np.array( a[1] )*self.regression_param[i]**2;
	# 		a = self.basis_function[i](self.Training_points, y, True);
	# 		k_r += np.array( a[1] )*self.regression_param[i]**2;

	# 	mean = Basis + k_l.dot( self.alpha )
		
	# 	v_r = cho_solve((self.L, True), k_r);
	# 	v_l = cho_solve((self.L.T, True), k_l.T);

	# 	#variance = Basis_v + self.kernel(x, y) - k_l.dot(L).dot(v);
	# 	#variance = Basis_v + self.kernel(x, y) - k_l.dot(sp.linalg.inv(self.kernel(self.Training_points))).dot(k_r);
	# 	#variance = Basis_v + self.kernel(x, y) - v_l.T.dot(v_r);
	# 	variance = Basis_v + self.kernel(x, y) - k_l.dot(v_r);

	# 	if return_variance is True:
	# 		return mean, variance;
	# 	else:
	# 		return mean;


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
		


