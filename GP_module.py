import numpy as np
import scipy as sp
from scipy.linalg import cholesky, cho_solve, solve_triangular
import math
import copy

#from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct, ConstantKernel, WhiteKernel



class GP:
	def __init__(self, Kernel, Basis= None, mode='G'):
		self.kernel = copy.deepcopy(Kernel);
		if mode in['G', 'S', 'O']:
			self.mode= mode;
		else:
			print("Error! Invalid GP mode");
			exit();
		if Basis is None: 
			self.basis_function = None;
			self.Nbasis= 0;
		else:
			self.basis_function = copy.deepcopy(Basis);
			self.Nbasis= len(Basis);

	def compute_Gramm_matrix(self, x1, x2):
		K = self.kernel(x1, x2);
		if self.mode != 'S':
			for i in range(self.Nbasis): 
				K += self.basis_function[i](x1, x2, return_variance=True)[1]*self.regression_param[i]**2;
		if type(self.Tychonov_regularization_coeff) is float: K[np.diag_indices_from(K)] += self.Tychonov_regularization_coeff;
		return K;


	def compute_loglikelihood(self, x1, y): #to check
		Basis = np.zeros((len(x1), 1));
		if self.basis_function is not None: 
			for i in range(self.Nbasis): 
				Basis += self.basis_function[i](x1)*self.regression_param[i];

		L = cholesky( self.compute_Gramm_matrix(x1, x1), lower=True); 
		alpha = cho_solve((L, True), (y - Basis ) )
		return - np.array(0.5*( y - Basis ).T.dot(alpha) - np.log(np.diag(L)).sum()).flatten() - 0.5*len(x1)*np.log(2*math.pi);



	def cost_function_likelihood(self, theta):
		self.kernel.theta = theta[0:len(self.kernel.theta)];
		if (len(self.kernel.theta) != len(theta)): 
			self.regression_param = np.array(theta[len(self.kernel.theta)::]);
			if self.Opt_Mode == 'MLLW': self.regression_param /= np.sum(np.absolute(self.regression_param)) + 1e-16;


		b = np.copy(self.Training_values);
		K = self.kernel(self.Training_points);

		if self.Opt_Mode == 'MLLS' and self.Nbasis != 0:
			tmp = np.array([x.flatten() for x in self.basis]).T;
			self.regression_param = cho_solve((  cholesky(tmp.T.dot(np.linalg.inv(K)).dot(tmp), lower=True), True), tmp.T.dot(np.linalg.inv(K)).dot(self.Training_values) );
			#print(self.regression_param)

		for i in range(self.Nbasis): 
			b -= self.basis[ i ]*self.regression_param[i];
			if self.mode != 'S':
				K += self.basis_v[i]*self.regression_param[i]**2;

		if type(self.Tychonov_regularization_coeff) is float: K[np.diag_indices_from(K)] += self.Tychonov_regularization_coeff;	

		L = cholesky(K, lower=True); 
		alpha = cho_solve((L, True), b )

		return np.array(0.5*b.T.dot(alpha) + np.log(np.diag(L)).sum()).flatten();



	# def cost_function_LOO(self, theta):
	# 	self.kernel.theta = theta[0:len(self.kernel.theta)];
	# 	if (len(self.kernel.theta) != len(theta)): 
	# 		self.regression_param = np.array(theta[len(self.kernel.theta)::]);

	# 	b = np.copy(self.Training_values);
	# 	K = self.kernel(self.Training_points);

	# 	for i in range(self.Nbasis): 
	# 		b -= self.basis[ i ]*self.regression_param[i];
	# 		if self.mode != 'S':
	# 			K += self.basis_v[i]*self.regression_param[i]**2;

	# 	if type(self.Tychonov_regularization_coeff) is float: K[np.diag_indices_from(K)] += self.Tychonov_regularization_coeff;	

	# 	L = cholesky(K, lower=True);
	# 	a = cho_solve((L, True), b )
		
	# 	k = self.kernel( self.Training_values, self.Training_values );
	# 	if self.mode == 'G':
	# 		for i in range(self.Nbasis): 	
	# 			k += self.k_tmp[i]*self.regression_param[i]**2;

	# 	elem_pert = np.eye(len(self.Training_points));
	# 	eps = 0.0;

	# 	for j in range( len(self.Training_points) ): 
	# 		da = cho_solve((L, True), elem_pert[:, j] ).reshape(-1, 1)
	# 		beta = a[j]/da[j];
	# 		#eps += ( b[j, 0] -  beta - k[:, j].dot( a - beta*da )  )[0]**2;
	# 		eps += ( b[j, 0] - k[:, j].dot( a - beta*da )  )[0]**2;
	# 		eps += np.linalg.norm(k);
	# 		#eps += np.linalg.norm(k)**2;

	# 	return eps;



	def cost_function_LOO(self, theta):
		self.kernel.theta = theta[0:len(self.kernel.theta)];
		if (len(self.kernel.theta) != len(theta)): 
			self.regression_param = np.array(theta[len(self.kernel.theta)::]);

		b = np.copy(self.Training_values);
		K = self.kernel(self.Training_points);

		for i in range(self.Nbasis): 
			b -= self.basis[ i ]*self.regression_param[i];
			if self.mode != 'S':
				K += self.basis_v[i]*self.regression_param[i]**2;

		if type(self.Tychonov_regularization_coeff) is float: K[np.diag_indices_from(K)] += self.Tychonov_regularization_coeff;	

		L = cholesky(K, lower=True);
		a = cho_solve((L, True), b )
		inv_K = np.linalg.inv(K);
		
		mll = 0.0;
		for j in range( len(self.Training_points) ): 

			#inv_Kii = np.delete(np.delete(np.linalg.inv(K), j, 0), j, 1);
			
			si2 = 1.0/inv_K[j, j];
			mu  = a[j]/si2;

			mll += np.log( si2 ) + mu**2/si2;

		# print(a)
		# print(mll)
		return mll;




	def fit(self, Training_points, Training_values, Tychonov_regularization_coeff, Opt_Mode='MLL'):
# Mode Opt: 	MLL: Maximum Log Likelihood
# 			MLLW: Maximum Log Likelihood, weighted average on rhos
# 			MLLD: Maximum Log Likelihood, rhos are decoupled and computed before maximizing Likelihood
# 			MLLS: Maximum Log Likelihood, rhos are computed via substitution with weighted averaged linear regression problem

		self.Tychonov_regularization_coeff = copy.deepcopy(Tychonov_regularization_coeff);
		self.Training_points  = copy.deepcopy(Training_points);
		self.Training_values  = copy.deepcopy(Training_values);
		self.regression_param = np.ones((self.Nbasis, 1));
		self.Opt_Mode = Opt_Mode

		if self.Opt_Mode == 'MLL' or Opt_Mode == 'MLLW' or Opt_Mode == 'MLLD' or Opt_Mode == 'MLLS':
			cost_function= self.cost_function_likelihood;
		elif self.Opt_Mode == 'LOO':
			cost_function= self.cost_function_LOO;
			if self.mode == 'G':
				self.k_tmp = [];
				for i in range(self.Nbasis):
					self.k_tmp.append( self.basis_function[i](self.Training_values, self.Training_points, True)[1] )
		else:
			print("Error! Optimization mode");
			exit();

		MIN = float("inf");
		bounds = self.kernel.bounds
		for i in self.regression_param: 
			if Opt_Mode == 'MLLW':
				bounds = np.append(bounds, [[0.0, 1.0]], axis=0)
			elif Opt_Mode != 'MLLD' and Opt_Mode != 'MLLS':
				bounds = np.append(bounds, [[-10.0, 10.0]], axis=0)
			

		self.basis   = [];
		self.basis_v = [];
		for i in range(self.Nbasis): 
			a = self.basis_function[i](self.Training_points, return_variance=True);
			self.basis.append(   a[0] );
			self.basis_v.append( a[1] );

		if Opt_Mode == 'MLLD' and self.Nbasis != 0:
			tmp = np.array([x.flatten() for x in self.basis]).T;
			self.regression_param = cho_solve((  cholesky(tmp.T.dot(tmp), lower=True), True), tmp.T.dot(self.Training_values) );
			print(self.regression_param)
			# self.regression_param = np.linalg.inv( tmp.T.dot(tmp) ).dot(tmp.T).dot(self.Training_values);
			# print(self.regression_param)


		for int_it in range(10):
			InternalRandomGenerator = np.random.RandomState();
			x0 = InternalRandomGenerator.uniform(bounds[:, 0], bounds[:, 1]);
			#res = sp.optimize.minimize(cost_function, x0, method="L-BFGS-B", bounds=bounds)
			res = sp.optimize.minimize(cost_function, x0, method="L-BFGS-B", bounds=bounds, tol= 1e-09, options={'disp': None, 'maxcor': 10, 'ftol': 1e-09, 'maxiter': 15000})
			if (cost_function(res.x) < MIN):
				MIN   = cost_function(res.x)
				MIN_x = np.copy(res.x);

		self.kernel.theta     = np.copy(MIN_x[0:len(self.kernel.theta)]);
		if self.Opt_Mode != 'MLLD' and self.Opt_Mode != 'MLLS':
			self.regression_param = np.copy(MIN_x[len(self.kernel.theta)::]);
			if self.Opt_Mode == 'MLLW':
				self.regression_param /= np.sum(np.absolute(self.regression_param)) + 1e-16;

		b = np.copy(self.Training_values);
		for i in range(self.Nbasis): 
			b -= self.basis[i]*self.regression_param[i];

		K = self.kernel(self.Training_points);
		if self.mode != 'S':
			for i in range(self.Nbasis): 
				K += self.basis_v[i]*self.regression_param[i]**2;
		if type(self.Tychonov_regularization_coeff) is float:
			K[np.diag_indices_from(K)] += self.Tychonov_regularization_coeff;
		self.L     = cholesky(K, lower=True);

		#self.L     = cholesky(self.compute_Gramm_matrix(self.Training_points, self.Training_points), lower=True);
		self.alpha = cho_solve((self.L, True), b)





	def predict(self, x1, x2=None, return_variance= False):
		if x2 is None: x2 = x1.copy();
		Basis   = np.zeros((len(x1), 1));
		Basis_v = np.zeros((len(x1), len(x2)));

		k_l = self.kernel(x1, self.Training_points)
		k_r = self.kernel(self.Training_points, x2)

		for i in range(self.Nbasis): 
			a = self.basis_function[i](x1, x2, True)
			Basis   += a[0]*self.regression_param[i];
			Basis_v += a[1]*self.regression_param[i]**2;

		if self.mode == 'G':
			for i in range(self.Nbasis): 	
				k_l += np.array( self.basis_function[i](x1, self.Training_points, True)[1] )*self.regression_param[i]**2;
				k_r += np.array( self.basis_function[i](self.Training_points, x2, True)[1] )*self.regression_param[i]**2;

		mean = Basis + k_l.dot( np.array(self.alpha) )
		
		if return_variance is True:
			v_r = cho_solve((self.L,   True), k_r  );
			if np.array_equal(x1, x2):
				variance = Basis_v + self.kernel(x1) - k_l.dot(v_r);
			else:
				variance = Basis_v + self.kernel(x1, x2) - k_l.dot(v_r);
			return mean, variance;
		else:
			return mean;


	def score(self, x, y, sample_weight=None):
		return 1 - ( ( y - self.predict(x, return_variance= False) )**2 ).sum() / ((y - y.mean())**2).sum()


	def Qcriteria(self, x, y, eps=0.00):
		y_gp, sigma_gp = self.predict(x, return_variance= True);
		y_gp = y_gp.reshape(-1, 1);
		sigma_gp = np.sqrt(np.diag(sigma_gp)).reshape(-1, 1)

		return np.absolute( 0.5*sp.special.erf( np.divide((y - y_gp) , (np.sqrt(2)*(sigma_gp+eps) )) ) ) * ( np.absolute(y - y_gp) * sigma_gp)


	def L2normCreteria(self, x, y, eps=0.00):
		y_gp= self.predict(x);
		y_gp = y_gp.reshape(-1, 1);

		return np.absolute( y - y_gp ).sum();
		


