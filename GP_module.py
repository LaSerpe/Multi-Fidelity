import numpy as np
import scipy as sp
from scipy.linalg import cholesky, cho_solve, solve_triangular
import math
import copy

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import KernelDensity

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
			si2 = 1.0/inv_K[j, j];
			mu  = a[j]/si2;

			mll += np.log( si2 ) + mu**2/si2;
		return mll;


	def cost_function_LASSO(self, theta):
		b = np.copy(self.Training_values);
		for i in range(self.Nbasis): 
			b -= self.basis[ i ]*theta[i];
		return np.linalg.norm(b)**2/len(b);

	def opt_LASSO(self, LAMBDA):
		MIN = float("inf");
		con = ({'type': 'ineq', 'fun': lambda x: x}, {'type': 'ineq', 'fun': lambda x:  -np.linalg.norm(x) + LAMBDA})
		# con = ({'type': 'ineq', 'fun': lambda x:  -np.linalg.norm(x) + t_LASSO})
		for int_it in range(10):
			InternalRandomGenerator = np.random.RandomState();
			x0 = InternalRandomGenerator.uniform(size=(self.Nbasis, 1));
			res = sp.optimize.minimize(self.cost_function_LASSO, x0, method="SLSQP", constraints=con, tol= 1e-09, options={'disp': None, 'ftol': 1e-09, 'maxiter': 15000})
			if (self.cost_function_LASSO(res.x) < MIN):
				MIN   = self.cost_function_LASSO(res.x)
				MIN_x = np.copy(res.x);
		return MIN_x;



	def cost_function_LASSO_LOO(self, theta, exclude):
		b = np.copy(self.Training_values);
		for i in range(self.Nbasis): 
			b -= self.basis[ i ]*theta[i];

		eps = np.linalg.norm( np.delete( b, exclude, 0) )**2;

		return eps/( len(self.Training_values)-1 ) + self.LAMBDA*np.linalg.norm(theta);


	def opt_LASSO_LOO(self, LAMBDA):
		self.LAMBDA = LAMBDA;
		MIN = float("inf");
		con = ({'type': 'ineq', 'fun': lambda x: x})

		eps = 0.0;
		for i in range( len(self.Training_values) ):

			for int_it in range(10):
				InternalRandomGenerator = np.random.RandomState();
				x0 = InternalRandomGenerator.uniform(size=(self.Nbasis, 1));
				res = sp.optimize.minimize(self.cost_function_LASSO_LOO, x0, args=(i), method="SLSQP", constraints=con, tol= 1e-09, options={'disp': None, 'ftol': 1e-09, 'maxiter': 15000})
				if (self.cost_function_LASSO(res.x) < MIN):
					MIN   = self.cost_function_LASSO_LOO(res.x, i)
					MIN_x = np.copy(res.x);
					self.regression_param = np.copy(res.x);

			tmp = 0.0;
			for j in range(self.Nbasis): 
				tmp += (self.basis[ j ][i]*self.regression_param[j]);
			eps += (self.Training_values[i] - tmp )**2;


		return eps;




	def fit(self, Training_points, Training_values, Tychonov_regularization_coeff, Opt_Mode='MLL', LASSO= False):
# Mode Opt: 	MLL: Maximum Log Likelihood
# 			MLLW: Maximum Log Likelihood, weighted average on rhos
# 			MLLD: Maximum Log Likelihood, rhos are decoupled and computed before maximizing Likelihood
# 			MLLS: Maximum Log Likelihood, rhos are computed via substitution with weighted averaged linear regression problem

		self.Tychonov_regularization_coeff = copy.deepcopy(Tychonov_regularization_coeff);
		self.Training_points  = copy.deepcopy(Training_points);
		self.Training_values  = copy.deepcopy(Training_values);
		self.regression_param = np.ones((self.Nbasis, 1));
		self.Opt_Mode = Opt_Mode;
		self.LASSO = LASSO;


		if self.Opt_Mode == 'MLL_MC': self.MC_fit(); return;

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
			self.regression_param = cho_solve((  cholesky(tmp.T.dot(tmp) + self.Tychonov_regularization_coeff*np.eye(self.Nbasis), lower=True), True), tmp.T.dot(self.Training_values) );

			# if LASSO:
			# 	self.regression_param = self.opt_LASSO(np.linalg.norm(self.regression_param)/2.0);

			if LASSO:
				MIN = float("inf");
				con = ({'type': 'ineq', 'fun': lambda x: x})
				# con = ({'type': 'ineq', 'fun': lambda x:  -np.linalg.norm(x) + t_LASSO})
				for int_it in range(10):
					InternalRandomGenerator = np.random.RandomState();
					x0 = InternalRandomGenerator.uniform();
					res = sp.optimize.minimize(self.opt_LASSO_LOO, x0, method="SLSQP", constraints=con, tol= 1e-09, options={'disp': None, 'ftol': 1e-09, 'maxiter': 15000})
					if (self.opt_LASSO_LOO(res.x) < MIN):
						MIN   = self.opt_LASSO_LOO(res.x)
						MIN_x = np.copy(res.x);

				self.LAMBDA = MIN_x;
				self.opt_LASSO_LOO(self.LAMBDA);
				print('Penalty factor LAMBDA: ', self.LAMBDA)
				print('Min ERR: ', MIN)


			# if LASSO:
			# 	MIN = float("inf");
			# 	err_l = [];
			# 	Lam = np.linspace(0.1, 4.0, 100);
			# 	for lam in Lam:
			# 		err_l.append( self.opt_LASSO_LOO(lam) );
			# 		if (err_l[-1] < MIN):
			# 			MIN   = err_l[-1]
			# 			MIN_x = np.copy(lam);

			# 	self.LAMBDA = MIN_x;
			# 	self.opt_LASSO_LOO(self.LAMBDA);
			# 	print('Penalty factor LAMBDA: ', self.LAMBDA)
			# 	print('Min ERR: ', MIN)
			# 	plt.figure()
			# 	plt.plot(Lam, err_l)




		MIN = float("inf");
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



	def MC_fit(self):

		cost_function= self.cost_function_likelihood;
		

		bounds = self.kernel.bounds
		for i in self.regression_param: 
			bounds = np.append(bounds, [[-10.0, 10.0]], axis=0)
			

		self.basis   = [];
		self.basis_v = [];
		for i in range(self.Nbasis): 
			a = self.basis_function[i](self.Training_points, return_variance=True);
			self.basis.append(   a[0] );
			self.basis_v.append( a[1] );
		

		dim_sto = len(self.kernel.theta) + len(self.regression_param);
		InternalRandomGenerator = np.random.RandomState();

		# position_track = [] 
		# position_track.append( np.array(InternalRandomGenerator.uniform(0, 1, dim_sto)).T );
		# PI_track = [];
		# PI_track.append( - cost_function( [ bounds[k, 0] + (bounds[k, 1] - bounds[k, 0]) * position_track[-1][k]   for k in range(dim_sto)] )[0] );


		position_track = [] 
		position_track.append( np.array(InternalRandomGenerator.uniform(bounds[:, 0], bounds[:, 1])).T );
		PI_track = [];
		PI_track.append( - cost_function( position_track[-1] )[0] );


		Nburn = 20000;
		Nmcmc = 50000;
		alpha = 2.38**2/dim_sto;
		#alpha = 2.38**2/dim_sto;
		update_cov = 1000;
		give_info = 200000;

		#C = alpha*np.eye(dim_sto);
		C = alpha*np.diag(  np.array([ (bounds[k, 1] - bounds[k, 0])**2/12.0 for k in range(dim_sto)])  );
		L = cholesky(C, lower=True);
		mid_acceptance_rate = 0;

		for iOut in [Nburn, Nmcmc]:
			count = 0;

			if iOut == Nburn: 
				print('Burn-in phase...')
			if iOut == Nmcmc: 
				print('MCMC phase...')
				C = alpha*np.cov( np.array(position_track[-update_cov:]).T )+0.001*np.eye(dim_sto);
				L = cholesky(C, lower=True);


			for i in range(iOut):

				if( i % update_cov == 0 and iOut == Nburn and i != 0):
					C = alpha*np.cov( np.array(position_track[-update_cov:]).T )+0.001*np.eye(dim_sto);
					L = cholesky(C, lower=True);


				# if( i % give_info == 0 ):
				# 	print( float(mid_acceptance_rate) / give_info*100)
				# 	mid_acceptance_rate = 0;
	        
				#position = position_old[:] + 0.05*(InternalRandomGenerator.uniform(bounds[:, 0], bounds[:, 1])-position_old); 
				position = position_track[-1] + L.dot(np.array(InternalRandomGenerator.normal(0.0, 1.0, dim_sto)).T );


				#if any(k < 0.0 for k in position) or any(k > 1.0 for k in position):
				if any(position[k] < bounds[k, 0] for k in range(dim_sto)) or any(position[k] > bounds[k, 1] for k in range(dim_sto)):
					position_track.append(position_track[-1]);
					PI_track.append(PI_track[-1]);
					continue;
				else:
					#PI = - cost_function( [ bounds[k, 0] + (bounds[k, 1] - bounds[k, 0]) * position[k]   for k in range(dim_sto)] )[0];
					PI = - cost_function( position )[0];


				if( (PI - PI_track[-1]) >= np.log(InternalRandomGenerator.uniform()) ):
					position_track.append(position[:].T);
					PI_track.append(PI);
					count += 1;
					mid_acceptance_rate=mid_acceptance_rate+1;
				else:
					position_track.append(position_track[-1]);
					PI_track.append(PI_track[-1]);

			if iOut == Nburn: 
				print('BURN-IN acceptance rate: ' + str(float(count)/Nburn*100))
				pt = np.array(position_track[0:Nburn]);
				pi = np.array(PI_track[0:Nburn]);

			if iOut == Nmcmc:
				print('MCMC acceptance rate: ' + str(float(count)/Nmcmc*100))
				pt = np.array(position_track[Nburn::]);
				pi = np.array(PI_track[Nburn::]);


			fig, axs = plt.subplots(dim_sto, dim_sto, sharex=True, figsize=(10,10))#, gridspec_kw={'hspace':0.1})

			for iFig in range(dim_sto):
				for jFig in range(iFig+1, dim_sto):
					axs[iFig, jFig].scatter(pt[:, iFig], pt[:, jFig], s=2, c=pi);
					axs[iFig, jFig].axis([bounds[iFig, 0], bounds[iFig, 1], bounds[jFig, 0], bounds[jFig, 1]]);


				axs[iFig, iFig].hist(pt[:, iFig], 50, density=True, alpha=0.75)

				# kde_1 = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(pt[:, iFig].reshape(1, -1)).score_samples(np.linspace(0, 1, 100))
				# for jFig in range(0, i):
				# 	kde_2 = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(pt[:, jFig].reshape(1, -1)).score_samples(np.linspace(0, 1, 100))

				# 	axs[iFig, jFig].plot_surface(X, Y, pt[:][iFig]*pt[:][jFig], s=5);
				# 	#axs[iFig, jFig].axis([0, 1, 0, 1]);

			fig, axs = plt.subplots(dim_sto, sharex=True, figsize=(10,10))
			for iFig in range(dim_sto):
				axs[iFig].plot(pt[:, iFig])
			#axs[iFig].axis([0, len(pt), 0, 1]);


# fig.tight_layout()
# plt.savefig('../PAPER/figure/analysis/EPM_compare_diff.pdf')


		mll_index = np.argmax(PI_track[Nburn::]);
		#tmp = [ bounds[k, 0] + (bounds[k, 1] - bounds[k, 0]) * position_track[mll_index][k]   for k in range(dim_sto)]
		tmp = position_track[mll_index];


		self.kernel.theta     = np.copy(tmp[0:len(self.kernel.theta)]);
		self.regression_param = np.copy(tmp[len(self.kernel.theta)::]);

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
		


