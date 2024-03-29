import numpy as np
import math

x_min = 0.0;
x_max = 1.0;

#sn_1 =  0.0; sn_2 =  0.0; sn_3 =  0.0; sn_4 =  0.0; sn_5 =  0.0; sn_6 =  0.0; sn_7 =  0.0; sn_8 =  0.0;
#sn_1 = 1e-4; sn_2 = 1e-4; sn_3 = 1e-4; sn_4 = 1e-4; sn_5 = 1e-4; sn_6 = 1e-4; sn_7 = 1e-4; sn_8 = 1e-4;
#sn_1 = 1e-2; sn_2 = 1e-2; sn_3 = 1e-2; sn_4 = 1e-2; sn_5 = 1e-2; sn_6 = 1e-2; sn_7 = 1e-2; sn_8 = 1e-2;

sn_1 = 1e-12; sn_2 = 1e-12; sn_3 = 1e-12; sn_4 = 1e-12; sn_5 = 1e-12; sn_6 = 1e-12; sn_7 = 1e-12; sn_8 = 1e-12; sn_9 = 1e-12;

def model_1(x):
	return [x + sn_1*np.random.normal(0.0, 1.0), sn_1];

def model_2(x):
	return [0.7*np.sin(x/x_max*math.pi*8)*x + sn_2*np.random.normal(0.0, 1.0), sn_2];

def model_3(x):
	return [-5.0*x + 1.0 + sn_3*np.random.normal(0.0, 1.0), sn_3];

def model_4(x):
	return [np.sin(x/x_max*math.pi*8)*x + x + sn_4*np.random.normal(0.0, 1.0), sn_4];

def model_5(x):
	return [-5.0*x**2 + sn_5*np.random.normal(0.0, 1.0), sn_5];

def model_6(x):
	return [np.sin(x/x_max*math.pi*8.2)*x + x + sn_6*np.random.normal(0.0, 1.0), sn_6];

def model_7(x):
	return [1.05*x + 0.1 + sn_7*np.random.normal(0.0, 1.0), sn_7];

def model_8(x):
	return [0.7*np.sin(x/x_max*math.pi*7.5)*x + sn_8*np.random.normal(0.0, 1.0), sn_8];

def model_9(x):
	return [np.sin(x/x_max*math.pi*8.5)*x + x + sn_9*np.random.normal(0.0, 1.0), sn_9];



# freq = 4.0;
# def model_9(x):
# 	return [ 2.0*np.sin(x/x_max*math.pi* ( freq) +0.05) + 0.05*x, 2e-8*np.random.normal(0.0, 1.0)];

# def model_10(x):
# 	return [-2.0*np.sin(x/x_max*math.pi* ( freq) -0.05)  + 0.05*x, 2e-8*np.random.normal(0.0, 1.0)];

# def model_11(x):
# 	return [np.sin(x/x_max*math.pi* ( 2*freq) +0.05)  + 0.05*x, 2e-8*np.random.normal(0.0, 1.0)];

# def model_12(x):
# 	return [np.sin(x/x_max*math.pi* ( 2*freq)), 2e-8*np.random.normal(0.0, 1.0)];




def model_Sacher_1(x):
	return [0.5*(6*x-2)**2*np.sin(12*x-4) + 10*(x-1), 2e-8*np.random.normal(0.0, 1.0)];

def model_Sacher_2(x):
	return [ 2*model_Sacher_1(x)[0] - 20*(x-1), 2e-8*np.random.normal(0.0, 1.0)];





# Hartmann 1-D

alpha = [1.0, 1.2, 3.0, 3.2];
P = 1e-4*np.array([[1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0], [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0], [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0], [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]]);
A = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0], [0.05, 10.0, 17.0, 0.1, 8.0, 14.0], [3.0, 3.5, 1.7, 10.0, 17.0, 8.0], [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]]);

def Hartmann(x):
	f = 0.0;
	for i in range(4):
		tmp = 0.0;
		for j in range(len(x)):
			tmp -= A[i][j]*(x[j] - P[i][j])**2;

		f += alpha[i]*np.exp(tmp)
	return f;

u0= -5.0;
delta = 0.1;
def U_1(x):
	xd = x + delta/1.0*np.ones((np.shape(x)));
	return [0.5* (Hartmann( xd )**2/u0 + u0), 2e-8*np.random.normal(0.0, 1.0)];

def U_2(x):
	xd = x + delta/2.0*np.ones((np.shape(x)));
	y= U_1( xd )[0];
	y = 0.5* (Hartmann( xd )**2/y + y );
	return [y, 2e-8*np.random.normal(0.0, 1.0)];

def U_3(x):
	xd = x + delta/3.0*np.ones((np.shape(x)));
	y= U_2( xd )[0];
	y = 0.5* (Hartmann( xd )**2/y + y );
	return [y, 2e-8*np.random.normal(0.0, 1.0)];

def U_4(x):
	xd = x + delta/4.0*np.ones((np.shape(x)));
	y= U_3( xd )[0];
	y = 0.5* (Hartmann( xd )**2/y + y );
	return [y, 2e-8*np.random.normal(0.0, 1.0)];

def U_5(x):
	xd = x + delta/5.0*np.ones((np.shape(x)));
	y= U_4( xd )[0];
	y = 0.5* (Hartmann( xd )**2/y + y );
	return [y, 2e-8*np.random.normal(0.0, 1.0)];

def U_6(x):
	xd = x + delta/6.0*np.ones((np.shape(x)));
	y= U_5( xd )[0];
	y = 0.5* (Hartmann( xd )**2/y + y );
	return [y, 2e-8*np.random.normal(0.0, 1.0)];

def U_7(x):
	xd = x + delta/7.0*np.ones((np.shape(x)));
	y= U_6( xd )[0];
	y = 0.5* (Hartmann( xd )**2/y + y );
	return [y, 2e-8*np.random.normal(0.0, 1.0)];


# class U_models:
# 	def __init__(self, u0, levels= 1):
# 		self.levels = levels;
# 		self.u0 = u0;

# 	def U_generate_functions(self):
# 		F = [];
# 		F.append(0.5* (Hartmann/self.u0 + self.u0 ) );
# 		for i in range(self.levels-1):
# 			F.append(0.5* (Hartmann**2/F[-1] + F[-1] ) );

# 		return F;





def P0(x):
	return [x**0, 2e-8*np.random.normal(0.0, 1.0)];

def P1(x):
	return [x**1, 2e-8*np.random.normal(0.0, 1.0)];

def P2(x):
	return [x**2, 2e-8*np.random.normal(0.0, 1.0)];

def P3(x):
	return [x**3, 2e-8*np.random.normal(0.0, 1.0)];

def P4(x):
	return [x**4, 2e-8*np.random.normal(0.0, 1.0)];

def P5(x):
	return [x**5, 2e-8*np.random.normal(0.0, 1.0)];

def P6(x):
	return [x**6, 2e-8*np.random.normal(0.0, 1.0)];










def Pm1(x):
	return [ x**0 + x**1 + 3, 2e-8*np.random.normal(0.0, 1.0)];

def Pm2(x):
	return [-x**3 + x**5, 2e-8*np.random.normal(0.0, 1.0)];

def Pm3(x):
	return [ x - x**3 + 3, 2e-8*np.random.normal(0.0, 1.0)];

def Pm4(x):
	return [ x**1 + x**5, 2e-8*np.random.normal(0.0, 1.0)];


# def Pm1(x):
# 	return [ x**0 + x**1 - x**3 + x**5 + (x>0.5)*(x**2 + 5), 2e-8*np.random.normal(0.0, 1.0)];

# def Pm2(x):
# 	return [ x**0 + x**1 - x**3 + x**5 - (x>0.5)*(x**2 + 1), 2e-8*np.random.normal(0.0, 1.0)];

# def Pm3(x):
# 	return [ x**2 + 2, 2e-8*np.random.normal(0.0, 1.0)];


def PT(x):
	return [x**0 + x**1 - x**3 + x**5 + 3, 1e-12*np.random.normal(0.0, 1.0)];








