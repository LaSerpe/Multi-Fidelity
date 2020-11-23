import numpy as np
import math

x_min = 0.0;
x_max = 1.0;


def model_1(x):
	return [x, 1e-8*np.random.normal(0.0, 1.0)];

def model_2(x):
	return [0.7*(np.sin(x/x_max*math.pi*8)*x + 0.0), 1e-8*np.random.normal(0.0, 1.0)];

def model_3(x):
	return [-5.0*x + 1.0, 2e-8*np.random.normal(0.0, 1.0)];

def model_4(x):
	return [np.sin(x/x_max*math.pi*8)*x + x, 2e-8*np.random.normal(0.0, 1.0)];

def model_5(x):
	return [-5.0*x**2, 2e-8*np.random.normal(0.0, 1.0)];

def model_6(x):
	return [np.sin(x/x_max*math.pi*8.2)*x + x, 2e-8*np.random.normal(0.0, 1.0)];

def model_7(x):
	return [np.sin(x/x_max*math.pi*8.4)*x + x, 2e-8*np.random.normal(0.0, 1.0)];

def model_8(x):
	return [np.sin(x/x_max*math.pi*8.6)*x + x, 2e-8*np.random.normal(0.0, 1.0)];



freq = 4.0;
def model_9(x):
	return [2.0*np.sin(x/x_max*math.pi* ( 1*freq) ) + 1.0, 2e-8*np.random.normal(0.0, 1.0)];

# def model_9(x):
# 	return [ np.sin(x/x_max*math.pi*freq*2)*x**2 , 2e-8*np.random.normal(0.0, 1.0)];

def model_10(x):
	return [2.0*np.sin(x/x_max*math.pi* (-1*freq) )  + 1.0, 2e-8*np.random.normal(0.0, 1.0)];

def model_11(x):
	return [np.sin(x/x_max*math.pi* (-1*freq))  + 1.0, 2e-8*np.random.normal(0.0, 1.0)];

def model_12(x):
	return [np.sin(x/x_max*math.pi* ( 1*freq))  + 1.0, 2e-8*np.random.normal(0.0, 1.0)];




def model_Sacher_1(x):
	return [0.5*(6*x-2)**2*np.sin(12*x-4) + 10*(x-1), 2e-8*np.random.normal(0.0, 1.0)];

def model_Sacher_2(x):
	return [ 2*model_Sacher_1(x)[0] - 20*(x-1), 2e-8*np.random.normal(0.0, 1.0)];



