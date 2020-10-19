import numpy as np
import math

x_min = 0.0;
x_max = 1.0;

def truth(x):
	return np.sin(x/x_max*math.pi*8)*x + x;

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





