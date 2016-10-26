# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:42:48 2016

@author: camacho
"""
import numpy as np

# Squared Exponential Kernel
def ExpSquared(x1, x2, ES_theta, ES_l):
    f1 = ES_theta**2
    f2 = ES_l**2
    f3 = (np.array(x1)-np.array(x2))**2
    return f1*np.exp(-0.5*f3/f2)
   
# Rational Quadratic Kernel
def RatQuadratic(x1, x2, RQ_theta, RQ_l, RQ_alpha):
    f1 = RQ_theta**2
    f2 = RQ_l**2
    f3 = (x1-x2)**2
    return f1*(1+(0.5*f3/(RQ_alpha*f2)))**(-RQ_alpha)

# Periodic Kernel    
def ExpSineSquared(x1, x2, ESS_theta, ESS_l, ESS_P): 
    f1 = ESS_theta**2
    f2 = ESS_l**2
    f3 = (x1-x2)
    return f1*np.exp(-2*(np.sin(np.pi*f3/ESS_P))**2/f2)

# Locally Periodic Kernel (identico a fazer ExpSineSquared*ExpSquared)
def Local_ExpSineSquared(x1, x2, LESS_theta, LESS_l, LESS_P): 
    f1 = LESS_theta**2
    f2 = LESS_l**2
    f3 = (x1-x2)
    f4 = (x1-x2)**2
    return f1*np.exp(-2*(np.sin(np.pi*f3/LESS_P))**2/f2)*np.exp(-0.5*f4/f2)

## Linear Kernel
#def Linear(x1, x2,L_thetab,L_thetav,L_c):
#    f1 = L_thetab**2
#    f2 = L_thetav**2
#    return f1+f2*(x1-L_c)*(x2-L_c)
    
#Soma de Periodic com Squared Exponential (ExpSineSquared+ExpSquared)
def Sum_ExpSineSquared_ExpSquared(x1, x2, ESS_theta, ESS_l, ESS_P, ES_theta, ES_l):
    return ExpSineSquared(x1,x2,ESS_theta,ESS_l,ESS_P)+ExpSquared(x1,x2,ES_theta,ES_l)
    
# Squared Exponential Kernel com white noise
def ExpSquared_WN(x1, x2, ES_theta, ES_l, WN):
    f1 = ES_theta**2
    f2 = ES_l**2
    f3 = (x1-x2)**2
    return f1*np.exp(-0.5*f3/f2)
    
# Periodic Kernel com white noise
def ExpSineSquared_WN(x1, x2, ESS_theta, ESS_l, ESS_P,WN): 
    f1 = ESS_theta**2
    f2 = ESS_l**2
    f3 = (x1-x2)
    return f1*np.exp(-2*(np.sin(np.pi*f3/ESS_P))**2/f2)
