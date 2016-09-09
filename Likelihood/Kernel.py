# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:42:48 2016

@author: camacho
"""
import numpy as np

# Squared Exponential Kernel
def ExpSquared(theta, l, x1, x2):
    f1 = theta**2
    f2 = l**2
    f3 = (x1-x2)**2
    return f1*np.exp(-0.5*f3/f2)

# Rational Quadratic Kernel
def RatQuadratic(theta, l, alpha, x1, x2):
    f1 = theta**2
    f2 = l**2
    f3 = (x1-x2)**2
    return f1*(1+(0.5*f3/(alpha*f2)))**(-alpha)

# Periodic Kernel    
def ExpSineSquared(theta, l, P, x1, x2): 
    f1 = theta**2
    f2 = l**2
    f3 = (x1-x2)
    return f1*np.exp(-2*(np.sin(np.pi*f3/P))**2/f2)

# Locally Periodic Kernel
def Local_ExpSineSquared(theta, l, P, x1, x2): 
    f1 = theta**2
    f2 = l**2
    f3 = (x1-x2)
    f4 = (x1-x2)**2
    return f1*np.exp(-2*(np.sin(np.pi*f3/P))**2/f2)*np.exp(-0.5*f4/f2)

# Linear Kernel
def Linear(thetab,thetav,c,x1, x2):
    f1 = thetab**2
    f2 = thetav**2
    return f1+f2*(x1-c)*(x2-c)