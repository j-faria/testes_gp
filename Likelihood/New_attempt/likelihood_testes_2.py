# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:26:46 2016

@author: camacho
"""
import numpy as np
import Kernel;reload(Kernel);kl = Kernel
from time import time
from matplotlib import pyplot as pl
from  sympy import diff

import george
from george.kernels import *

#####  DADOS INICIAS  #########################################################
#np.random.seed(seed=10901)
x = 10 * np.sort(np.random.rand(100))
yerr = 0.2 * np.ones_like(x)
y = np.sin(5*x) + yerr * np.random.randn(len(x))

###############################################################################
ker=kl.ExpSineSquared(1.0,1.0,1.0)
##### Lonely kernel #####
#EXEMPLO 1 - ExpSquared
x1 = 10 * np.sort(np.random.rand(101))
yerr1 = 0.2 * np.ones_like(x1)
y1 = np.sin(x1) + yerr1 * np.random.randn(len(x1))

print('-> lonely kernel')
kernel0=kl.ExpSquared(19.0, 2.0)
kl.likelihood(kernel0, x1, x1, y1, yerr1)

#start = time() # Calculation using george 
#kernel = 19**2*ExpSquaredKernel(2.0**2)
#gp = george.GP(kernel)
#gp.compute(x1,yerr1)
#print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y1))

print 'variaveis ->', kl.variables(kernel0)
print 'numero de variaveis ->', kl.variablesLen(kernel0)
variavel=kl.variables(kernel0) #deveolve 19.0 e 2.0
#print kl.grad_log_p(kernel0,x1,x1,y1,yerr1) #devolve None
#print  kl.grad_lnlikelihood(kernel0,y1)

import numpy, algopy

#cg = algopy.CGraph()
#
#x = x1
#y = kernel0
#
#cg.independentFunctionList = [x]
#cg.dependentFunctionList = [y]
#print cg.gradient(numpy.array(x1))

x = UTPM.init_jacobian(variavel)
y = kernel0
algopy_jacobian = UTPM.extract_jacobian(y)
print('jacobian = ',algopy_jacobian)





###### Sum of kernels #####
##EXEMPLO 2
#print('-> sum of kernels')
#kernel1 = kl.ExpSquared(1.0,1.0) + kl.ExpSineSquared(1.0,1.0,1.0) + WhiteNoise(1.0)
#kl.likelihood(kernel1, x, x, y, yerr)
#
#start = time() # Calculation using george 
#kernel = 1.**2*ExpSquaredKernel(1.0**2) + ExpSine2Kernel(2.0/1.0**2, 1.0)
#gp = george.GP(kernel)
#gp.compute(x,yerr)
#print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))
#
#print 'variaveis ->', kl.variables(kernel1)