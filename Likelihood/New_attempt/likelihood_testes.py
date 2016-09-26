# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:56:24 2016

@author: camacho
"""
import numpy as np
import Kernel;reload(Kernel);kl = Kernel
from time import time
from matplotlib import pyplot as pl

import george
from george.kernels import *

#####  DADOS INICIAS  #########################################################
#np.random.seed(seed=10901)
x = 10 * np.sort(np.random.rand(100))
yerr = 0.2 * np.ones_like(x)
y = np.sin(5*x) + yerr * np.random.randn(len(x))
#pl.plot(x,y)

###############################################################################
#likelihood(kernels, x dado, x a calcular, y, yerr)

#   Os parametros foram dados às três pancadas só mesmo para ver se o python faz
#os calculos sem dar erro, por isso a likelihood deverá dar valores estranhos.

##### Lonely kernel #####
#EXEMPLO 1 - ExpSquared
x1 = 10 * np.sort(np.random.rand(101))
yerr1 = 0.2 * np.ones_like(x1)
y1 = np.sin(x1) + yerr1 * np.random.randn(len(x1))

print('-> lonely kernel')
kl.likelihood(kl.ExpSquared(19.0, 2.0), x1, x1, y1, yerr1)
start = time() # Calculation using george 
kernel = 19**2*ExpSquaredKernel(2.0**2)
gp = george.GP(kernel)
gp.compute(x1,yerr1)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y1))

#EXEMPLO 2 - ExpSineSquared
x2 = 10 * np.sort(np.random.rand(102))
yerr2 = 0.2 * np.ones_like(x2)
y2 = np.sin(5*x2) + yerr2 * np.random.randn(len(x2))

print('-> lonely kernel')
kl.likelihood(kl.ExpSineSquared(1.0, 2.0, 2.0), x2, x2, y2, yerr2)
start = time() # Calculation using george 
kernel = 1**2*ExpSine2Kernel(2.0/2.0**2,2.0)
gp = george.GP(kernel)
gp.compute(x2,yerr2)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y2))

#### Sum of kernels #####
#EXEMPLO 5
print('-> sum of kernels')
k1 = kl.ExpSquared(1.0,1.0) + kl.ExpSineSquared(1.,1.,1.) + kl.WhiteNoise(1.0)
#k2 = kl.Sum(kl.ExpSquared(1.0,1.0),kl.ExpSineSquared(1.,1.,1.))
covk1 = kl.likelihood(k1, x, x, y, yerr)
#covk2 = kl.likelihood(k2, x, x, y, yerr)

start = time() # Calculation using george 
kernel = 1.**2*ExpSquaredKernel(1.0**2) + ExpSine2Kernel(2/1., 1.) + WhiteKernel(1.0)
gp = george.GP(kernel)
gp.compute(x,yerr)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))

print('-> sum of kernels')
k1 = kl.ExpSquared(2.0,1.0) + kl.ExpSineSquared(3.0,5.0,1.0)
#k2 = kl.Sum(kl.ExpSquared(1.0,1.0),kl.ExpSineSquared(1.,1.,1.))
covk1 = kl.likelihood(k1, x, x, y, yerr)
#covk2 = kl.likelihood(k2, x, x, y, yerr)

start = time() # Calculation using george 
kernel = 2.**2*ExpSquaredKernel(1.0**2) + 3.**2*ExpSine2Kernel(2.0/5.0**2, 1.0) 
gp = george.GP(kernel)
gp.compute(x,yerr)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))


##### Multiplication  of kernels #####
#EXEMPLO 4 - Local_ExpSineSquared
x3 = 10 * np.sort(np.random.rand(103))
yerr3 = 0.2 * np.ones_like(x3)
y3 = np.sin(5*x3) + yerr3 * np.random.randn(len(x3))

print('-> multiplication of kernels')
#kl.likelihood(kl.Local_ExpSineSquared(1.0, 2.0, 2.0), x3, x3, y3, yerr3)
a=kl.ExpSquared(1.0,2.0)*kl.ExpSineSquared(1.0,2.0,2.0)*kl.WhiteNoise(1.0)
kl.likelihood(a, x3,x3,y3,yerr3)

start = time() # Calculation using george 
kernel = 1**2*ExpSine2Kernel(2.0/2.0**2,2.0)*ExpSquaredKernel(2.0**2)*WhiteKernel(1.0)
gp = george.GP(kernel)
gp.compute(x3,yerr3)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y3))

print('-> multiplication of kernels')
a=kl.ExpSquared(1.0,1.0)*kl.ExpSineSquared(1.0,1.0,1.0)
kl.likelihood(a, x3, x3, y3, yerr3)
#kl.likelihood(kl.Local_ExpSineSquared(1.0,1.0,1.0), x, x, y, yerr)
#   A Local_ExpSineSquared = ExpSquared * ExpSineSquared, logo se a
#multiplicação estiver a ser bem feita acho as duas devem dar os mesmos valores

start = time() # Calculation using george 
kernel = ExpSquaredKernel(1.0**2) * ExpSine2Kernel(2.0/1.0, 1.0)
gp = george.GP(kernel)
gp.compute(x3,yerr3)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y3))


##Multiplication  and sum of kernels
#print('-> multiplication and sum of kernels')
##multiplicar com white noise incluido
#b=kl.ExpSquared(1.0,1.0)*kl.ExpSineSquared(1.,1.,1.) +kl.WhiteNoise(1.)
#kl.likelihood(b, x, x, y, yerr)
#
#start = time() # Calculation using george 
#kernel = 1.**2*ExpSquaredKernel(1.0**2) * ExpSine2Kernel(1., 1.) +  WhiteKernel(1.)
#gp = george.GP(kernel)
#gp.compute(x,yerr)
#print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))
