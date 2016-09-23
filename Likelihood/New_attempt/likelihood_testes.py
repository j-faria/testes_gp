# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:56:24 2016

@author: camacho
"""
import numpy as np
import Kernel
reload(Kernel)
kl = Kernel
from time import time

import george
from george.kernels import *

#####  DADOS INICIAS  #########################################################
x = 10 * np.sort(np.random.rand(100))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

###############################################################################
#likelihood(kernels, x dado, x a calcular, y, yerr)

#   Os parametros foram dados às três pancadas só mesmo para ver se o python faz
#os calculos sem dar erro, por isso a likelihood deverá dar valores estranhos.

#Lone kernel
print('-> lonely kernel')
kl.likelihood(kl.ExpSquared(19.0, 2.0), x, x, y, yerr)

start = time() # Calculation using george 
kernel = 19**2*ExpSquaredKernel(2.0**2)
gp = george.GP(kernel)
gp.compute(x,yerr)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))


#Sum of kernels
print('-> sum of kernels')
k1 = kl.ExpSquared(1.0,1.0) + kl.ExpSineSquared(1.,1.,1.)
k2 = kl.Sum(kl.ExpSquared(1.0,1.0),kl.ExpSineSquared(1.,1.,1.))
covk1 = kl.likelihood(k1, x, x, y, yerr)
covk2 = kl.likelihood(k2, x, x, y, yerr)

start = time() # Calculation using george 
kernel = 1.**2*ExpSquaredKernel(1.0**2) + ExpSine2Kernel(1., 1.)
gp = george.GP(kernel)
gp.compute(x,yerr)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))


#Multiplication  of kernels
print('-> multiplication of kernels')
a=kl.ExpSquared(1.0,1.0)*kl.ExpSineSquared(1.0,1.0,1.0)
kl.likelihood(a, x, x, y, yerr)
kl.likelihood(kl.Local_ExpSineSquared(1.0,1.0,1.0), x, x, y, yerr)
#   A Local_ExpSineSquared = ExpSquared * ExpSineSquared, logo se a
#multiplicação estiver a ser bem feita acho as duas devem dar os mesmos valores

start = time() # Calculation using george 
kernel = 1.**2*ExpSquaredKernel(1.0**2) * ExpSine2Kernel(1., 1.)
gp = george.GP(kernel)
gp.compute(x,yerr)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))


#Multiplication  and sum of kernels
print('-> multiplication and sum of kernels')
#multiplicar com white noise incluido
b=kl.ExpSquared(1.0,1.0)*kl.ExpSineSquared(1.,1.,1.) +kl.WhiteNoise(1.)
likelihood(b, x, x, y, yerr)

start = time() # Calculation using george 
kernel = 1.**2*ExpSquaredKernel(1.0**2) * ExpSine2Kernel(1., 1.) +  WhiteKernel(1.)
gp = george.GP(kernel)
gp.compute(x,yerr)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))
