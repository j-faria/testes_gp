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
x = 10 * np.sort(np.random.rand(20))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

###############################################################################
#likelihood(kernels, x dado, x a calcular, y, yerr)

#   Os parametros foram dados às três pancadas só mesmo para ver se o python faz
#os calculos sem dar erro, por isso a likelihood deverá dar valores estranhos.

#kernel sozinha
print('-> lonely kernel')
kl.likelihood(kl.ExpSquared(19, 2), x, x, y, yerr)
# Calculo usando o george 
start = time()
kernel = 19**2*ExpSquaredKernel(2.0**2)
gp = george.GP(kernel)
gp.compute(x,yerr)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))

#somar
print('-> sum of kernels')
k = kl.ExpSquared(1.,1.) + kl.ExpSineSquared(1.,1.,1.)
covK = kl.likelihood(k, x, x, y, yerr)
# Calculo usando o george 
start = time()
kernel = ExpSquaredKernel(1.) + ExpSine2Kernel(1., 1.)
gp = george.GP(kernel)
gp.compute(x,yerr)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))


#multiplicar
print('-> multiplication of kernels')
a=kl.Sum_Kernel(kl.ExpSquared(10,1)*kl.ExpSineSquared(1,1,5))
kl.likelihood(a, x, x, y, yerr)
kl.likelihood(kl.Local_ExpSineSquared(10,1,5), x, x, y, yerr)
#   A Local_ExpSineSquared = ExpSquared * ExpSineSquared, logo se a
#multiplicação estiver a ser bem feita as duas devem dar os mesmos valores


#print('-> multiplication and sum of kernels')
##multiplicar com white noise incluido
#likelihood(ExpSquared(10,1)*ExpSineSquared(1,1,5) +WhiteNoise(2), x, x, y, yerr)