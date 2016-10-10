# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:26:46 2016

@author: camacho
"""

import Kernel;reload(Kernel);kl = Kernel
import Likelihood as lk
import numpy as np
from time import time

import george
from george.kernels import *

### EXEMPLO 1 - ExpSquared
#np.random.seed(1001)
x1 = 10 * np.sort(np.random.rand(101))
yerr1 = 0.2 * np.ones_like(x1)
y1 = np.sin(x1) + yerr1 * np.random.randn(len(x1))

kernel1=kl.ExpSquared(19.0, 2.0)
lk.likelihood(kernel1, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel1, x1, x1, y1, yerr1)

# Calculation using george 
start = time() 
kernelg1 = 19.0**2*ExpSquaredKernel(2.0**2)
gp = george.GP(kernelg1)
gp.compute(x1,yerr1)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y1))
print 'gradient_george ->', gp.grad_lnlikelihood(y1)

xx1,_ = gp.parse_samples(x1)
print gp.kernel.gradient(xx1).shape
print gp.kernel.gradient(xx1)[:,:,0]



#### EXEMPLO 2 - ExpSineSquared
#x2 = 10 * np.sort(np.random.rand(102))
#yerr2 = 0.2 * np.ones_like(x2)
#y2 = np.sin(x2) + yerr2 * np.random.randn(len(x2))
#
#kernel2=kl.ExpSineSquared(15.0, 1.0, 10.0)
#lk.likelihood(kernel2, x2, x2, y2, yerr2)
#lk.gradient_likelihood(kernel2, x2, x2, y2, yerr2)
#
## Calculation using george
#start = time()  
#kernelg2 = 15.0**2*ExpSine2Kernel(2.0/1.0**2,10.0)
#gp = george.GP(kernelg2)
#gp.compute(x2,yerr2)
#print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y2))
#print 'gradient_george ->', gp.grad_lnlikelihood(y2)


#### EXEMPLO 3 - ExpSquared + ExpSineSquared
#x3 = 10 * np.sort(np.random.rand(103))
#yerr3 = 0.2 * np.ones_like(x3)
#y3 = np.sin(x3) + yerr3 * np.random.randn(len(x3))
#
#kernel3=kl.ExpSquared(19.0, 2.0) + kl.ExpSineSquared(15.0, 1.0, 10.0)
#kl.likelihood(kernel3, x3, x3, y3, yerr3)
#kl.gradient_likelihood(kernel3, x3, x3, y3, yerr3)
#
## Calculation using george
#start = time()  
#kernelg3 = 19.0**2*ExpSquaredKernel(2.0**2) + 15.0**2*ExpSine2Kernel(2.0/1.0**2,10.0)
#gp = george.GP(kernelg3)
#gp.compute(x3,yerr3)
#print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y3))
#print 'gradient_george ->', gp.grad_lnlikelihood(y3)
