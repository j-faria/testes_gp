# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:26:46 2016

@author: camacho
"""
import numpy as np
import Kernel;reload(Kernel);kl = Kernel
from time import time

import george
from george.kernels import *

### EXEMPLO 1 - ExpSquared
x1 = 10 * np.sort(np.random.rand(101))
yerr1 = 0.2 * np.ones_like(x1)
y1 = np.sin(x1) + yerr1 * np.random.randn(len(x1))


kernel0=kl.ExpSquared(19.0, 2.0)
kl.likelihood(kernel0, x1, x1, y1, yerr1)
kl.gradient_likelihood(kernel0, x1, x1, y1, yerr1)

# Calculation using george 
start = time() 
kernelg1 = 19.0**2*ExpSquaredKernel(2.0**2)
gp = george.GP(kernelg1)
gp.compute(x1,yerr1)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y1))
print 'gradient_george ->', gp.grad_lnlikelihood(y1)


### EXEMPLO 2 - ExpSineSquared
x2 = 10 * np.sort(np.random.rand(102))
yerr2 = 0.2 * np.ones_like(x2)
y2 = np.sin(x2) + yerr2 * np.random.randn(len(x2))


kernel2=kl.ExpSineSquared(15.0, 1.0, 10.0)
kl.likelihood(kernel2, x2, x2, y2, yerr2)
kl.gradient_likelihood(kernel2, x2, x2, y2, yerr2)

# Calculation using george
start = time()  
kernelg2 = 15.0**2*ExpSine2Kernel(2.0/1.0**2,10.0)
gp = george.GP(kernelg2)
gp.compute(x2,yerr2)
print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y2))
print 'gradient_george ->', gp.grad_lnlikelihood(y2)