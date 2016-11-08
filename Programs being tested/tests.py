# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:26:46 2016

@author: camacho
"""
import Kernel;reload(Kernel);kl = Kernel
import Likelihood as lk
import numpy as np

import matplotlib.pyplot as pl

import george
import george.kernels as ge

#INITIAL DATA
np.random.seed(1001)
x1 = 10 * np.sort(np.random.rand(101))
yerr1 = 0.2 * np.ones_like(x1)
y1 = np.sin(x1) + yerr1 * np.random.randn(len(x1))
###############################################################################

#EXAMPLE OURS-GEORGE USING ExpSineSquared
# Calculation using our stuff
print  '########## EXEMPLE EQUAL TO GEORGE ##########'
kernel=kl.ExpSineGeorge(2.0/1.1**2, 7.1)
lk.likelihood(kernel, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel, x1, x1, y1, yerr1)

# Calculation using george
kernelgeo = ge.ExpSine2Kernel(2.0/1.1**2, 7.1)
gp = george.GP(kernelgeo)
gp.compute(x1,yerr1)
print 'likelihood_george ->', gp.lnlikelihood(y1)
print 'gradient_george ->', gp.grad_lnlikelihood(y1)
print ' '
###############################################################################

#EXEMPLO 1 - ExpSquared
# Calculation using our stuff
print '########## EXPSQUARED ##########'
kernel1=kl.ExpSquared(19.0, 2.0)
lk.likelihood(kernel1, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel1, x1, x1, y1, yerr1)

# Calculation using george 
kernelg1 = 19.0**2*ge.ExpSquaredKernel(2.0**2)
gp = george.GP(kernelg1)
gp.compute(x1,yerr1)
print 'likelihood_george ->', gp.lnlikelihood(y1)
print 'gradient_george ->', gp.grad_lnlikelihood(y1)
print ' '
###############################################################################

# EXEMPLO 2 - ExpSineSquared
# Calculation using our stuff
print '########## EXPSINESQUARED'
kernel2=kl.ExpSineSquared(15.0, 2.0, 10.0)
lk.likelihood(kernel2, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel2, x1, x1, y1, yerr1)

# Calculation using george
kernelg2 = 15.0**2*ge.ExpSine2Kernel(2.0/2.0**2,10.0)
gp = george.GP(kernelg2)
gp.compute(x1,yerr1)
print 'likelihood_george ->', gp.lnlikelihood(y1)
print 'gradient_george ->', gp.grad_lnlikelihood(y1)
print ' '
###############################################################################

#### EXEMPLO 3 - RatQuadratic
# Calculation using our stuff
print '########## RATQUADRATIC ##########'
kernel3=kl.RatQuadratic(1.0,1.5,1.0)
lk.likelihood(kernel3, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel3, x1, x1, y1, yerr1)

# Calculation using george 
kernelg3 = 1.0**2*ge.RationalQuadraticKernel(1.5,1.0**2)
gp = george.GP(kernelg3)
gp.compute(x1,yerr1)
print 'likelihood_george ->', gp.lnlikelihood(y1)
print 'gradient_george ->', gp.grad_lnlikelihood(y1)
print ' '
###############################################################################

### EXEMPLO 10 - ExpSquared + ExpSineSquared
# Calculation using our stuff
print '########## SUM ES+ESS ##########'
kernel3=kl.ExpSquared(19.1, 1.3) + kl.ExpSineSquared(15.0, 1.5, 11.0) 
lk.likelihood(kernel3, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel3, x1, x1, y1, yerr1)

# Calculation using george
kernelg3 = 19.1**2*ge.ExpSquaredKernel(1.3**2) + 15.0**2*ge.ExpSine2Kernel(2.0/1.5**2,11.0)
gp = george.GP(kernelg3)
gp.compute(x1,yerr1)
print 'likelihood_george ->',  gp.lnlikelihood(y1)
print 'gradient_george ->', gp.grad_lnlikelihood(y1)
print ' '
###############################################################################

### EXEMPLO 11 - RatQuadratic + ExpSineSquared
# Calculation using our stuff
print '########## SUM RQ+ESS ##########'
kernel3=kl.RatQuadratic(11.0,1.5,1.0) + kl.ExpSineSquared(15.0, 1.0, 10.0) 
lk.likelihood(kernel3, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel3, x1, x1, y1, yerr1)

# Calculation using george
kernelg3 = 11.0**2*ge.RationalQuadraticKernel(1.5,1.0**2) + 15.0**2*ge.ExpSine2Kernel(2.0/1.0**2,10.0)
gp = george.GP(kernelg3)
gp.compute(x1,yerr1)
print 'likelihood_george ->',  gp.lnlikelihood(y1)
print 'gradient_george ->', gp.grad_lnlikelihood(y1)
print ' '
###############################################################################

### EXEMPLO 12 - ExpSquared+RatQuadratic
# Calculation using our stuff
print '########## SUM ES+RQ ##########'
kernel4=kl.ExpSquared(19.3, 1.5) +kl.RatQuadratic(11.6,1.6,1.6) 
lk.likelihood(kernel4, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel4, x1, x1, y1, yerr1)

# Calculation using george
kernelg4 = 19.3**2*ge.ExpSquaredKernel(1.5**2) +11.6**2*ge.RationalQuadraticKernel(1.6,1.6**2) 
gp = george.GP(kernelg4)
gp.compute(x1,yerr1)
print 'likelihood_george ->',  gp.lnlikelihood(y1)
print 'gradient_george ->', gp.grad_lnlikelihood(y1)
print ' '
###############################################################################

### EXEMPLO 21 - ExpSineGeorge * ExpSineGeorge
# Calculation using our stuff
print '########## MULTIPLICACAO George*George ##########'
kernel3=kl.ExpSineGeorge(2.0/1.1**2, 7.1)*kl.ExpSineGeorge(2.0/1.5**2, 10.1)
lk.likelihood(kernel3, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel3, x1, x1, y1, yerr1)

# Calculation using george
kernelg3=ge.ExpSine2Kernel(2.0/1.1**2, 7.1)*ge.ExpSine2Kernel(2.0/1.5**2, 10.1)
gp = george.GP(kernelg3)
gp.compute(x1,yerr1)
print 'likelihood_george ->',  gp.lnlikelihood(y1)
print 'gradient_george ->', gp.grad_lnlikelihood(y1)
print ' '
###############################################################################

### EXEMPLO 22 - ExpSquared * ExpSineSquared
# Calculation using our stuff
print '########## MULTIPLICACAO ES*ESS ##########'
kernel3=kl.ExpSquared(19.1, 1.3) * kl.ExpSineSquared(15.0, 1.5, 11.0) 
lk.likelihood(kernel3, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel3, x1, x1, y1, yerr1)

# Calculation using george
kernelg3 = 19.1**2*ge.ExpSquaredKernel(1.3**2) * 15.0**2*ge.ExpSine2Kernel(2.0/1.5**2,11.0)
gp = george.GP(kernelg3)
gp.compute(x1,yerr1)
print 'likelihood_george ->',  gp.lnlikelihood(y1)
print 'gradient_george ->', gp.grad_lnlikelihood(y1)
print ' '
###############################################################################

#### EXEMPLO 23 - WhiteNoise * ExpSineSquared
## Calculation using our stuff
#print '########## MULTIPLICACAO WN*ESS ##########'
#kernel3=kl.WhiteNoise(2.2)*kl.ExpSquared(13.1, 1.1)
#lk.likelihood(kernel3, x1, x1, y1, yerr1)
#lk.gradient_likelihood(kernel3, x1, x1, y1, yerr1)
#
## Calculation using george
#kernelg3 = ge.WhiteKernel(2.2) * 12.1**2*ge.ExpSquaredKernel(1.1**2)
#gp = george.GP(kernelg3)
#gp.compute(x1,yerr1)
#print 'likelihood_george ->',  gp.lnlikelihood(y1)
#print 'gradient_george ->', gp.grad_lnlikelihood(y1)
#print ' '
###############################################################################

##Devolve a matrix do gradiente no george
#xx1,_ = gp.parse_samples(x1)
#print gp.kernel.gradient(xx1).shape
#print gp.kernel.gradient(xx1)[:,:,0]

#xx2,_ = gp.parse_samples(x2)
#print gp.kernel.gradient(xx2).shape
#print gp.kernel.gradient(xx2)[:,:,0]
#print gp.kernel.gradient(xx2)[:,:,1]