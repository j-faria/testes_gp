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


###EXAMPLE OURS-GEORGE USING ExpSineSquared
#np.random.seed(1001)
#x2 = 10 * np.sort(np.random.rand(102))
#yerr2 = 0.2 * np.ones_like(x2)
#y2 = np.sin(x2) + yerr2 * np.random.randn(len(x2))
##pl.plot(x2,y2)
#
#print  '########## EXEMPLE EQUAL TO GEORGE ##########'
#kernel=kl.ExpSineGeorge(2.0/1.1**2, 7.1)
#lk.likelihood(kernel, x2, x2, y2, yerr2)
#lk.gradient_likelihood(kernel, x2, x2, y2, yerr2)
#
## Calculation using george
#kernelgeo = ge.ExpSine2Kernel(2.0/1.1**2, 7.1)
#gp = george.GP(kernelgeo)
#gp.compute(x2,yerr2)
#print 'likelihood_george ->', gp.lnlikelihood(y2)
#print 'gradient_george ->', gp.grad_lnlikelihood(y2)
###############################################################################

## EXEMPLO 1 - ExpSquared
#np.random.seed(1001)
#x1 = 10 * np.sort(np.random.rand(101))
#yerr1 = 0.2 * np.ones_like(x1)
#y1 = np.sin(x1) + yerr1 * np.random.randn(len(x1))
#
## Calculation using our stuff
#print '########## EXPSQUARED ##########'
#kernel1=kl.ExpSquared(19.0, 2.0)
#lk.likelihood(kernel1, x1, x1, y1, yerr1)
#lk.gradient_likelihood(kernel1, x1, x1, y1, yerr1)
#
## Calculation using george 
#kernelg1 = 19.0**2*ge.ExpSquaredKernel(2.0**2)
#gp = george.GP(kernelg1)
#gp.compute(x1,yerr1)
#print 'likelihood_george ->', gp.lnlikelihood(y1)
#print 'gradient_george ->', gp.grad_lnlikelihood(y1)
###############################################################################

## EXEMPLO 2 - ExpSineSquared
## Calculation using our stuff
#print '########## EXPSINESQUARED'
#kernel2=kl.ExpSineSquared(15.0, 2.0, 10.0)
#lk.likelihood(kernel2, x2, x2, y2, yerr2)
#print 'gradient ->', lk.gradient_likelihood(kernel2, x2, x2, y2, yerr2)
#
## Calculation using george
#kernelg2 = 15.0**2*ge.ExpSine2Kernel(2.0/2.0**2,10.0)
#gp = george.GP(kernelg2)
#gp.compute(x2,yerr2)
#print 'likelihood_george ->', gp.lnlikelihood(y2)
#print 'gradient_george ->', gp.grad_lnlikelihood(y2)
###############################################################################

##### EXEMPLO 3 - RatQuadratic
#np.random.seed(1003)
#x3 = 10 * np.sort(np.random.rand(103))
#yerr3 = 0.2 * np.ones_like(x3)
#y3 = np.sin(x3) + yerr3 * np.random.randn(len(x3))
#
#print '########## RATQUADRATIC ##########'
#kernel3=kl.RatQuadratic(1.0,1.5,1.0)
#lk.likelihood(kernel3, x3, x3, y3, yerr3)
#lk.gradient_likelihood(kernel3, x3, x3, y3, yerr3)
#
## Calculation using george 
#kernelg3 = 1.0**2*ge.RationalQuadraticKernel(1.5,1.0**2)
#gp = george.GP(kernelg3)
#gp.compute(x3,yerr3)
#print 'likelihood_george ->', gp.lnlikelihood(y3)
#print 'gradient_george ->', gp.grad_lnlikelihood(y3)
###############################################################################

### EXEMPLO 10 - ExpSquared + ExpSineSquared
x1 = 10 * np.sort(np.random.rand(103))
yerr1 = 0.2 * np.ones_like(x1)
y1 = np.sin(x1) + yerr1 * np.random.randn(len(x1))

print '########## SOMA ES+ESS ##########'
kernel3=kl.ExpSquared(19.1, 1.3) + kl.ExpSineSquared(15.0, 1.5, 11.0) 
lk.likelihood(kernel3, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel3, x1, x1, y1, yerr1)

# Calculation using george
kernelg3 = 19.1**2*ge.ExpSquaredKernel(1.3**2) + 15.0**2*ge.ExpSine2Kernel(2.0/1.5**2,11.0)
gp = george.GP(kernelg3)
gp.compute(x1,yerr1)
print 'likelihood_george ->',  gp.lnlikelihood(y1)
print 'gradient_george ->', gp.grad_lnlikelihood(y1)
###############################################################################

### EXEMPLO 11 - RatQuadratic + ExpSineSquared
x1 = 10 * np.sort(np.random.rand(103))
yerr1 = 0.2 * np.ones_like(x1)
y1 = np.sin(x1) + yerr1 * np.random.randn(len(x1))

print '########## Soma RQ+ESS ##########'
kernel3=kl.RatQuadratic(11.0,1.5,1.0) + kl.ExpSineSquared(15.0, 1.0, 10.0) 
lk.likelihood(kernel3, x1, x1, y1, yerr1)
lk.gradient_likelihood(kernel3, x1, x1, y1, yerr1)

# Calculation using george
kernelg3 = 11.0**2*ge.RationalQuadraticKernel(1.5,1.0**2) + 15.0**2*ge.ExpSine2Kernel(2.0/1.0**2,10.0)
gp = george.GP(kernelg3)
gp.compute(x1,yerr1)
print 'likelihood_george ->',  gp.lnlikelihood(y1)
print 'gradient_george ->', gp.grad_lnlikelihood(y1)
###############################################################################

##Devolve a matrix do gradiente no george
#xx1,_ = gp.parse_samples(x1)
#print gp.kernel.gradient(xx1).shape
#print gp.kernel.gradient(xx1)[:,:,0]
#
#xx2,_ = gp.parse_samples(x2)
#print gp.kernel.gradient(xx2).shape
#print gp.kernel.gradient(xx2)[:,:,0]
#print gp.kernel.gradient(xx2)[:,:,1]