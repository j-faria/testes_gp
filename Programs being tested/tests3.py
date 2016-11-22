# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:30:54 2016

@author: camacho
"""

import Kernel;reload(Kernel);kl = Kernel
import kernel_likelihood;reload(kernel_likelihood); lk= kernel_likelihood
import kernel_optimization as opt

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
print ''
print '3 - RPROP'

#EXAMPLE OURS-GEORGE USING ExpSineSquared
print  '########## EXEMPLE EQUAL TO GEORGE ##########'

kernel1=kl.ExpSineGeorge(2.0/1.1**2, 7.1)
print 'Initial kernel ->', kernel1; 
lk.likelihood(kernel1, x1, x1, y1, yerr1)
print''
#print 'gradient ->', lk.gradient_likelihood(kernel1, x1, x1, y1, yerr1); print ''


#opt.optimization(kernel1, x1, x1, y1, yerr1,method='CGA')
#opt.optimization(kernel1, x1, x1, y1, yerr1,method='SDA')
opt.optimization(kernel1, x1, x1, y1, yerr1,method='RPROP')

print '########## Calculations from george ##########'
kernel = ge.ExpSine2Kernel(2.0/1.1**2, 7.1)
gp = george.GP(kernel)
gp.compute(x1,yerr1)


print 'Initial kernel ->', kernel
#print 'likelihood_george ->', gp.lnlikelihood(y1)
#print 'gradient_george ->', gp.grad_lnlikelihood(y1); print ''

### OPTIMIZE HYPERPARAMETERS
import scipy.optimize as op
# Define the objective function (negative log-likelihood in this case).
def nll(p):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    ll = gp.lnlikelihood(y1, quiet=True)

    # The scipy optimizer doesn't play well with infinities.
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    return -gp.grad_lnlikelihood(y1, quiet=True)
    #print 'cenas', -gp.grad_lnlikelihood(y1, quiet=True)
# You need to compute the GP once before starting the optimization.
#gp.compute(x1,yerr1)
## Print the initial ln-likelihood.
#print(gp.lnlikelihood(y1))
## Run the optimization routine.

p0 = gp.kernel.vector
#print 'p0=',p0
#print 'nll cena=', gp.kernel[:]
#results = op.minimize(nll, p0,jac=grad_nll)
results = op.minimize(nll, p0,method='CG', jac=grad_nll,options={'maxiter':10})

# Update the kernel and print the final log-likelihood.
gp.kernel[:] = results.x
print 'total iterations =', 10
print 'likelihood_george =', gp.lnlikelihood(y1)
print 'kernel_george =', kernel #kernel final
###############################################################################