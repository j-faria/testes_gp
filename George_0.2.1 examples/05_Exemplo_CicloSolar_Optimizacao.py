# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 12:11:35 2016

@author: camacho
"""

########## IMPORTAR DADOS INICIAIS ##########
import numpy as np
import matplotlib.pyplot as pl

pl.close("all") #fecha todas as figuras anteriores

data= np.loadtxt('SN_m_tot_V2.0.txt') #data[linha,coluna]
#t=data[:,2] #todos os ciclos
#y=data[:,3]
#t=data[2965:3120,2] #ciclo 23
#y=data[2965:3120,3]
t=data[2965:-1,2] #ciclo 23 + ciclo 24
y=data[2965:-1,3]
yerr = np.ones_like(t)

########## USAR PROCESSOS GAUSSIANOS ##########
from george import kernels, GP, HODLRSolver
import george

#k1 = ExpSquaredKernel(200.0**2)
k2 = 100**2*kernels.ExpSine2Kernel((2.0/0.5)**2,135)
#k3 = 100**2*kernels.ExpSine2Kernel((2.0/1)**2,1200)
k0 = kernels.WhiteKernel(100)    

kernel = k2 #+ k0

    #optimization  - find the “best-fit” hyperparameters
gp = GP(kernel, mean=np.mean(y))
#gp = GP(kernel, solver=HODLRSolver)
gp.compute(t,yerr)
print(gp.lnlikelihood(y))
print(gp.grad_lnlikelihood(y))
print(kernel) #kernel  inicial

mu, cov = gp.predict(y, t) #mean mu and covariance cov
std = np.sqrt(np.diag(cov))

    #Graficos todos xpto - kernel inicial
pl.figure()
pl.plot(t,y,'.')
pl.grid()
pl.show()
pl.fill_between(t, mu+std, mu-std, color="k", alpha=0.1)
pl.plot(t, mu+std, color="k", alpha=1, lw=0.25)
pl.plot(t, mu-std, color="k", alpha=1, lw=0.25)
pl.plot(t, mu, color="r", alpha=1, lw=0.9)
#pl.errorbar(x, y, yerr=yerr,xerr=None, fmt=".k", capsize=0)
pl.xlabel("$tempo$")
pl.ylabel("$manchas$")

########## OPTIMIZAR HIPERPARAMETROS ##########
import scipy.optimize as op

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    ll = gp.lnlikelihood(y, quiet=True)

    # The scipy optimizer doesn't play well with infinities.
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    return -gp.grad_lnlikelihood(y, quiet=True)

## You need to compute the GP once before starting the optimization.
#gp.compute(t,yerr)
#
## Print the initial ln-likelihood.
#print(gp.lnlikelihood(y))

# Run the optimization routine.
p0 = gp.kernel.vector
results = op.minimize(nll, p0, jac=grad_nll)

# Update the kernel and print the final log-likelihood.
gp.kernel[:] = results.x
print('likelihoo:',gp.lnlikelihood(y))
print('kernel:',kernel) #kernel final

########## PARTE GRAFICA ##########
    #Compute the predicted values of the function at a fine grid of points 
#conditioned on the observed data
#x=t
#x = np.linspace(1996.124, 2008.958, 155) #ciclo 23
#x = np.linspace(min(t), max(t), len(t))
#x = np.linspace(1996.5, 2009, len(t))
#x = np.linspace(max(t),2050,(2050-max(t))*12) #previsao ?
#x=np.linspace(min(t),2050,(2050-min(t))*12)
mu, cov = gp.predict(y, t) #mean mu and covariance cov
std = np.sqrt(np.diag(cov))

    #Graficos todos xpto - kernel final
pl.figure()
pl.plot(t,y,'.')
pl.grid()
pl.show()
pl.fill_between(t, mu+std, mu-std, color="k", alpha=0.1)
pl.plot(t, mu+std, color="k", alpha=1, lw=0.25)
pl.plot(t, mu-std, color="k", alpha=1, lw=0.25)
pl.plot(t, mu, color="r", alpha=1, lw=0.9)
#pl.errorbar(x, y, yerr=yerr,xerr=None, fmt=".k", capsize=0)
pl.xlabel("$tempo$")
pl.ylabel("$manchas$")

