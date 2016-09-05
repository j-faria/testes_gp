# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:36:18 2016

@author: camacho
"""

########## GERAR DADOS INICIAIS ALEATORIOS ##########
import numpy as np
import matplotlib.pyplot as pl

pl.close("all") #fecha todas as figuras anteriores

    #Generate some fake noisy data.
x= 10*np.sort(np.random.rand(20))
yerr= 2*np.ones_like(x)

y= x**2 + np.random.randn(len(x))
x=np.sort(x + np.random.randn(len(x)))
#y=yerr * np.random.randn(20)
pl.figure()
pl.plot(x,y,'.') #faz o plot de (x,y) com pontinhos

########## USAR PROCESSOS GAUSSIANOS ##########
from george import kernels, GP, HODLRSolver
import george

k1= kernels.DotProductKernel(1)
k2= kernels.DotProductKernel(1)

kernel= kernels.Product(k1,k2) #multiplicar kernels
#kernel =  k2

    #optimization  - find the “best-fit” hyperparameters
#gp = GP(kernel, mean=np.mean(y))
gp= GP(kernel, solver=HODLRSolver)
gp.compute(x,yerr)
print(gp.lnlikelihood(y))
print(gp.grad_lnlikelihood(y))


    #Compute the predicted values of the function at a fine grid of points 
#conditioned on the observed data
mu, cov = gp.predict(y, x) #mean mu and covariance cov
std = np.sqrt(np.diag(cov))

    #Graficos todos xpto
pl.fill_between(x, mu+std, mu-std, color="k", alpha=0.1)
pl.plot(x, mu+std, color="k", alpha=1, lw=0.25)
pl.plot(x, mu-std, color="k", alpha=1, lw=0.25)
pl.plot(x, mu, color="r", alpha=1, lw=0.9)
pl.errorbar(x, y,  yerr, fmt=".k", capsize=0)
pl.xlabel("$x$")
pl.ylabel("$y$")

