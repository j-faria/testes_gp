# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:23:01 2016

@author: camacho
"""

#Exemplo da pagina do george
########## GERAR DADOS INICIAIS ALEATORIOS ##########
import numpy as np
import matplotlib.pyplot as pl

pl.close("all") #fecha todas as figuras anteriores

    #Generate some fake noisy data.
x = 10 * np.sort(np.random.rand(20))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))
pl.figure()
pl.plot(x,y,'*') #faz o plot de (x,y) com estrelinhas

########## USAR PROCESSOS GAUSSIANOS ##########
    #importar o modulo george e a kernel a usar
import george
from george.kernels import ExpSquaredKernel, Matern32Kernel, CosineKernel

    #Set up the Gaussian process.
kernel = 1*ExpSquaredKernel(1.0)  #original do exemplo
#kernel = CosineKernel(1.0)
gp = george.GP(kernel)

    #Pre-compute the factorization of the matrix.
gp.compute(x, yerr)

    #Compute the log likelihood.
print(gp.lnlikelihood(y))
print(gp.grad_lnlikelihood(y))
#like=gp.lnlikelihood(y)

    #Compute the predicted values of the function at a fine grid of points 
#conditioned on the observed data
t = np.linspace(0, 10, 500)
mu, cov = gp.predict(y, t) #mean mu and covariance cov
std = np.sqrt(np.diag(cov))

    #Graficos todos xpto
pl.figure()
pl.fill_between(t, mu+std, mu-std, color="k", alpha=0.1)
pl.plot(t, mu+std, color="k", alpha=1, lw=0.25)
pl.plot(t, mu-std, color="k", alpha=1, lw=0.25)
pl.plot(t, mu, color="k", alpha=1, lw=0.5)
pl.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
pl.xlabel("$x$")
pl.ylabel("$y$")

