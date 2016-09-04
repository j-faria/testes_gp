# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:14:28 2016

@author: camacho
"""

########## IMPORTAR DADOS INICIAIS ##########
import numpy as np
import matplotlib.pyplot as pl

pl.close("all") #fecha todas as figuras anteriores

data= np.loadtxt('SN_m_tot_V2.0.txt') #data[linha,coluna]
t=data[2965:3120,2] #ciclo 23
y=data[2965:3120,3]
#t=data[2965:-1,2] #ciclo 23 + ciclo 24
#y=data[2965:-1,3]

pl.figure()
pl.plot(t,y,'*')
pl.grid()
pl.show()

########## USAR PROCESSOS GAUSSIANOS ##########
from george import kernels, GP, HODLRSolver
import george

#k1 = ExpSquaredKernel(200.0**2)
k2 = (100**2)*kernels.ExpSine2Kernel((2.0/0.01)**2,11)

k3 = kernels.WhiteKernel(0.2)    

kernel = k2 #+ k3

    #optimization  - find the “best-fit” hyperparameters
#gp = GP(kernel, mean=np.mean(y))
gp = GP(kernel, solver=HODLRSolver)
gp.compute(t)
print(gp.lnlikelihood(y))
print(gp.grad_lnlikelihood(y))


    #Compute the predicted values of the function at a fine grid of points 
#conditioned on the observed data
#x = np.linspace(1996.124, 2008.958, 155) #ciclo 23
x = np.linspace(min(t), max(t), len(t))

#x = np.linspace(max(t),2025,(2025-max(t))*12) #previsao ?
mu, cov = gp.predict(y, t) #mean mu and covariance cov
std = np.sqrt(np.diag(cov))

    #Graficos todos xpto
pl.fill_between(t, mu+std, mu-std, color="k", alpha=0.1)
pl.plot(t, mu+std, color="k", alpha=1, lw=0.25)
pl.plot(t, mu-std, color="k", alpha=1, lw=0.25)
pl.plot(t, mu, color="k", alpha=1, lw=0.5)
#pl.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
pl.xlabel("$x$")
pl.ylabel("$y$")



