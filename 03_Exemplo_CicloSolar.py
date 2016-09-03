# -*- coding: utf-8 -*-
### TRABALHO EM CURSO ###

########## IMPORTAR DADOS INICIAIS ##########
import numpy as np
import matplotlib.pyplot as pl

pl.close("all") #fecha todas as figuras anteriores

data= np.loadtxt('SN_m_tot_V2.0.txt') #data[linha,coluna]
t=data[:,2]
y=data[:,3]

pl.figure()
pl.plot(t,y,'-')
pl.grid()
pl.show()

########## USAR PROCESSOS GAUSSIANOS ##########
from george import kernels, GP
import george


    #para garantir que nao vai para valores negativos
#k0 = kernels.ConstantKernel(1)

    #ciclo de ~11 anos
#k1 = 50**2 *  kernels.ExpSine2Kernel(2.0 / 1.5**2, 1.0)
k1 = 100**2*kernels.CosineKernel(11) #amplitude=100 perido=11

    #white noise
k2 = 10**2*kernels.WhiteKernel(0.2)
    
    #ciclo de ~50 anos
k3 = 100*2*kernels.CosineKernel(50)

kernel   = k1 + k2 + k3
    #optimization  - find the “best-fit” hyperparameters
#gp = GP(kernel, mean=np.mean(y))
gp = GP(kernel, solver=george.HODLRSolver)
gp.compute(t)
print(gp.lnlikelihood(y))
print(gp.grad_lnlikelihood(y))

    #Compute the predicted values of the function at a fine grid of points 
#conditioned on the observed data
x = np.linspace(max(t), 2050, 2015)
mu, cov = gp.predict(y, x) #mean mu and covariance cov
std = np.sqrt(np.diag(cov))

    #Graficos todos xpto
pl.fill_between(x, mu+std, mu-std, color="k", alpha=0.1) #cinza que preenche o espaço
pl.plot(x, mu+std, color="k", alpha=1, lw=0.25)     #mu+std
pl.plot(x, mu-std, color="k", alpha=1, lw=0.25)     #mu-std
pl.plot(x, mu, color="k", alpha=1, lw=0.5)          #mu
#pl.errorbar(t, y, fmt=".k", capsize=0)  #erros de metição?
pl.xlabel("$x$")
pl.ylabel("$y$")
