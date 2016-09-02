# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:01:18 2016

@author: camacho
"""

########## IMPORTAR DADOS INICIAIS ALEATORIOS ##########
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as pl

pl.close("all") #fecha todas as figuras anteriores

data = sm.datasets.get_rdataset("co2").data
t = np.array(data.time)
y = np.array(data.co2)

#pl.figure() #gráfico dos dados importados
pl.plot(t,y,'-')

########## USAR PROCESSOS GAUSSIANOS ##########
from george import kernels

    #long term smooth rising trend
k1 = 66.0**2 * kernels.ExpSquaredKernel(67.0**2)

    #decay away from exact periodicity
k2 = 2.4**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(2.0 / 1.3**2, 1.0)

    #medium term irregularities
k3 = 0.66**2 * kernels.RationalQuadraticKernel(0.78, 1.2**2)

    #noise model
k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2) + kernels.WhiteKernel(0.19)

    #final covariance function
kernel = k1 + k2 + k3 + k4

    #optimization  - find the “best-fit” hyperparameters
gp = george.GP(kernel, mean=np.mean(y))
gp.compute(t)
print(gp.lnlikelihood(y))
print(gp.grad_lnlikelihood(y))

    #Compute the predicted values of the function at a fine grid of points 
#conditioned on the observed data
x = np.linspace(max(t), 2025, 2000)
mu, cov = gp.predict(y, x) #mean mu and covariance cov
std = np.sqrt(np.diag(cov))

    #Graficos todos xpto
pl.fill_between(x, mu+std, mu-std, color="k", alpha=0.1) #cinza que preenche o espaço
pl.plot(x, mu+std, color="k", alpha=1, lw=0.25)     #mu+std
pl.plot(x, mu-std, color="k", alpha=1, lw=0.25)     #mu-std
pl.plot(x, mu, color="k", alpha=1, lw=0.5)          #mu
#pl.errorbar(t, y, fmt=".k", capsize=0)
pl.xlabel("$x$")
pl.ylabel("$y$")









