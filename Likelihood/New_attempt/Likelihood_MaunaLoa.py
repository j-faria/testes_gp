# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:59:59 2016

@author: camacho
"""
import numpy as np
import statsmodels.api as sm

##### EXEMPLO DO SITE DO GEORGE #####
data = sm.datasets.get_rdataset("co2").data
t = np.array(data.time)
y = np.array(data.co2)

#Calculo usando o George
from george import kernels, GP, HODLRSolver
k1 = 66.0**2 * kernels.ExpSquaredKernel(67.0**2)
k2 = 2.4**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(2.0 / 1.3**2, 1.0)
k3 = 0.66**2 * kernels.RationalQuadraticKernel(0.78, 1.2**2)
k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2) + kernels.WhiteKernel(0.19)
kernel1 = k1 + k2 + k3 + k4

import george
#gp=GP(kernel, solver=HODLRSolver)
gp = george.GP(kernel1, mean=np.mean(y))
gp.compute(t)
print('Likelihood do george')
print(gp.lnlikelihood(y))

#calculo vindo dos programas feitos por nós
from likelihood import *

yerr=1e-6 #não falam em erro mas se nao der valor a matriz bate mal 

x1=ExpSquared(66.0,67.0)
x2=ExpSquared(2.4,90)*ExpSineSquared(2.4,1.3,1.0)
x3=RatQuadratic(0.66,1.2,0.78)
x4=ExpSquared(0.18,1.6)+WhiteNoise(0.19)
kernel2=x1+x2+x3+x4
print('-> Likelihood nossa')
likelihood(kernel2, t, t, y, yerr)

#Duas hipoteses: ou estou a meter os valor na kernel mal (acho que não)
#               ou a minha maneira de somar e multiplicar kernels nao funciona