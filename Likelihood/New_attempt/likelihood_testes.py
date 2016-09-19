# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:56:24 2016

@author: camacho
"""

from likelihood import *

x = 10 * np.sort(np.random.rand(2000))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

x1=x
x2=x
#considerando a kernel ExpSquared com parametros:
theta1 = 1
l1 = 1
#cosiderando a kernel ExpSineSQuared com parametros:
theta2= 10
l2=1
P2=5

#a=ExpSineSquared(x1,x2,10,1,5)
#
#def argumentcount(x): #conta as variaveis de x
#    return x.__code__.co_argcount

#likelihood(kernels, x dado, x a calcular, y, yerr)
likelihood(ExpSquared(theta1,l1), x, x, y, yerr)

#somar
likelihood(ExpSquared(theta1,l1)+ExpSineSquared(theta2,l2,P2), x, x, y, yerr)

#multiplicar
likelihood(ExpSquared(theta1,l1)*ExpSineSquared(theta2,l2,P2), x, x, y, yerr)