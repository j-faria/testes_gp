# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:56:24 2016

@author: camacho
"""

from likelihood import *

#####  DADOS INICIAS  #########################################################
x = 10 * np.sort(np.random.rand(2000))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

x1=x
x2=x
###############################################################################
#likelihood(kernels, x dado, x a calcular, y, yerr)

#   Os parametros foram dados às três pancadas só mesmo para ver se o python faz
#os calculos sem dar erro, por isso a likelihood deverá dar valores estranhos.

#kernel sozinha
print('-> lonely kernel')
likelihood(ExpSquared(19,2), x, x, y, yerr)

print('-> sum of kernels')
#somar
likelihood(ExpSquared(10,1)+ExpSineSquared(1,1,5), x, x, y, yerr)

print('-> multiplication of kernels')
#multiplicar
likelihood(ExpSquared(10,1)*ExpSineSquared(1,1,5), x, x, y, yerr)
likelihood(Local_ExpSineSquared(10,1,5), x,  x, y ,yerr)
#   A Local_ExpSineSquared = ExpSquared * ExpSineSquared, logo se a
#multiplicação estiver a ser bem feita as duas devem dar os mesmos valores

print('-> multiplication and sum of kernels')
#multiplicar com white noise incluido
likelihood(ExpSquared(theta1,l1)*ExpSineSquared(theta2,l2,P2) +WhiteNoise(2), x, x, y, yerr)