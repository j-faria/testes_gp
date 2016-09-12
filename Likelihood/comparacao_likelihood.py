# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:05:53 2016

@author: camacho
"""

########## GERAR DADOS INICIAIS ALEATORIOS ##########
import numpy as np
import matplotlib.pyplot as pl

pl.close("all") #fecha todas as figuras anteriores

np.random.seed(10)

    #Generate some fake noisy data.
x = 10 * np.sort(np.random.rand(20))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))


########## CALCULO USANDO O GEOREGE ##########
    #importar o modulo george e a kernel a usar
import george
from george.kernels import ExpSquaredKernel

    #Set up the Gaussian process.
kernel = ExpSquaredKernel(1.0)  #original do exemplo
#kernel = CosineKernel(1.0)
gp = george.GP(kernel)

    #Pre-compute the factorization of the matrix.
gp.compute(x,yerr)

    #Compute the log likelihood.
print(gp.lnlikelihood(y))


########## CALCULO  DA LIKELIHOOD ##########
from Kernel import *

#definir kernel a usar
kernel = ExpSquared 
theta = 1
l = 1
x1=x
x2=x

#calcular matrix de covariancia K, K* e K**
K=np.zeros((len(x),len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        K[i,j]=kernel(theta,l,x1[i],x2[j])
K=K+yerr**2*np.identity(len(x))      

K_star=np.zeros(len(x))
for i in range(len(x)):
    for j in range(len(x)):
        K_star[i]=kernel(theta,l,x1[i],x2[j]) 
    
K_2star=kernel(theta,l,K_star,K_star) 

#para usar cholesky a matriz tem de ser positiva definida
L = np.linalg.cholesky(K)
L_inv= np.linalg.inv(L)
K_inv= np.linalg.inv(K)

y = np.array(y)
ystar_mean = np.dot(np.dot(K_star,K_inv),y)
ystar_var = np.dot(np.dot(K_star,K_inv),K_star.T)

#Calculo da log likelihood
n=len(x)
log_p2 = -0.5*np.dot(np.dot(np.dot(y.T,L.T),L_inv),y) - sum(np.log(np.diag(L))) \
            - n*0.5*np.log(2*np.pi)            
print(log_p2)