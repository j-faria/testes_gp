# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:51:16 2016

@author: camacho
"""
import numpy as np
import matplotlib.pyplot as pl
pl.close("all") #fecha todas as figuras anteriores

## EXEMPLO 1 - Generate some fake noisy data.
#x = 10 * np.sort(np.random.rand(20))
#yerr = 0.2 * np.ones_like(x)
#y = np.sin(x) + yerr * np.random.randn(len(x))
#
########### CALCULO USANDO O GEOREGE ##########
#    #importar o modulo george e a kernel a usar
#import george
#from george.kernels import ExpSquaredKernel
#
#    #Set up the Gaussian process.
#kernel = ExpSquaredKernel(1.0)  #original do exemplo
##kernel = CosineKernel(1.0)
#gp = george.GP(kernel)
#
#    #Pre-compute the factorization of the matrix.
#gp.compute(x,yerr)
#
#    #Compute the log likelihood.
#print(gp.lnlikelihood(y))
#
#
########### CALCULO  DA LIKELIHOOD ##########
#from Kernel import *
#
##definir kernel a usar
#kernel = ExpSquared 
#theta = 1
#l = 1
#x1=x
#x2=x
#
##calcular matrix de covariancia K, K* e K**
#K=np.zeros((len(x),len(x)))
#for i in range(len(x)):
#    for j in range(len(x)):
#        K[i,j]=kernel(theta,l,x1[i],x2[j])
#K=K+yerr**2*np.identity(len(x))      
#
#K_star=np.zeros(len(x))
#for i in range(len(x)):
#    for j in range(len(x)):
#        K_star[i]=kernel(theta,l,x1[i],x2[j]) 
#    
#K_2star=kernel(theta,l,K_star,K_star) 
#
##para usar cholesky a matriz tem de ser positiva definida
#L = np.linalg.cholesky(K)
#L_inv= np.linalg.inv(L)
#K_inv= np.dot(L_inv,L.T)
#
#y = np.array(y)
#ystar_mean = np.dot(np.dot(K_star,K_inv),y)
#ystar_var = np.dot(np.dot(K_star,K_inv),K_star.T)
#
##Calculo da log likelihood
#n=len(x)
#log_p2 = -0.5*np.dot(np.dot(np.dot(y.T,L.T),L_inv),y) - sum(np.log(np.diag(L))) \
#            - n*0.5*np.log(2*np.pi)            
#print(log_p2)

###############################################################################

# EXEMPLO 2 - ExpSineSquared
x = 50 * np.sort(np.random.rand(100))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x)) + 3
pl.plot(x,y,'.')

########## CALCULO USANDO O GEOREGE ##########
    #importar o modulo george e a kernel a usar
from george.kernels import ExpSine2Kernel

    #Set up the Gaussian process.
kernelgeorge = 1.5**2*ExpSine2Kernel(2,15)
#kernel = CosineKernel(1.0)
gp = george.GP(kernelgeorge)

    #Pre-compute the factorization of the matrix.
gp.compute(x,yerr)

    #Compute the log likelihood.
print(gp.lnlikelihood(y))

########## CALCULO  DA LIKELIHOOD ##########

#definir kernel a usar
from Kernel import *
kernel = ExpSineSquared 
theta = 1.5
l = 2.0
P = 15
x1=x
x2=x

#from likelihood_func import *
#likelihood(kernel,x,x,"theta=1.5","l=2.0","P=15")
#calcular matrix de covariancia K, K* e K**
K=np.zeros((len(x),len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        K[i,j]=kernel(theta,l,P,x1[i],x2[j])
K=K+yerr**2*np.identity(len(x))      

K_star=np.zeros(len(x))
for i in range(len(x)):
    for j in range(len(x)):
        K_star[i]=kernel(theta,l,P,x1[i],x2[j])
    
K_2star=kernel(theta,l,P,K_star,K_star)

#para usar cholesky a matriz tem de ser positiva definida
L = np.linalg.cholesky(K)
L_inv= np.linalg.inv(L)
K_inv= np.dot(L_inv,L.T)

y = np.array(y)
ystar_mean = np.dot(np.dot(K_star,K_inv),y)
ystar_var = np.dot(np.dot(K_star,K_inv),K_star.T)

#Calculo da log likelihood
n=len(x)
log_p = -0.5*np.dot(np.dot(np.dot(y.T,L.T),L_inv),y) - sum(np.log(np.diag(L))) \
        - n*0.5*np.log(2*np.pi)            
print(log_p)

###############################################################################