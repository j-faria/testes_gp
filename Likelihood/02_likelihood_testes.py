# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:51:16 2016

@author: camacho
"""
import numpy as np
import matplotlib.pyplot as pl
pl.close("all") #fecha todas as figuras anteriores

########### EXEMPLO 1 - Generate some fake noisy data. ##########
#x = 10 * np.sort(np.random.rand(20))
#yerr = 0.2 * np.ones_like(x)
#y = np.sin(x) + yerr * np.random.randn(len(x))
#
##### CALCULO USANDO O GEORGE ###
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
##### CALCULO  DA LIKELIHOOD ###
#from Kernel import *
#
##definir kernel a usar
#kernel = ExpSquared 
#ES_theta = 1
#ES_l = 1
#x1=x
#x2=x
#
##calcular matrix de covariancia K, K* e K**
#K=np.zeros((len(x),len(x)))
#for i in range(len(x)):
#    for j in range(len(x)):
#        K[i,j]=kernel(x1[i],x2[j],ES_theta,ES_l)
#K=K+yerr**2*np.identity(len(x))      
#
#K_star=np.zeros(len(x))
#for i in range(len(x)):
#    for j in range(len(x)):
#        K_star[i]=kernel(x1[i],x2[j],ES_theta,ES_l)
#    
#K_2star=kernel(K_star,K_star,ES_theta,ES_l) 
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

########### EXEMPLO 2 - ExpSineSquared ##########
#x = 50 * np.sort(np.random.rand(100))
#yerr = 0.2 * np.ones_like(x)
#y = np.sin(x) + yerr * np.random.randn(len(x)) + 3
#pl.plot(x,y,'.')
#
#### CALCULO USANDO O GEORGE ###
#    #importar o modulo george e a kernel a usar
#import george
#from george.kernels import ExpSine2Kernel
#
#    #Set up the Gaussian process.
#kernelgeorge = 1.5**2*ExpSine2Kernel(2,15)
##kernel = CosineKernel(1.0)
#gp = george.GP(kernelgeorge)
#
#    #Pre-compute the factorization of the matrix.
#gp.compute(x,yerr)
#
#    #Compute the log likelihood.
#print(gp.lnlikelihood(y))
#
#### CALCULO  DA LIKELIHOOD ###
#
##definir kernel a usar
#from Kernel import *
#kernel = ExpSineSquared 
#ESS_theta = 1.5
#ESS_l = 2.0
#ESS_P = 15
#x1=x
#x2=x
#
##calcular matrix de covariancia K, K* e K**
#K=np.zeros((len(x),len(x)))
#for i in range(len(x)):
#    for j in range(len(x)):
#        K[i,j]=kernel(x1[i],x2[j],ESS_theta,ESS_l,ESS_P)
#K=K+yerr**2*np.identity(len(x))      
#
#K_star=np.zeros(len(x))
#for i in range(len(x)):
#    for j in range(len(x)):
#        K_star[i]=kernel(x1[i],x2[j],ESS_theta,ESS_l,ESS_P)
#    
#K_2star=kernel(K_star,K_star,ESS_theta,ESS_l,ESS_P)
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
#log_p = -0.5*np.dot(np.dot(np.dot(y.T,L.T),L_inv),y) - sum(np.log(np.diag(L))) \
#        - n*0.5*np.log(2*np.pi)            
#print(log_p)

###############################################################################

########### EXEMPLO 3 - Local_ExpSineSquared ##########
#x = 50 * np.sort(np.random.rand(100))
#yerr = 0.2 * np.ones_like(x)
#y = np.sin(x) + yerr * np.random.randn(len(x)) + 3
#pl.plot(x,y,'.')
#
#### CALCULO USANDO O GEOREGE ###
#    #importar o modulo george e a kernel a usar
#import george
#from george.kernels import ExpSquaredKernel, ExpSine2Kernel
#
#    #Set up the Gaussian process.
#kernelgeorge = 1.5**2*ExpSquaredKernel(50**2)*ExpSine2Kernel(2,15)
##kernel = CosineKernel(1.0)
#gp = george.GP(kernelgeorge)
#
#    #Pre-compute the factorization of the matrix.
#gp.compute(x,yerr)
#
#    #Compute the log likelihood.
#print(gp.lnlikelihood(y))
#
#### CALCULO  DA LIKELIHOOD ###
#
##definir kernel a usar
#from Kernel import *
#kernel = Local_ExpSineSquared 
#LESS_theta = 1.5
#LESS_l = 2.0
#LESS_P = 15
#x1=x
#x2=x
#
##calcular matrix de covariancia K, K* e K**
#K=np.zeros((len(x),len(x)))
#for i in range(len(x)):
#    for j in range(len(x)):
#        K[i,j]=kernel(x1[i],x2[j],LESS_theta,LESS_l,LESS_P)
#K=K+yerr**2*np.identity(len(x))      
#
#K_star=np.zeros(len(x))
#for i in range(len(x)):
#    for j in range(len(x)):
#        K_star[i]=kernel(x1[i],x2[j],LESS_theta,LESS_l,LESS_P)
#    
#K_2star=kernel(K_star,K_star,LESS_theta,LESS_l,LESS_P)
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
#log_p = -0.5*np.dot(np.dot(np.dot(y.T,L.T),L_inv),y) - sum(np.log(np.diag(L))) \
#        - n*0.5*np.log(2*np.pi)            
#print(log_p)

###############################################################################

########## EXEMPLO 4 - Soma de ExpSineSquared com ExpSquared ##########
x = 10 * np.sort(np.random.rand(50))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x)**2 + yerr * np.random.randn(len(x))
pl.plot(x,y,'*')

#### CALCULO USANDO O GEORGE ###
    #importar o modulo george e a kernel a usar
import george
from george.kernels import ExpSquaredKernel, ExpSine2Kernel

    #Set up the Gaussian process.
k1 = 10**2*ExpSine2Kernel(2.0/1**2,4)
k2 = 1**2*ExpSquaredKernel(10.0)  #original do exemplo
kernel = k1+k2
#kernel = CosineKernel(1.0)
gp = george.GP(kernel)

    #Pre-compute the factorization of the matrix.
gp.compute(x,yerr)

    #Compute the log likelihood.
print(gp.lnlikelihood(y))


#### CALCULO  DA LIKELIHOOD ###
#definir kernel a usar
from Kernel import *
kernel = Sum_ExpSineSquared_ExpSquared
x1=x
x2=x
ESS_theta=10
ESS_l=1
ESS_P= 4
ES_theta=1
ES_l=10

#calcular matrix de covariancia K, K* e K**
K=np.zeros((len(x),len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        K[i,j]=kernel(x1[i], x2[j],ESS_theta,ESS_l,ESS_P,ES_theta,ES_l)
K=K+yerr**2*np.identity(len(x))      

K_star=np.zeros(len(x))
for i in range(len(x)):
    for j in range(len(x)):
        K_star[i]=kernel(x1[i],x2[j],ESS_theta,ESS_l,ESS_P,ES_theta,ES_l)
    
K_2star=kernel(K_star,K_star,ESS_theta,ESS_l,ESS_P,ES_theta,ES_l) 

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