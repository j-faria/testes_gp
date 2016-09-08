# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:06:31 2016

@author: camacho
"""
import numpy as np
#import matplotlib.pyplot as pl
from Kernel import *
pl.close("all") #fecha todas as figuras abertas anteriormente

########## Dados Iniciais ##########
# Retirado do 01_Exemplo_simples
np.random.seed(10)  #Generate some fake noisy data.
x = 10 * np.sort(np.random.rand(20))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))
#pl.figure()
#pl.plot(x,y,'*') #faz o plot de (x,y) com estrelinhas


########## definir kernel a usar e parametros ##########
kernel = ExpSquared
theta = 1
l = 1
x1=x
x2=x

#kernel = Local_ExpSineSquared
#theta = 1
#l = 10
#P=2
#x1=x
#x2=x

########## Calculos da log likelihood ##########
if kernel == ExpSquared:
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(theta,l,x1[i],x2[j])
    
    K=K+yerr*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(xstar)):
            K_star[i]=kernel(theta,l,x1[i],x2[j])
        
    K_2star=kernel(theta,l,K_star,K_star)
    
    # PONTO 3 e 4
    L = np.linalg.cholesky(K)
    L_trans = L.T
    L_inv = np.linalg.inv(L)
    K_inv= np.dot( np.linalg.inv(L_trans), L_inv)
    
    y = np.array(y)
    ystar_mean = np.dot(np.dot(K_star,K_inv),y)
    ystar_var = np.dot(np.dot(K_star,K_inv),K_star.T)
    
    #Calculo da log likelihood
    n=len(x)
    log_p = -0.5*np.dot(np.dot(y.T,K_inv),y) - sum(np.log(np.diag(L))) - n*0.5*np.log(2*np.pi)
    print(log_p)

elif kernel == ExpSineSquared or kernel == Local_ExpSineSquared:
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(theta,l,P,x1[i],x2[j])
    
    K=K+yerr*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(xstar)):
            K_star[i]=kernel(theta,l,P,x1[i],x2[j])
        
    K_2star=kernel(theta,l,P,K_star,K_star)
    
    # PONTO 3 e 4
    L = np.linalg.cholesky(K)
    L_trans = L.T
    L_inv = np.linalg.inv(L)
    K_inv= np.dot( np.linalg.inv(L_trans), L_inv)
    
    y = np.array(y)
    ystar_mean = np.dot(np.dot(K_star,K_inv),y)
    ystar_var = np.dot(np.dot(K_star,K_inv),K_star.T)
    
    #Calculo da log likelihood
    n=len(x)
    log_p = -0.5*np.dot(np.dot(y.T,K_inv),y) - sum(np.log(np.diag(L))) - n*0.5*np.log(2*np.pi)
    print(log_p)

elif kernel == RatQuadratic:   
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(theta,l,alpha,x1[i],x2[j])
    
    K=K+yerr*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(xstar)):
            K_star[i]=kernel(theta,l,alpha,x1[i],x2[j])
        
    K_2star=kernel(theta,l,alpha,K_star,K_star)
    
    # PONTO 3 e 4
    L = np.linalg.cholesky(K)
    L_trans = L.T
    L_inv = np.linalg.inv(L)
    K_inv= np.dot( np.linalg.inv(L_trans), L_inv)
    
    y = np.array(y)
    ystar_mean = np.dot(np.dot(K_star,K_inv),y)
    ystar_var = np.dot(np.dot(K_star,K_inv),K_star.T)
    
    #Calculo da log likelihood
    n=len(x)
    log_p = -0.5*np.dot(np.dot(y.T,K_inv),y) - sum(np.log(np.diag(L))) - n*0.5*np.log(2*np.pi)
    print(log_p)

elif kernel == Linear:
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(thetab,thetav,c,x1[i],x2[j])
    
    K=K+yerr*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(xstar)):
            K_star[i]=kernel(thetab,thetav,c,x1[i],x2[j])
        
    K_2star=kernel(thetab,thetav,c,K_star,K_star)
    
    # PONTO 3 e 4
    L = np.linalg.cholesky(K)
    L_trans = L.T
    L_inv = np.linalg.inv(L)
    K_inv= np.dot( np.linalg.inv(L_trans), L_inv)
    
    y = np.array(y)
    ystar_mean = np.dot(np.dot(K_star,K_inv),y)
    ystar_var = np.dot(np.dot(K_star,K_inv),K_star.T)
    
    #Calculo da log likelihood
    n=len(x)
    log_p = -0.5*np.dot(np.dot(y.T,K_inv),y) - sum(np.log(np.diag(L))) - n*0.5*np.log(2*np.pi)
    print(log_p)
else:
    print("Qual é  a kernel?")