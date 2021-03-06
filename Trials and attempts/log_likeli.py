# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:06:31 2016

@author: camacho
"""
#   Só foi testado com uma kernel(ExpSquared) mas comparando os valores obtidos por este script
#e os valores obtidos pelo george usando o lnlikelihood, o george até agora é melhor.

import numpy as np
import sympy as  sp
from Kernel import *
import matplotlib.pyplot as pl
#pl.close("all") #fecha todas as figuras abertas anteriormente

########## Dados Iniciais ##########
# Retirado do 01_Exemplo_simples
#np.random.seed(10)  #Generate some fake noisy data.
#x = 10 * np.sort(np.random.rand(20))
#yerr = 0.2 * np.ones_like(x)
#y = np.sin(x) + yerr * np.random.randn(len(x))

x =[-1.5, -1, -0.75, -0.4, -0.25, 0]
y = [0.55*-3, 0.55*-2, 0.55*-0.6, 0.55*0.4, 0.55*1, 0.55*1.6]
yerr=0.3 * np.ones_like(x)

#pl.plot(x,y,"*")

########## definir kernel a usar e parametros ##########
kernel = ExpSquared
theta = 1
l = 1
x1=x
x2=x

########## Calculos da log likelihood ##########
if kernel is ExpSquared:
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(x1[i],x2[j],theta,l)
    K=K+yerr**2*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            K_star[i]=kernel(x1[i],x2[j],theta,l) 
        
    K_2star=kernel(K_star,K_star,theta,l,) 
    
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
##########
elif kernel is ExpSineSquared or kernel is Local_ExpSineSquared:
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(x1[i],x2[j],theta,l,P)
    K=K+yerr**2*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            K_star[i]=kernel(x1[i],x2[j],theta,l,P)
        
    K_2star=kernel(K_star,K_star,theta,l,P)
    
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
##########
elif kernel is RatQuadratic:   
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(x1[i],x2[j],theta,l,alpha)
    K=K+yerr**2*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            K_star[i]=kernel(x1[i],x2[j],theta,l,alpha)
        
    K_2star=kernel(K_star,K_star,theta,l,alpha)
    
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
##########
#elif kernel is Linear:
#    #calcular matrix de covariancia K, K* e K**
#    K=np.zeros((len(x),len(x)))
#    for i in range(len(x)):
#        for j in range(len(x)):
#            K[i,j]=kernel(x1[i],x2[j],thetab,thetav,c)
#    K=K+yerr**2*np.identity(len(x))      
#    
#    K_star=np.zeros(len(x))
#    for i in range(len(x)):
#        for j in range(len(x)):
#            K_star[i]=kernel(x1[i],x2[j],thetab,thetav,c)
#        
#    K_2star=kernel(K_star,K_star,thetab,thetav,c)
#    
#    #para usar cholesky a matriz tem de ser positiva definida
#    L = np.linalg.cholesky(K)
#    L_inv = np.linalg.inv(L)
#    K_inv= np.dot(L_inv,L.T)
#    
#    y = np.array(y)
#    ystar_mean = np.dot(np.dot(K_star,K_inv),y)
#    ystar_var = np.dot(np.dot(K_star,K_inv),K_star.T)
#    
#    #Calculo da log likelihood
#    n=len(x)
#    log_p = -0.5*np.dot(np.dot(np.dot(y.T,L.T),L_inv),y) - sum(np.log(np.diag(L))) \
#            - n*0.5*np.log(2*np.pi)            
#    print(log_p)
##########
elif kernel is Sum_ExpSineSquared_ExpSquared or kernel is Mul_ExpSineSquared_ExpSquared:
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
##########
elif kernel is ExpSquared_WN:
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(x1[i],x2[j],ES_theta,ES_l,WN)
    K=K+yerr**2*np.identity(len(x))
    K=K+WN**2*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            K_star[i]=kernel(x1[i],x2[j],ES_theta,ES_l,WN)
        
    K_2star=kernel(K_star,K_star,ES_theta,ES_l,WN) 
    
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

##########    
else:
    print("Qual é a kernel?")