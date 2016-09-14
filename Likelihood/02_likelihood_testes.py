# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:51:16 2016

@author: camacho
"""
import numpy as np
import matplotlib.pyplot as pl
pl.close("all") #fecha todas as figuras anteriores

import george
from george.kernels import *
from Kernel import *
    
def exemplo_1():
    ########## EXEMPLO 1 - Generate some fake noisy data. ##########
    x = 10 * np.sort(np.random.rand(20))
    yerr = 0.2 * np.ones_like(x)
    y = np.sin(x) + yerr * np.random.randn(len(x))
    
    #### CALCULO USANDO O GEORGE ###
        #Set up the Gaussian process.
    kernel = ExpSquaredKernel(1.0)  #original do exemplo
    #kernel = CosineKernel(1.0)
    gp = george.GP(kernel)
    
        #Pre-compute the factorization of the matrix.
    gp.compute(x,yerr)
    
        #Compute the log likelihood.
    print('george ex1',gp.lnlikelihood(y))
    
    
    #### CALCULO  DA LIKELIHOOD ### 
    #definir kernel a usar
    kernel = ExpSquared 
    ES_theta = 1
    ES_l = 1
    x1=x
    x2=x
    
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(x1[i],x2[j],ES_theta,ES_l)
    K=K+yerr**2*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            K_star[i]=kernel(x1[i],x2[j],ES_theta,ES_l)
        
    K_2star=kernel(K_star,K_star,ES_theta,ES_l) 
    
    #para usar cholesky a matriz tem de ser positiva definida
    L = np.linalg.cholesky(K)
    L_inv= np.linalg.inv(L)
    K_inv= np.dot(L_inv,L.T)
    
    y = np.array(y)
    ystar_mean = np.dot(np.dot(K_star,K_inv),y)
    ystar_var = np.dot(np.dot(K_star,K_inv),K_star.T)
    
    #Calculo da log likelihood
    n=len(x)
    log_p2 = -0.5*np.dot(np.dot(np.dot(y.T,L.T),L_inv),y) - sum(np.log(np.diag(L))) \
                - n*0.5*np.log(2*np.pi)            
    print('ex1',log_p2)

###############################################################################
def exemplo_2():
    ########## EXEMPLO 2 - ExpSineSquared ##########
    x = 50 * np.sort(np.random.rand(100))
    yerr = 0.2 * np.ones_like(x)
    y = np.sin(x) + yerr * np.random.randn(len(x)) + 3
    
    ### CALCULO USANDO O GEORGE ###  
        #Set up the Gaussian process.
    kernelgeorge = 1.5**2*ExpSine2Kernel(2,15)
    #kernel = CosineKernel(1.0)
    gp = george.GP(kernelgeorge)
    
        #Pre-compute the factorization of the matrix.
    gp.compute(x,yerr)
    
        #Compute the log likelihood.
    print('george ex2',gp.lnlikelihood(y))
    
    ### CALCULO  DA LIKELIHOOD ###
    
    #definir kernel a usar
    kernel = ExpSineSquared 
    ESS_theta = 1.5
    ESS_l = 2.0
    ESS_P = 15
    x1=x
    x2=x
    
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(x1[i],x2[j],ESS_theta,ESS_l,ESS_P)
    K=K+yerr**2*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            K_star[i]=kernel(x1[i],x2[j],ESS_theta,ESS_l,ESS_P)
        
    K_2star=kernel(K_star,K_star,ESS_theta,ESS_l,ESS_P)
    
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
    print('ex2',log_p)

###############################################################################
def exemplo_3():
    ########## EXEMPLO 3 - Local_ExpSineSquared ##########
    x = 50 * np.sort(np.random.rand(100))
    yerr = 0.2 * np.ones_like(x)
    y = np.sin(x) + yerr * np.random.randn(len(x)) + 3
    
    ### CALCULO USANDO O GEOREGE ###
        #Set up the Gaussian process.
    kernelgeorge = 1.5**2*ExpSquaredKernel(50**2)*ExpSine2Kernel(2,15)
    #kernel = CosineKernel(1.0)
    gp = george.GP(kernelgeorge)
    
        #Pre-compute the factorization of the matrix.
    gp.compute(x,yerr)
    
        #Compute the log likelihood.
    print('george ex3',gp.lnlikelihood(y))
    
    ### CALCULO  DA LIKELIHOOD ###
    #definir kernel a usar
    kernel = Local_ExpSineSquared 
    LESS_theta = 1.5
    LESS_l = 2.0
    LESS_P = 15
    x1=x
    x2=x
    
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(x1[i],x2[j],LESS_theta,LESS_l,LESS_P)
    K=K+yerr**2*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            K_star[i]=kernel(x1[i],x2[j],LESS_theta,LESS_l,LESS_P)
        
    K_2star=kernel(K_star,K_star,LESS_theta,LESS_l,LESS_P)
    
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
    print('ex3',log_p)

###############################################################################
def exemplo_4():
    ########## EXEMPLO 4 - Soma de ExpSineSquared com ExpSquared ##########
    x = 10 * np.sort(np.random.rand(50))
    yerr = 0.2 * np.ones_like(x)
    y = np.sin(x)**2 + yerr * np.random.randn(len(x))
    
    #### CALCULO USANDO O GEORGE ###
        #Set up the Gaussian process.
    k1 = 10**2*ExpSine2Kernel(2.0/1**2,4)
    k2 = 1**2*ExpSquaredKernel(10.0**2)  #original do exemplo
    kernel = k1+k2
    #kernel = CosineKernel(1.0)
    gp = george.GP(kernel)
    
        #Pre-compute the factorization of the matrix.
    gp.compute(x,yerr)
    
        #Compute the log likelihood.
    print('george ex4',gp.lnlikelihood(y))
    
    
    #### CALCULO  DA LIKELIHOOD ###
    #definir kernel a usar
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
    print('ex4',log_p)

###############################################################################
def exemplo_5():
    ########## EXEMPLO 5 - ExpSquared com white noise ##########
    x = 10 * np.sort(np.random.rand(20))
    yerr = 0.2 * np.ones_like(x)
    y = np.sin(x) + yerr * np.random.randn(len(x))
    
    #### CALCULO USANDO O GEORGE ###
        #Set up the Gaussian process.
    k1 = ExpSquaredKernel(1.0)  #original do exemplo
    k2 = WhiteKernel(2**2)
    kernel = k1+k2
    gp = george.GP(kernel)
    
        #Pre-compute the factorization of the matrix.
    gp.compute(x,yerr)
    
        #Compute the log likelihood.
    print('george ex5',gp.lnlikelihood(y))
    
    
    ##### CALCULO  DA LIKELIHOOD ###
    #definir kernel a usar
    kernel = ExpSquared_WN
    x1=x
    x2=x
    ES_theta = 1
    ES_l = 1
    WN = 2
    
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
    print('ex5',log_p)
 
###############################################################################   

def exemplo_6():
        ########## EXEMPLO 6 - Linear ##########
    np.random.seed(11119)
    x=100 * np.sort(np.random.rand(50))
    yerr = 0.5 * np.ones_like(x)
    y = np.sort(x*np.random.randn(len(x)) + yerr*np.random.randn(len(x)))
    #pl.plot(x,y,'*')
    
    ##### CALCULO  DA LIKELIHOOD ###
    #definir kernel a usar
    kernel = Linear
    x1=x
    x2=x
    L_thetab = 0.62
    L_thetav = 0.1
    L_c=50
    
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(x1[i],x2[j],L_thetab,L_thetav,L_c)
    K=K+yerr**2*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            K_star[i]=kernel(x1[i],x2[j],L_thetab,L_thetav,L_c)
        
    K_2star=kernel(K_star,K_star,L_thetab,L_thetav,L_c)
    
    #para usar cholesky a matriz tem de ser positiva definida
    L = np.linalg.cholesky(K)
    L_inv = np.linalg.inv(L)
    K_inv= np.dot(L_inv,L.T)
    
    y = np.array(y)
    ystar_mean = np.dot(np.dot(K_star,K_inv),y)
    ystar_var = np.dot(np.dot(K_star,K_inv),K_star.T)
    
    #Calculo da log likelihood
    n=len(x)
    log_p = -0.5*np.dot(np.dot(np.dot(y.T,L.T),L_inv),y) - sum(np.log(np.diag(L))) \
            - n*0.5*np.log(2*np.pi)            
    print('ex6',log_p)
    
###############################################################################
def exemplo_7():
    ########## EXEMPLO 7 - ExpSineSquared com white noise ##########
    x = 10 * np.sort(np.random.rand(20))
    yerr = 0.2 * np.ones_like(x)
    y = np.sin(x) + yerr * np.random.randn(len(x))
    
    #### CALCULO USANDO O GEORGE ###
        #Set up the Gaussian process.
    k1 = 10**2*ExpSine2Kernel(2.0/1**2,4)
    k2 = WhiteKernel(2**2)
    kernel = k1+k2
    gp = george.GP(kernel)
    
        #Pre-compute the factorization of the matrix.
    gp.compute(x,yerr)
    
        #Compute the log likelihood.
    print('george ex7',gp.lnlikelihood(y))
    
    
    ##### CALCULO  DA LIKELIHOOD ###
    #definir kernel a usar
    kernel = ExpSineSquared_WN
    x1=x
    x2=x
    ESS_theta = 1.5
    ESS_l = 2.0
    ESS_P = 15
    WN = 2
    
    #calcular matrix de covariancia K, K* e K**
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            K[i,j]=kernel(x1[i],x2[j],ESS_theta,ESS_l,ESS_P,WN)
    K=K+yerr**2*np.identity(len(x))
    K=K+WN**2*np.identity(len(x))      
    
    K_star=np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            K_star[i]=kernel(x1[i],x2[j],ESS_theta,ESS_l,ESS_P,WN)
        
    K_2star=kernel(K_star,K_star,ESS_theta,ESS_l,ESS_P,WN) 
    
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
    print('ex7',log_p)

###############################################################################    
exemplo_1() #ExpSquared
exemplo_2() #ExpSineSquared 
exemplo_3() #Local_ExpSineSquared
exemplo_4() #Soma de ExpSineSquared com ExpSquared
exemplo_5() #ExpSquared com white noise
#exemplo_6() #Linear
exemplo_7()