# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:27:49 2016

@author: camacho
"""
import Kernel;reload(Kernel);kl = Kernel
import numpy as np
from time import time   
import inspect as i

##### LIKELIHOOD
def lnlike(K, r): #log-likelihood calculations
    from scipy.linalg import cho_factor, cho_solve
    L1 = cho_factor(K) # tuple (L, lower)
    sol = cho_solve(L1, r) # this is K^-1*(r)
    n = r.size
    logLike = -0.5*np.dot(r, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)
    return logLike

def likelihood(kernel, x, xcalc, y, yerr): #covariance matrix calculations
    start = time() #Corrected and faster version    
    K = np.zeros((len(x),len(x))) #covariance matrix K
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]

            K[i,j] = kernel(x1, x2)
    K=K+yerr**2*np.identity(len(x))      
    log_p_correct = lnlike(K, y)
    print 'Took %f seconds' % (time() - start), ('log_p_correct',log_p_correct)    
    return K


##### LIKELIHOOD GRADIENT
def variables(kernel): #devolve o valor das variaveis da kernel usada
    return [i for i in kernel.pars[:]] 
 
def grad_logp(kernel,x,xcalc,y,yerr,cov_matrix):
    K_grad = np.zeros((len(x),len(x))) 
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            K_grad[i,j] = kernel(x1, x2)
    #K_grad=K_grad+yerr**2*np.identity(len(x))#sinceramente não sei se esta linha faz sentido neste calculo
        
    K_inv = np.linalg.inv(cov_matrix)    
    alpha = np.dot(K_inv,y)
    alpha_trans = alpha.T
    
#formulas do gradiente tiradas do Rasmussen&Williams chapter 5, equaçao(5.9)
       
    #os return grad comentados dao valores absurdos e não sei o porquê
    grad = 0.5 * np.dot(y.T,np.dot(K_inv,np.dot(K_grad,np.dot(K_inv,y)))) \
            -0.5 * np.einsum('ij,ij',K_inv,K_grad)   
            #-0.5*np.trace(np.dot(K_inv,K_grad))
    return K_grad
#    calc_aux=np.dot(alpha,alpha_trans)-K_inv        
#    calc_aux = (alpha * alpha_trans) - K_inv
#    grad_outro = 0.5 * np.trace(np.dot(calc_aux,K_grad))
#    grad_adaptgeorge = 0.5 * np.einsum('ij,ij',calc_aux,K_grad) #adaptado do george
#    return K_grad
  
#    return grad_outro
#    return grad_adaptgeorge
#    return np.dot(calc_aux,K_grad)

def gradient_likelihood(kernel,x,xcalc,y,yerr):
    import inspect
    cov_matrix=likelihood(kernel,x,xcalc,y,yerr)#ele volta a imprimir a likelihood acho que 
                                                #por causa disto mas preciso da matriz de 
                                                #covariancia original
    if isinstance(kernel,kl.ExpSquared):
        a=variables(kernel)[0] #devolve os valores de theta 
        b=variables(kernel)[1] # e de l
        grad1=grad_logp(kl.dExpSquared_dtheta(a,b),x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kl.dExpSquared_dl(a,b),x,xcalc,y,yerr,cov_matrix)
        print 'gradient ->', grad1
    elif isinstance(kernel,kl.ExpSineSquared) is True:
        a=variables(kernel)[0] #devolve os valores de theta
        b=variables(kernel)[1] #de l
        c=variables(kernel)[2] # e de P
        grad1=grad_logp(kl.dExpSineSquared_dtheta(a,b,c),x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kl.dExpSineSquared_dl(a,b,c),x,xcalc,y,yerr,cov_matrix)
        grad3=grad_logp(kl.dExpSineSquared_dP(a,b,c),x,xcalc,y,yerr,cov_matrix)
        print 'gradient ->', grad1, grad2, grad3    
#    elif isinstance(kernel,RatQuadratic) is True:
#        a=variables(kernel)[0] #devolve os valores de theta
#        b=variables(kernel)[1] #de l
#        c=variables(kernel)[2] # e de alpha
#        grad1=grad_logp(dRatQuadratic_dtheta(a,b,c),x,xcalc,y,yerr,cov_matrix)
#        grad2=grad_logp(dRatQuadratic_dl(a,b,c),x,xcalc,y,yerr,cov_matrix)
#        grad3=grad_logp(dRatQuadratic_dalpha(a,b,c),x,xcalc,y,yerr,cov_matrix)
#        print 'gradient ->', grad1, grad2, grad3    
#    elif isinstance(kernel,Sum) is True:
#        initial=kernel.__dict__
#        for i in range(len(initial)):
#            cond_i = initial.popitem()
#            print 'gradient -> Olha olha é uma soma com', cond_i 
#    
    else:
        print 'gradient -> no calculation'    
    #   Nao apliquei a mesma logica às kernels exponential e matern pois
    #até isto funcionar como deve ser não vale a pena fazer
    #funcionar como deve ser = saber se estou a calcular o gradiente bem
    #e arranjar maneira de isto funcionar com somas e multiplicaçoes de kernels