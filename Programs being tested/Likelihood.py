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
def likelihood(kernel, x, xcalc, y, yerr): #covariance matrix calculations   
    K = np.zeros((len(x),len(x))) #covariance matrix K
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            K[i,j] = kernel(x1, x2)
    K=K+yerr**2*np.identity(len(x))      
    log_p_correct = lnlike(K, y)
    print 'likelihood ->', log_p_correct    
    return K

def lnlike(K, r): #log-likelihood calculations
    from scipy.linalg import cho_factor, cho_solve
    L1 = cho_factor(K) # tuple (L, lower)
    sol = cho_solve(L1, r) # this is K^-1*(r)
    n = r.size
    logLike = -0.5*np.dot(r, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)        
    return logLike


##### LIKELIHOOD GRADIENT
def likelihood_aux(kernel, x, xcalc, y, yerr): #covariance matrix calculations   
    K = np.zeros((len(x),len(x))) #covariance matrix K
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            K[i,j] = kernel(x1, x2)
    K=K+yerr**2*np.identity(len(x))      
    log_p_correct = lnlike(K, y) 
    return K

def grad_logp(kernel,x,xcalc,y,yerr,cov_matrix): #covariance matrix of derivatives
    K_grad = np.zeros((len(x),len(x)))  #covariance matrix K_grad
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            K_grad[i,j] = kernel(x1, x2)
    K_inv = np.linalg.inv(cov_matrix)    
    alpha = np.dot(K_inv,y)
    A = np.outer(alpha, alpha) - K_inv #this comes from george
    grad_george = 0.5 * np.einsum('ij,ij', K_grad, A) #this comes from george
    return grad_george 

def gradient_likelihood(kernel,x,xcalc,y,yerr):
    import inspect
    cov_matrix=likelihood_aux(kernel,x,xcalc,y,yerr)
    if isinstance(kernel,kl.ExpSquared):
        grad1=grad_logp(kernel.dES_dtheta, x, xcalc, y, yerr, cov_matrix)
        grad2=grad_logp(kernel.dES_dl, x, xcalc, y, yerr, cov_matrix)
        print 'gradient ->', grad1, grad2
        return grad1, grad2
    elif isinstance(kernel,kl.ExpSineSquared):
        grad1=grad_logp(kernel.dESS_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dESS_dl,x,xcalc,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dESS_dP,x,xcalc,y,yerr,cov_matrix)
        #print 'gradient ->', grad1, grad2, grad3
        return grad1, grad2, grad3 
    elif isinstance(kernel,kl.RatQuadratic):
        grad1=grad_logp(kernel.dRQ_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dRQ_dalpha,x,xcalc,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dRQ_dl,x,xcalc,y,yerr,cov_matrix)
        print 'gradient ->', grad1, grad2, grad3
        #return grad1, grad2, grad3 
    elif isinstance(kernel,kl.Exponential):
        grad1=grad_logp(kernel.dExp_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dExp_dl,x,xcalc,y,yerr,cov_matrix)
        print 'gradient ->', grad1, grad2
        #return grad1, grad2, grad3 
    elif isinstance(kernel,kl.ExpSineGeorge):
        grad1=grad_logp(kernel.dE_dGamma,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dE_dP,x,xcalc,y,yerr,cov_matrix) 
        print 'gradient ->', grad1, grad2
    elif isinstance(kernel,kl.Matern_32):
        grad1=grad_logp(kernel.dM32_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dM32_dl,x,xcalc,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.Matern_52):
        grad1=grad_logp(kernel.dM52_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dM52_dl,x,xcalc,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.ExpSineGeorge):
        grad1=grad_logp(kernel.dE_dGamma,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dE_dP,x,xcalc,y,yerr,cov_matrix) 
        return grad1, grad2
    elif isinstance(kernel,kl.Sum):
        gradient_sum(kernel,x,xcalc,y,yerr)
    elif isinstance(kernel,kl.Product):
        gradient_mul(kernel,x,xcalc,y,yerr)
                
    else:
        print 'gradient -> NOPE'    


##### LIKELIHOOD GRADIENT FOR SUMS -- SEEMS TO WORK      
def gradient_sum(kernel,x,xcalc,y,yerr):
    from numpy import arange
    kernelOriginal=kernel #para nao perder a original com os calculos todos
    a=kernel.__dict__
    grad_result=[]    
    for i in arange(1,len(kernel.__dict__)+1):
        var = "k%i" %i
        k_i = a[var]
        calc = gradient_likelihood_sum(k_i,x,xcalc,y,yerr,kernelOriginal)
        grad_result.insert(1,calc)
        grad_final  =[]
        for j in range(len(grad_result)):         
           grad_final = grad_final + list(grad_result[j])
    print 'gradient sum ->', grad_final
    return grad_final
    #Devolve NoneType -> acontece se faltar return no gradient_likelihood
            
def gradient_likelihood_sum(kernel,x,xcalc,y,yerr,kernelOriginal):
    import inspect
    cov_matrix=likelihood_aux(kernelOriginal,x,xcalc,y,yerr)
    if isinstance(kernel,kl.ExpSquared):
        grad1=grad_logp(kernel.dES_dtheta, x, xcalc, y, yerr, cov_matrix)
        grad2=grad_logp(kernel.dES_dl, x, xcalc, y, yerr, cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.ExpSineSquared):
        grad1=grad_logp(kernel.dESS_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dESS_dl,x,xcalc,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dESS_dP,x,xcalc,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
    elif isinstance(kernel,kl.RatQuadratic):
        grad1=grad_logp(kernel.dRQ_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dRQ_dalpha,x,xcalc,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dRQ_dl,x,xcalc,y,yerr,cov_matrix)
        return grad1, grad2, grad3 
    elif isinstance(kernel,kl.Exponential):
        grad1=grad_logp(kernel.dExp_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dExp_dl,x,xcalc,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.Matern_32):
        grad1=grad_logp(kernel.dM32_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dM32_dl,x,xcalc,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.Matern_52):
        grad1=grad_logp(kernel.dM52_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dM52_dl,x,xcalc,y,yerr,cov_matrix)
        return grad1, grad2
    elif isinstance(kernel,kl.ExpSineGeorge):
        grad1=grad_logp(kernel.dE_dGamma,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dE_dP,x,xcalc,y,yerr,cov_matrix) 
        return grad1, grad2                
    else:
        print 'gradient -> NOPE'


##### LIKELIHOOD GRADIENT FOR PRODUCTS -- SEEMS TO WORK       
def gradient_mul(kernel,x,xcalc,y,yerr):
    #print 'Work in progress'
    from numpy import arange
    kernelOriginal=kernel #para nao perder a original com os calculos todos
    cov_matrix=likelihood_aux(kernelOriginal,x,xcalc,y,yerr) #matrix cov original
    a=kernel.__dict__
    len_dict=len(kernel.__dict__)
    grad_result=[] #para aqui irao os gradientes finais no fim
    kernelaux1=[] #para meter as kernels
    kernelaux2=[] #para meter as derivadas das kernels
    for i in arange(1,len_dict+1):
        var = "k%i"%i; #ki = a[var]
        kernelaux1.append(a[var])
        kernelaux2.append(kernel_deriv(a[var]))
    #print  kernelaux1; print kernelaux2
    A1=len(kernelaux1) #quantas kernels temos
    B1=len(kernelaux2) #quantos grupos de derivadas temos => A1=B1
    #print len(kernelaux2[1])
    for i1 in range(A1):
        #print 'i1=',i1
        for i2 in range(B1):
            #print 'i2=',i2
            if i1==i2:
                pass
            else:
                 B2=len(kernelaux2[i2])
                 for j in range(B2):
                     #print 'j=',j
                     result=grad_logp(kernelaux1[i1]*kernelaux2[i2][j],x,xcalc,y,yerr,cov_matrix)
                     grad_result.insert(0,result)
    print 'gradient ->', grad_result    
    return grad_result
   
           
def kernel_deriv(kernel):
    import inspect
    if isinstance(kernel,kl.ExpSquared):
        return kernel.dES_dtheta, kernel.dES_dl
    elif isinstance(kernel,kl.ExpSineSquared):
        return kernel.dESS_dtheta, kernel.dESS_dl, kernel.dESS_dP
    elif isinstance(kernel,kl.ExpSineGeorge):
        return kernel.dE_dGamma, kernel.dE_dP
    else:
        print 'Faltam mais elif com  as outras kernels!'
        
        
        
##### OLD STUFF KEPT JUST IN CASE #############################################
#def gradient_mul(kernel,x,xcalc,y,yerr):
#    print 'Work in progress'
#    from numpy import arange
#    kernelOriginal=kernel #para nao perder a original com os calculos todos
#    cov_matrix=likelihood_aux(kernelOriginal,x,xcalc,y,yerr) #matrix cov original
#    a=kernel.__dict__; len_dict=len(kernel.__dict__)
#    grad_result=[]; kernel_result=[]
#    for i in arange(1,len_dict+1):
#        var = "k%i"%i; ki = a[var]
#        j=i+i; k=i+i+1        
#        Aj=[];Aj.insert(1,ki)
#        print j,'=', Aj
#        Ak=[];Ak.insert(1,kernel_identify(ki))
#        print k,'=', Ak
#        #Ai=[]; Ai.insert(1,ki)
#        #print 'A%i ='%i, Ai
#        #return Ai        
#        #ai=[]; ai.insert(1,kernel_identify(ki))
#        #print 'a%i ='%i, ai
#        #return ai
#    #listas= listmaker(kernel)    
#    #print listas    
##    for j in arange(1,len_dict+1):
##        for k in arange(1,len_dict+1):
##            if j!=k:
##                #NewKernel=a[j]*A[k]
##                print 'j', j, 'k', k
##                #NKernel=A%j+a%k
##                #print Nkernel
#                #print Aj                
#  
##def listmaker(kernel):
##    from numpy import arange
##    a=kernel.__dict__; len_dict=len(kernel.__dict__)    
##    for i in arange(1,len_dict+1):
##        var = "k%i"%i; ki = a[var]
##        Ai=[]; Ai.insert(1,ki)
##        print 'A%i ='%i, Ai
##        #return Ai        
##        ai=[]; ai.insert(1,kernel_identify(ki))
##        return Ai, ai