# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:42:48 2016

@author: camacho
"""
import numpy as np
from sympy import KroneckerDelta as kd
from time import time   
import inspect as i
 
##### KERNELS 
class Kernel(object):
    def __init__(self, *args):
        # put all Kernel arguments in an array pars
        self.pars = np.array(args)

    def __call__(self, x1, x2, i, j):
        raise NotImplementedError

    def __add__(self, b):
        return Sum(self, b)
    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        return Product(self, b)
    def __rmul__(self, b):
        return self.__mul__(b)

    def __repr__(self):
        """ Representation of each Kernel instance """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))

class _operator(Kernel):
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    @property
    def pars(self):
        return np.append(self.k1.pars, self.k2.pars)

class Sum(_operator): #sum of kernels
    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)

    def __call__(self, x1, x2, i, j):
        return self.k1(x1, x2, i, j) + self.k2(x1, x2, i, j)

class Product(_operator): #multiplication of kernels
    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)
        
    def __call__(self, x1, x2, i, j):
        return self.k1(x1, x2, i, j) * self.k2(x1, x2, i, j)


class ExpSquared(Kernel):
    def __init__(self, ES_theta, ES_l):
        # because we are "overwriting" the function __init__
        # we use this weird super function
        super(ExpSquared, self).__init__(ES_theta, ES_l)
        
        self.ES_theta = ES_theta
        self.ES_l = ES_l

    def __call__(self, x1, x2, i, j):
        f1 = self.ES_theta**2
        f2 = self.ES_l**2
        f3 = (x1 - x2)**2
        return f1 * np.exp(-0.5 * f3 / f2)
        
class ExpSquared_l(Kernel): #derivada em ordem a ES_l
    def __init__(self, ES_theta, ES_l):
        super(ExpSquared_l, self).__init__(ES_theta, ES_l)

        self.ES_theta = ES_theta
        self.ES_l = ES_l       

    def __call__(self,x1,x2,i,j):
        f1=ES_theta**2
        f2=(x1-x2)**2
        f3=ES_l**3
        f4=ES_l**2
        return f1* (f2/f3)*  np.exp(-0.5 * f2/f4)
 
class ExpSineSquared(Kernel):
    def __init__(self, ESS_theta, ESS_l, ESS_P):
        super(ExpSineSquared, self).__init__(ESS_theta, ESS_l, ESS_P)

        self.ESS_theta = ESS_theta
        self.ESS_l = ESS_l
        self.ESS_P = ESS_P
    
    def __call__(self, x1, x2, i, j):
        f1 = self.ESS_theta**2
        f2 = self.ESS_l**2
        f3 = (x1-x2)
        f4 = self.ESS_P
        return f1*np.exp(-2*(np.sin(np.pi*f3/f4))**2/f2)     

class RatQuadratic(Kernel):
    def __init__(self, RQ_theta, RQ_l, RQ_alpha):
        super(RatQuadratic, self).__init__(RQ_theta, RQ_l, RQ_alpha)
        self.RQ_theta = RQ_theta
        self.RQ_l = RQ_l
        self.RQ_alpha = RQ_alpha
    
    def __call__(self, x1, x2, i, j):
        f1 = self.RQ_theta**2
        f2 = self.RQ_l**2
        f3 = (x1-x2)**2
        f4 = self.RQ_alpha
        return f1*(1+(0.5*f3/(f4*f2)))**(-f4)
        
class Local_ExpSineSquared(Kernel): #equal to ExpSquared*ExpSineSquared
    def __init__(self, LESS_theta, LESS_l, LESS_P):
        super(Local_ExpSineSquared, self).__init__(LESS_theta, LESS_l, LESS_P)
        self.LESS_theta = LESS_theta
        self.LESS_l = LESS_l
        self.LESS_P = LESS_P
        
    def __call__(self, x1 ,x2 , i,  j):
        f1 = self.LESS_theta**2                               
        f2 = self.LESS_l**2
        f3 = (x1-x2)
        f4 = (x1-x2)**2
        f5 = self.LESS_P
        return f1*np.exp(-2*(np.sin(np.pi*f3/f5))**2/f2)*np.exp(-0.5*f4/f2)        

class WhiteNoise(Kernel):
    def __init__(self,WN_theta):
        super(WhiteNoise,self).__init__(WN_theta)
        self.WN_theta=WN_theta
    
    def __call__(self, x1, x2, i, j):
        f1=self.WN_theta**2
        #f2=(x1-x2)     
        f3=kd(i,j)
        return f1*f3
# In case the white noise is proved to be wrong, it will be necessary
#to remove i and j from all  classes __call__
        
class Exponential(Kernel): #Matern 1/2 = Exponential
    def __init__(self,Exp_theta,Exp_l):
        super(Exponential,self).__init__(Exp_theta,Exp_l)
        self.Exp_theta=Exp_theta        
        self.Exp_l=Exp_l

    def __call__(self, x1,  x2, i, j):
        f1=x1-x2
        f2=self.Exp_l
        f3=self.Exp_theta**2
        return f3*np.exp(-f1/f2)

class Matern_32(Kernel): #Matern 3/2
    def __init__(self,M32_theta,M32_l):
        super(Matern_32,self).__init__(M32_theta,M32_l)
        self.M32_theta=M32_theta   
        self.M32_l=M32_l  
        
    def __call__(self, x1, x2, i, j):
        f1=np.sqrt(3.0)*(x1-x2)
        f2=self.M32_l
        f3=self.M32_theta**2
        return f3*(1.0 + f1/f2)*np.exp(-f1/f2)
        
class Matern_52(Kernel): #Matern 5/2
    def __init__(self,M52_l):
        super(Matern_52,self).__init__(M52_theta,M52_l)
        self.M52_theta=M52_theta        
        self.M52_l=M52_l

    def __call__(self, x1, x2, i, j):
        f1=np.sqrt(5.0)*(x1-x2)
        f2=(x1-x2)**2        
        f3=self.M52_l
        f4=self.M52_l**2
        f5=self.M52_theta**2
        return f5*(1.0 + f1/f3 + (5.0*f4)/(3.0*f4))*np.exp(-f1/f3)



     
##### LIKELIHOOD
def lnlike(K, r): #log-likelihood calculations
    from scipy.linalg import cho_factor, cho_solve
    L1 = cho_factor(K)  # tuple (L, lower)
    # this is K^-1*(r)
    sol = cho_solve(L1, r)
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
        i=i
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            j=j
            K[i,j] = kernel(x1, x2, i, j)
    K=K+yerr**2*np.identity(len(x))      
    log_p_correct = lnlike(K, y)
    print 'Took %f seconds' % (time() - start), ('log_p_correct',log_p_correct)    
    return K


##### LIKELIHOOD GRADIENT
def variables(kernel):   #devolve as variaveis da kernel, a rever que bate mal
    return [i for i in kernel.__dict__.keys() if i[:1] != '_']

def gradient(K_grad,r):
    from scipy.linalg import cho_factor, cho_solve
    L1 = cho_factor(K_grad)  # tuple (L, lower)
    # this is K^-1*(r)
    sol = cho_solve(L1, r)
    grad_likelihood = 0.5*np.trace(np.dot((np.dot(sol,sol.T)-np.linalg.inv(K)),K_grad))
    return grad_likelihood
    
def grad_log_p(kernel,x,xcalc,y,yerr):
    if kernel is ExpSquared:
        kernel = ExpSquared_l(variables(kernel))
        K_grad = np.zeros((len(x),len(x))) #covariance matrix K
        for i in range(len(x)):
            x1 = x[i]
            i=i
            for j in range(len(xcalc)):                      
                x2 = xcalc[j]
                j=j
                K_grad[i,j] = kernel(x1, x2, i, j)
        K_grad=K_grad+yerr**2*np.identity(len(x)) 
        gradient_logp = gradient(K_grad,y)
    print gradient_logp
    
    
    
    
    
    
    
    
###### A PARTIR DAQUI ACHO QUE NÃO É NECESSARIO MAS DEIXO FICAR NA MESMA ######
    
#def likelihood(kernel, x, xcalc, y, yerr): #covariance matrix calculations
#    start = time() #Corrected and faster version    
#    K = np.zeros((len(x),len(x))) #covariance matrix K
#    for i in range(len(x)):
#        x1 = x[i]
#        i=i
#        for j in range(len(xcalc)):                      
#            x2 = xcalc[j]
#            j=j
#            K[i,j] = kernel(x1, x2, i, j)
#    K=K+yerr**2*np.identity(len(x))     
#    start = time() #Original and slower version
#    #para usar cholesky a matriz tem de ser positiva definida
#    #L=sp.linalg.lu(K)
#    L = np.linalg.cholesky(K)
#    L_inv= np.linalg.inv(L)
#    y = np.array(y)
#    #Calculo da log likelihood
#    n=len(x)
#    log_p = -0.5*np.dot(np.dot(np.dot(y.T,L.T),L_inv),y) \
#            - sum(np.log(np.diag(L))) \
#            - n*0.5*np.log(2*np.pi)            
#    print 'Took %f seconds' % (time() - start), ('log_p',log_p)    
    
###### SUM and MULTIPLICATION #####
#class Combinable:
#    def __init__(self, f):
#        self.f = f
#
#    def __call__(self, *args):
#        return self.f(args)
#
#    def __add__(self, g):
#        return Sum_Ker(self.f,g)    
#        
#    def __mul__(self, g):
#        return Mul_Ker(self.f, g)
#
#class Sum_Kernel(Combinable): # naofunciona
#    def __init__(self, f, g):
#        self.f = f
#        self.g = g
#    
#    def __call__(self, *args):
#        #arranjar maneira de adicionar args[i]  consoante o valor de i
#        return self.f(args[0],args[1]) + self.g(args[2],args[3],args[4])
#        
#class Mul_Kernel(Combinable): # nao funciona
#    def __init__(self, f, g):
#        self.f = f
#        self.g = g
#    
#    def __call__(self, *args):
#        return self.f(args[0],args[1]) * self.g(args[2],args[3],args[4])

###### TESTES #####
#likelihood(ExpSquared(19, 2), x, x, y, yerr)
## Calculo usando o george 
#import george
#from george.kernels import ExpSquaredKernel
#start = time()
#kernel = ExpSquaredKernel(1.0)
#gp = george.GP(kernel)
#gp.compute(x,yerr)
#print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))
    
#likelihood(ExpSineSquared(19,2,5),x,x,y,yerr)
    
    #assert np.allclose(log_p,log_p_correct)
##### CONCLUSOES
#A minha versão demora cerca de 8 vezes mais

##### Kernels  criadas ########################################################
#def ExpSquared(x1, x2, *params): # Squared Exponential Kernel   
#    ES_theta, ES_l = params    
#    f1 = ES_theta**2
#    f2 = ES_l**2
#    f3 = (x1 - x2)**2
#    return f1 * np.exp(-0.5 * f3 / f2)
#
#def ExpSineSquared(x1, x2, *params): # Periodic Kernel
#    ESS_theta, ESS_l, ESS_P = params 
#    f1 = ESS_theta**2
#    f2 = ESS_l**2
#    f3 = (x1-x2)
#    return f1*np.exp(-2*(np.sin(np.pi*f3/ESS_P))**2/f2)
#       
#def RatQuadratic(x1, x2, *params): # Rational Quadratic Kernel
#    RQ_theta, RQ_l, RQ_alpha = params
#    f1 = RQ_theta**2
#    f2 = RQ_l**2
#    f3 = (x1-x2)**2
#    return f1*(1+(0.5*f3/(RQ_alpha*f2)))**(-RQ_alpha)
#
#def Local_ExpSineSquared(x1, x2, *params):             #Locally Periodic Kernel
#    LESS_theta, LESS_l, LESS_P = params                       #identico a fazer
#    f1 = LESS_theta**2                               #ExpSineSquared*ExpSquared
#    f2 = LESS_l**2
#    f3 = (x1-x2)
#    f4 = (x1-x2)**2
#    return f1*np.exp(-2*(np.sin(np.pi*f3/LESS_P))**2/f2)*np.exp(-0.5*f4/f2)

#i e j = 0 deve calcular mal
#i=0;j=0 #Se não definir nenhum valor inicial dá erro no calculo da likelihood
#def WhiteNoise(WN_theta): # White Noise Kernel
#    return (WN_theta**2)*kd(i,j)#*(x1-x2)

## Linear Kernel
#def Linear(x1, x2,L_thetab,L_thetav,L_c):
#    f1 = L_thetab**2
#    f2 = L_thetav**2
#    return f1+f2*(x1-L_c)*(x2-L_c)
#    
##Soma de Periodic com Squared Exponential (ExpSineSquared+ExpSquared)
#def Sum_ExpSineSquared_ExpSquared(x1, x2, ESS_theta, ESS_l, ESS_P, ES_theta, ES_l):
#    return ExpSineSquared(x1,x2,ESS_theta,ESS_l,ESS_P)+ExpSquared(x1,x2,ES_theta,ES_l)
#    
## Squared Exponential Kernel com white noise
#def ExpSquared_WN(x1, x2, ES_theta, ES_l, WN):
#    f1 = ES_theta**2
#    f2 = ES_l**2
#    f3 = (x1-x2)**2
#    return f1*np.exp(-0.5*f3/f2)
#    
## Periodic Kernel com white noise
#def ExpSineSquared_WN(x1, x2, ESS_theta, ESS_l, ESS_P,WN): 
#    f1 = ESS_theta**2
#    f2 = ESS_l**2
#    f3 = (x1-x2)
#    return f1*np.exp(-2*(np.sin(np.pi*f3/ESS_P))**2/f2)



