# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:42:48 2016

@author: camacho
"""
import numpy as np
from sympy import KroneckerDelta as kd
from time import time   
 
##### KERNELS 
class Kernel(object):
    def __init__(self, *args):
        self.pars = np.array(args) # put all Kernel arguments in an array pars

    def __call__(self, x1, x2):
        raise NotImplementedError
        #return self.k1(x1, x2, i, j) * self.k2(x1, x2, i, j)

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

    def __call__(self, x1, x2):
        return self.k1(x1, x2) + self.k2(x1, x2)

    #def dcall(self, parnum ...)
        #self.pars.size numero de param da soma

class Product(_operator): #multiplication of kernels
    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)
        
    def __call__(self, x1, x2):
        return self.k1(x1, x2) * self.k2(x1, x2)


class ExpSquared(Kernel):
    def __init__(self, ES_theta, ES_l):
        super(ExpSquared, self).__init__(ES_theta, ES_l)
        # because we are "overwriting" the function __init__
        # we use this weird super function
        
        self.ES_theta = ES_theta
        self.ES_l = ES_l
        
    def __call__(self, x1, x2):
        f1 = self.ES_theta**2
        f2 = self.ES_l**2
        f3 = (x1 - x2)**2
        return f1 * np.exp(-0.5 * f3 / f2)
         
class ExpSineSquared(Kernel):
    def __init__(self, ESS_theta, ESS_l, ESS_P):
        super(ExpSineSquared, self).__init__(ESS_theta, ESS_l, ESS_P)

        self.ESS_theta = ESS_theta
        self.ESS_l = ESS_l
        self.ESS_P = ESS_P
    
    def __call__(self, x1, x2):
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
    
    def __call__(self, x1, x2):
        f1 = self.RQ_theta**2
        f2 = self.RQ_l**2
        f3 = (x1-x2)**2
        f4 = self.RQ_alpha
        return f1*(1+(0.5*f3/(f4*f2)))**(-f4)
        
class WhiteNoise(Kernel):                             #In case the white noise
    def __init__(self,WN_theta):                      #is proved to be wrong
        super(WhiteNoise,self).__init__(WN_theta)     #it will be necessary
        self.WN_theta=WN_theta                        #to remove i and j from
                                                      #all classes __call__
    def __call__(self, x1, x2):
        f1=self.WN_theta**2                           #É extremamente lento
        #f2=(x1-x2)     
        #f3=kd(i,j)
        return f1*np.diag(np.ones_like(x1))

class Exponential(Kernel): #Matern 1/2 = Exponential
    def __init__(self,Exp_theta,Exp_l):
        super(Exponential,self).__init__(Exp_theta,Exp_l)
        self.Exp_theta=Exp_theta        
        self.Exp_l=Exp_l

    def __call__(self, x1,  x2):
        f1=x1-x2
        f2=self.Exp_l
        f3=self.Exp_theta**2
        return f3*np.exp(-f1/f2)

class Matern_32(Kernel): #Matern 3/2
    def __init__(self,M32_theta,M32_l):
        super(Matern_32,self).__init__(M32_theta,M32_l)
        self.M32_theta=M32_theta   
        self.M32_l=M32_l  
        
    def __call__(self, x1, x2):
        f1=np.sqrt(3.0)*(x1-x2)
        f2=self.M32_l
        f3=self.M32_theta**2
        return f3*(1.0 + f1/f2)*np.exp(-f1/f2)
        
class Matern_52(Kernel): #Matern 5/2
    def __init__(self,M52_l):
        super(Matern_52,self).__init__(M52_theta,M52_l)
        self.M52_theta=M52_theta        
        self.M52_l=M52_l

    def __call__(self, x1, x2):
        f1=np.sqrt(5.0)*(x1-x2)
        f2=(x1-x2)**2        
        f3=self.M52_l
        f4=self.M52_l**2
        f5=self.M52_theta**2
        return f5*(1.0 + f1/f3 + (5.0*f4)/(3.0*f4))*np.exp(-f1/f3)
    
    #def dcall(self,...):
    #    return #a expressao da derivada
    
    
#Kernels derivadas 
class dExpSquared_dtheta(Kernel): #derivative in order to ES_theta
    def __init__(self, ES_theta, ES_l):
        super(dExpSquared_dtheta,self).__init__(ES_theta, ES_l)
        
        self.ES_theta = ES_theta
        self.ES_l = ES_l

    def __call__(self,x1,x2):
        f1=self.ES_theta  #2*theta        
        f2=(x1-x2)**2       #(x1-x2)**2
        f3=self.ES_l**2     #l**2
        return  2.0*f1*np.exp(-0.5*f2/f3)        
    
class dExpSquared_dl(Kernel): #derivative in order to ES_l
    def __init__(self, ES_theta, ES_l):
        super(dExpSquared_dl,self).__init__(ES_theta, ES_l)
        
        self.ES_theta = ES_theta
        self.ES_l = ES_l       

    def __call__(self,x1,x2):
        f1=self.ES_theta**2     #theta**2
        f2=(x1-x2)**2           #(x1-x2)**2
        f3=self.ES_l**3         #l**3
        f4=self.ES_l**2         #l**2
        return f1* (f2/f3)*  np.exp(-0.5 * f2/f4) 

class dExpSineSquared_dtheta(Kernel): #derivative in order of ESS_theta
    def __init__(self, ESS_theta, ESS_l, ESS_P):
        super(dExpSineSquared_dtheta, self).__init__(ESS_theta, ESS_l, ESS_P)

        self.ESS_theta = ESS_theta
        self.ESS_l = ESS_l
        self.ESS_P = ESS_P
        
    def __call__(self,x1,x2):
        f1 = self.ESS_theta *2  #2*theta
        f2 = self.ESS_l**2      #l**2 
        f3 = np.pi/self.ESS_P   #
        f4 = x1-x2
        return f1*np.exp(-(2.0/f2)*np.sin(f3*f4))

class dExpSineSquared_dl(Kernel): #derivative in order of ESS_l
    def __init__(self, ESS_theta, ESS_l, ESS_P):
        super(dExpSineSquared_dl, self).__init__(ESS_theta, ESS_l, ESS_P)

        self.ESS_theta = ESS_theta
        self.ESS_l = ESS_l
        self.ESS_P = ESS_P
        
    def __call__(self,x1,x2):
        f1=4* self.ESS_theta**2
        f2=self.ESS_l**3
        f3=np.pi/self.ESS_P
        f4=x1-x2
        f5=self.ESS_l**2
        return (f1*np.sin(f3*f4)/f2) * np.exp(-2*np.sin(f3*f4)/f5)
        
class dExpSineSquared_dP(Kernel): #derivative in order of ESS_P
    def __init__(self, ESS_theta, ESS_l, ESS_P):
        super(dExpSineSquared_dP, self).__init__(ESS_theta, ESS_l, ESS_P)

        self.ESS_theta = ESS_theta
        self.ESS_l = ESS_l
        self.ESS_P = ESS_P
        
    def __call__(self,x1,x2):
        f1=2*np.pi*self.ESS_theta**2
        f2=self.ESS_l**2
        f3=np.pi/self.ESS_P
        f4=self.ESS_P**2
        f5=x1-x2
        return (f1/(f2*f4)) * f5*np.cos(f3*f5) * np.exp(-2*np.sin(f3*f5)/f2) 
            
class dRatQuadratic_dtheta(Kernel): #derivative in order of RQ_theta
    def __init__(self,RQ_theta,RQ_l,RQ_alpha):
        super(dRatQuadratic_dtheta, self).__init__(RQ_theta, RQ_l, RQ_alpha)
        self.RQ_theta = RQ_theta
        self.RQ_l = RQ_l
        self.RQ_alpha = RQ_alpha           
            
    def __call__(self,x1,x2):
        f1=self.RQ_theta*2
        f2=(x1-x2)**2
        f3=self.RQ_alpha
        f4=self.RQ_l**2
        return f1*(1 + 0.5*f2/(f3*f4))**(-f3)

class dRatQuadratic_dl(Kernel): #derivative in order of RQ_l
    def __init__(self,RQ_theta,RQ_l,RQ_alpha):
        super(dRatQuadratic_dl, self).__init__(RQ_theta, RQ_l, RQ_alpha)
        self.RQ_theta = RQ_theta
        self.RQ_l = RQ_l
        self.RQ_alpha = RQ_alpha      
            
    def __call__(self,x1,x2):
        f1=self.RQ_theta**2
        f2=(x1-x2)**2
        f3=self.RQ_alpha
        f4=self.RQ_l**2
        f5=self.RQ_l**3
        return (f1*f2/f5)*(1 + 0.5*f2/(f3*f4))**(-1-f3)
 
class dRatQuadratic_dalpha(Kernel): #derivative in order of RQ_alpha
    def __init__(self,RQ_theta,RQ_l,RQ_alpha):
        super(dRatQuadratic_dalpha, self).__init__(RQ_theta, RQ_l, RQ_alpha)
        self.RQ_theta = RQ_theta
        self.RQ_l = RQ_l
        self.RQ_alpha = RQ_alpha           
            
    def __call__(self,x1,x2):
        f1=self.RQ_theta*2
        f2=(x1-x2)**2
        f3=self.RQ_alpha
        f4=self.RQ_l**2
        func1=0.5*f2/(f3*f4*(1 + 0.5*f2/(f3*f4)))
        func2=1+ 0.5*f2/(f3*f4)
        return f1*(func1 - np.log(func2)) * func2**(-f3)        
    
        


        
        
