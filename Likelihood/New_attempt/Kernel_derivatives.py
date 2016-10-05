# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

#IGNORAR  POIS NAO TEM UTILIDADE
#   Sao as derivadas das kernels e como nao me queria ver livre delas
#agora que o script está escrito deixo ficar aqui

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

    
#Derivadas 
class dExpSquared_dtheta(Kernel): #derivadada em ordem a ES_theta
    def __init__(self, ES_theta, ES_l):
        super(dExpSquared_dtheta,self).__init__(ES_theta, ES_l)
        
        self.ES_theta = ES_theta
        self.ES_l = ES_l

    def __call__(self,x1,x2,i,j):
        f1=self.ES_theta *2            
        f2=(x1-x2)**2
        f3=self.ES_l**2
        return  f1*np.exp(-0.5*f2/f3)        
    
class dExpSquared_dl(Kernel): #derivada em ordem a ES_l
    def __init__(self, ES_theta, ES_l):
        super(dExpSquared_dtheta,self).__init__(ES_theta, ES_l)
        
        self.ES_theta = ES_theta
        self.ES_l = ES_l       

    def __call__(self,x1,x2,i,j):
        f1=self.ES_theta**2
        f2=(x1-x2)**2
        f3=self.ES_l**3
        f4=self.ES_l**2
        return f1* (f2/f3)*  np.exp(-0.5 * f2/f4) 

class dExpSineSquared_dtheta(Kernel): # derivada em ordem a ESS_theta
    def __init__(self, ESS_theta, ESS_l, ESS_P):
        super(dExpSineSquared_dtheta, self).__init__(ESS_theta, ESS_l, ESS_P)

        self.ESS_theta = ESS_theta
        self.ESS_l = ESS_l
        self.ESS_P = ESS_P
        
    def __call__(self,x1,x2,i,j):
        f1 = self.ESS_theta *2
        f2 = self.ESS_l**2
        f3 = np.pi/self.ESS_P
        f4 = x1-x2
        return f1*np.exp(-(2/f2)*np.sin(f3*f4))

class dExpSineSquared_dl(Kernel): # derivada em ordem a ESS_l
    def __init__(self, ESS_theta, ESS_l, ESS_P):
        super(dExpSineSquared_dl, self).__init__(ESS_theta, ESS_l, ESS_P)

        self.ESS_theta = ESS_theta
        self.ESS_l = ESS_l
        self.ESS_P = ESS_P
        
    def __call__(self,x1,x2,i,j):
        f1=4* self.ESS_theta**2
        f2=self.ESS_l**3
        f3=np.pi/self.ESS_P
        f4=x1-x2
        f5=self.ESS_l**2
        return (f1*np.sin(f3*f4)/f2) * np.exp(-2*np.sin(f3*f4)/f5)
        
class dExpSineSquared_dP(Kernel): #derivada em ordem a ESS_P
    def __init__(self, ESS_theta, ESS_l, ESS_P):
        super(dExpSineSquared_dP, self).__init__(ESS_theta, ESS_l, ESS_P)

        self.ESS_theta = ESS_theta
        self.ESS_l = ESS_l
        self.ESS_P = ESS_P
        
    def __call__(self,x1,x2,i,j):
        f1=2*np.pi*self.ESS_theta**2
        f2=self.ESS_l**2
        f3=np.pi/self.ESS_P
        f4=self.ESS_P**2
        f5=x1-x2
        return (f1/(f2*f4)) * f5*np.cos(f3*f5) * np.exp(-2*np.sin(f3*f5)/f2) 
            
class dRatQuadratic_dtheta(Kernel): #derivada em ordem a RQ_theta
    def __init__(self,RQ_theta,RQ_l,RQ_alpha):
        super(dRatQuadratic_dtheta, self).__init__(RQ_theta, RQ_l, RQ_alpha)
        self.RQ_theta = RQ_theta
        self.RQ_l = RQ_l
        self.RQ_alpha = RQ_alpha           
            
    def __call__(self,x1,x2,i,j):
        f1=self.RQ_theta*2
        f2=(x1-x2)**2
        f3=self.RQ_alpha
        f4=self.RQ_l**2
        return f1*(1 + 0.5*f2/(f3*f4))**(-f3)

class dRatQuadratic_dl(Kernel): #derivada em ordem a RQ_l
    def __init__(self,RQ_theta,RQ_l,RQ_alpha):
        super(dRatQuadratic_dl, self).__init__(RQ_theta, RQ_l, RQ_alpha)
        self.RQ_theta = RQ_theta
        self.RQ_l = RQ_l
        self.RQ_alpha = RQ_alpha      
            
    def __call__(self,x1,x2,i,j):
        f1=self.RQ_theta**2
        f2=(x1-x2)**2
        f3=self.RQ_alpha
        f4=self.RQ_l**2
        f5=self.RQ_l**3
        return (f1*f2/f5)*(1 + 0.5*f2/(f3*f4))**(-1-f3)
 
class dRatQuadratic_dalpha(Kernel): #derivada em ordem a RQ_alpha
    def __init__(self,RQ_theta,RQ_l,RQ_alpha):
        super(dRatQuadratic_dalpha, self).__init__(RQ_theta, RQ_l, RQ_alpha)
        self.RQ_theta = RQ_theta
        self.RQ_l = RQ_l
        self.RQ_alpha = RQ_alpha           
            
    def __call__(self,x1,x2,i,j):
        f1=self.RQ_theta*2
        f2=(x1-x2)**2
        f3=self.RQ_alpha
        f4=self.RQ_l**2
        func1=0.5*f2/(f3*f4*(1 + 0.5*f2/(f3*f4)))
        func2=1+ 0.5*f2/(f3*f4)
        return f1*(func1 - np.log(func2)) * func2**(-f3)        
