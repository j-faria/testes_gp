# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:30:55 2016

@author: camacho
"""

###### A PARTIR DAQUI ACHO QUE NÃO É NECESSARIO MAS DEIXO FICAR NA MESMA ######

#    # Rever - tirado do george e ver como adaptar
#    # self.gradient_symmetric e self.gradient_general  
#    def gradient(self, x1, x2=None):
#        x1 = np.ascontiguousarray(x1, dtype=np.float64)
#        if x2 is None:
#            g = self.gradient_symmetric(x1)
#        else:
#            x2 = np.ascontiguousarray(x2, dtype=np.float64)
#            g = self.gradient_general(x1, x2)
#        return g * self.vector_gradient[None, None, :]
#
#    @property
#    def vector_gradient(self):
#        return self.pars
#    #####

#class Local_ExpSineSquared(Kernel): #equal to ExpSquared*ExpSineSquared
#    def __init__(self, LESS_theta, LESS_l, LESS_P):
#        super(Local_ExpSineSquared, self).__init__(LESS_theta, LESS_l, LESS_P)
#        self.LESS_theta = LESS_theta
#        self.LESS_l = LESS_l
#        self.LESS_P = LESS_P
#        
#    def __call__(self, x1 ,x2 , i,  j):
#        f1 = self.LESS_theta**2                               
#        f2 = self.LESS_l**2
#        f3 = (x1-x2)
#        f4 = (x1-x2)**2
#        f5 = self.LESS_P
#        return f1*np.exp(-2*(np.sin(np.pi*f3/f5))**2/f2)*np.exp(-0.5*f4/f2)   

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


