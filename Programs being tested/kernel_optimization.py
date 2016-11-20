# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:33:29 2016

@author: camacho
"""
import Kernel;reload(Kernel);kl = Kernel
import kernel_likelihood;reload(kernel_likelihood); lk= kernel_likelihood

import numpy as np
import inspect
from time import time   


##### OPTIMIZATION ############################################################
def optimization(kernel,x,xcalc,y,yerr,method='CGA'):
    if method=='CGA':
        CGM(kernel,x,xcalc,y,yerr)
    if method=='SDA':
        SDA(kernel,x,xcalc,y,yerr)
    else:
        print 'vai dormir'        
       
#Auxiliary calculations       
def opt_likelihood(kernel, x, xcalc, y, yerr): #covariance matrix calculations   
    K = np.zeros((len(x),len(x))) #covariance matrix K
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            K[i,j] = kernel(x1, x2)
    K=K+yerr**2*np.identity(len(x))      
    log_p_correct = lk.lnlike(K, y)
    from scipy.linalg import cho_factor, cho_solve
    L1 = cho_factor(K) # tuple (L, lower)
    sol = cho_solve(L1, y) # this is K^-1*(r)
    n = y.size
    logLike = -0.5*np.dot(y, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)        
    return logLike

def opt_gradlike(kernel, x,xcalc,y,yerr):
    grd= lk.gradient_likelihood(kernel, x,xcalc,y,yerr) #gradient likelihood
    grd= [-grd for grd in grd] #isto só para inverter os si
    return grd    

def new_kernel(kernelFIRST,b):
    if isinstance(kernelFIRST,kl.ExpSquared):
        return kl.ExpSquared(b[0],b[1])
    elif isinstance(kernelFIRST,kl.ExpSineSquared):
        return kl.ExpSineSquared(b[0],b[1],b[2])
    elif  isinstance(kernelFIRST,kl.RatQuadratic):
        return kl.RatQuadratic(b[0],b[1],b[2])
    elif isinstance(kernelFIRST,kl.Exponential):
        return kl.Exponential(b[0],b[1])
    elif isinstance(kernelFIRST,kl.Matern_32):
        return kl.Matern_32(b[0],b[1])
    elif isinstance(kernelFIRST,kl.Matern_52):
        return kl.Matern_52(b[0],b[1])
    elif isinstance(kernelFIRST,kl.ExpSineGeorge):
        return kl.ExpSineGeorge(b[0],b[1])
    else:
        print 'Falta o white noise!'    



        
# Conjugate gradient (Fletcher-Reeves) Algorithm     
def CGA(kernel,x,xcalc,y,yerr,step=0.005,precision = 1e-5,iterations=5):
    kernelFIRST=kernel #just not to loose the original one
    xFIRST=x;xcalcFIRST=xcalc #just not to loose the original data
    yFIRST=y;yerrFIRST=yerr #just not to loose the original data
    
    it=0        

# Steepest descent Algorithm
def SDA(kernel,x,xcalc,y,yerr,step=0.005,precision = 1e-5,iterations=10):
    kernelFIRST=kernel #just not to loose the original one
    xFIRST=x;xcalcFIRST=xcalc #just not to loose the original data
    yFIRST=y;yerrFIRST=yerr #just not to loose the original data
    #If I don't do this "...FIRST" to save the data it would raise as error
    
    it=0
    while it<iterations:
#        print 'iteration number:',it+1
        hyperparms=[] #initial values of the hyperparameters 
        for k in range(len(kernel.__dict__['pars'])):
            hyperparms.append(kernel.__dict__['pars'][k])
#        print 'hyperparameters ->', hyperparms
#        hyperparms=[np.log(x) for x in hyperparms]
        hyperparms = hyperparms
#        if it==0:
#            hyperparms = [np.log(x) for x in hyperparms]
#            hyperparms = hyperparms
#        else:
#            hyperparms = [np.log(x) for x in hyperparms]
#            hyperparms = hyperparms
#        print 'hyperparameters ->', hyperparms       
#        print 'kernel ->', kernel; print ''
        
#        first_calc= opt_likelihood(kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #likelihood
        second_calc= opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #gradient likelihood
        #print 'opt_likelihood ->', first_calc
        #print '-gradient ->', second_calc; print ''
    
        #print 'antes', hyperparms #X_i
        new_hyperparms = [x*step for x in second_calc]
        #print '-LAMBDAxGRAD', new_hyperparms # - Lambda * gradFunction
        new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
        #print 'final', new_hyperparms #X_i+1
        kernel.__dict__['pars'][:]=new_hyperparms 
        a = kernel.__dict__['pars']
        
        b=[]    
        for ij in range(len(a)):
            b.append(a[ij])         
        kernel=new_kernel(kernelFIRST,b) #new kernel with hyperparams updated
#        print 'nova kernel ->',kernel; print''
        
#        first_calc= opt_likelihood(kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #likelihood
        #print 'opt_likelihood ->', first_calc
#        second_calc= opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #gradient likelihood        
#        print '-gradient ->', second_calc; print ''        
        
        it+=1 #should go back to the start and do the while, but error is raised
        first_calc= opt_likelihood(kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #likelihood    
    print 'iterations ->', it
    print 'new kernel ->', kernel
    print 'log likelihood ->',  first_calc
        
        
        
        
import tests
##### trash - do not delete ###################################################
        
#def optimization1(kernel,x,xcalc,y,yerr,step=0.01,precision = 1e-5,iterations=5):
#    kernelFIRST=kernel #just not to loose the original one
#    hyperparms=[] #initial values of the hyperparameters 
#    for i in range(len(kernel.__dict__['pars'])):
#        hyperparms.append(kernel.__dict__['pars'][i])
#    print hyperparms
#    i=0 
#    while i<iterations: #to limit the number of iterations
#        first_calc= opt_likelihood(kernel,x,xcalc,y,yerr ) #likelihood
#        second_calc= gradient_likelihood(kernel, x,xcalc,y,yerr) #gradient likelihood
#        grd= [-second_calc for second_calc in second_calc] #just to invert the grad
#
#        a=kernel.__dict__['pars']
#        print a
#        x1=kernel.__dict__['pars'][0]
#        x2=kernel.__dict__['pars'][1]
#        print 'x1=',x1, 'x2=',x2
#        
#        
#        results = op.minimize(first,a,jac=second)    
#      
#        print 'opt_likelihood ->', first_calc
#        print '-gradient ->', grd
#        print 'iteracoes ->', i
#
#        i+=1
#        print  kernel
#        
#    a=kernel.__dict__
#    p0=[]
#    for i in range(len(a['pars'])):
#        p0.append(a['pars'][i])
#    p0=np.array(p0)
#    print type(p0), p0
#    
#    for i in arange(1,len(kernel.__dict__)+1):
#        var = "k%i" %i
#        k_i = a[var]
#    print k_i
#    results = op.minimize(lik, p0, jac=grd)
#    #p0 é o valor dos hiperparametros
#    #jac  é igual a -gradient_likelihood
        
        
###### Steepest descent - com tempo rever que algo ta mal
##def optimization(kernel,x,xcalc,y,yerr,step=0.005,precision = 1e-5,iterations=5):
#def SDA(kernel,x,xcalc,y,yerr,step=0.005,precision = 1e-5,iterations=5):
#    kernelFIRST=kernel #just not to loose the original one
#    xFIRST=x;xcalcFIRST=xcalc #just not to loose the original data
#    yFIRST=y;yerrFIRST=yerr #just not to loose the original data
#    #If I don't do this "...FIRST" to save the data it would raise as error
#    
#    it=0
#    while it<iterations:
##        print 'iteration number:',it+1
#        hyperparms=[] #initial values of the hyperparameters 
#        for k in range(len(kernel.__dict__['pars'])):
#            hyperparms.append(kernel.__dict__['pars'][k])
##        print 'hyperparameters ->', hyperparms
##        hyperparms=[np.log(x) for x in hyperparms]
#        hyperparms = hyperparms
##        if it==0:
##            hyperparms = [np.log(x) for x in hyperparms]
##            hyperparms = hyperparms
##        else:
##            hyperparms = [np.log(x) for x in hyperparms]
##            hyperparms = hyperparms
##        print 'hyperparameters ->', hyperparms       
##        print 'kernel ->', kernel; print ''
#        
##        first_calc= opt_likelihood(kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #likelihood
#        second_calc= opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #gradient likelihood
#        #print 'opt_likelihood ->', first_calc
#        #print '-gradient ->', second_calc; print ''
#    
#        #print 'antes', hyperparms #X_i
#        new_hyperparms = [x*step for x in second_calc]
#        #print '-LAMBDAxGRAD', new_hyperparms # - Lambda * gradFunction
#        new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
#        #print 'final', new_hyperparms #X_i+1
#        kernel.__dict__['pars'][:]=new_hyperparms 
#        a = kernel.__dict__['pars']
#        
#        b=[]    
#        for ij in range(len(a)):
#            b.append(a[ij])         
#        kernel=new_kernel(kernelFIRST,b) #new kernel with hyperparams updated
##        print 'nova kernel ->',kernel; print''
#        
##        first_calc= opt_likelihood(kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #likelihood
#        #print 'opt_likelihood ->', first_calc
##        second_calc= opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #gradient likelihood        
##        print '-gradient ->', second_calc; print ''        
#        
#        it+=1 #should go back to the start and do the while, but error is raised
#        first_calc= opt_likelihood(kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #likelihood    
#    print 'iterations ->', it
#    print 'new kernel ->', kernel
#    print 'log likelihood ->',  first_calc