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
        CGA(kernel,x,xcalc,y,yerr)
    if method=='SDA':
        SDA(kernel,x,xcalc,y,yerr)
#    else:
#        print 'vai dormir'        
       
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
    grd= [-grd for grd in grd] #isto só para inverter os Si's
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

# Conjugate gradient (Fletcher-Reeves) Algorithm - DOES IT WORK? 
def CGA(kernel,x,xcalc,y,yerr,step=0.005,precision = 1e-5,iterations=10):
    kernelFIRST=kernel #just not to loose the original one
    xFIRST=x;xcalcFIRST=xcalc #just not to loose the original data
    yFIRST=y;yerrFIRST=yerr #just not to loose the original data
    it=0

    #First step
    hyperparms=[] #initial values of the hyperparameters 
    for k in range(len(kernel.__dict__['pars'])):
        hyperparms.append(kernel.__dict__['pars'][k])
    #hyperparms=[np.log(x) for x in hyperparms]
    hyperparms = hyperparms
    
    #first calc gives S1
    first_calc= opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #gradient likelihood
    #print 'first_calc', first_calc 
    #new_hyperparms first gives lambdaXS1
    new_hyperparms = [x*step for x in first_calc] #gives lambda x S1
    #print '-LAMBDAxGRAD', new_hyperparms # - Lambda * gradFunction
    #new_hyperparms now gives X1 + lambda x S1
    new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
    kernel.__dict__['pars'][:]=new_hyperparms 
    a = kernel.__dict__['pars']
    
    b=[]    
    for ij in range(len(a)):
        b.append(a[ij])         
    kernel=new_kernel(kernelFIRST,b) #new kernel with hyperparams updated
    print 'iteration number:',it+1
    print 'nova kernel ->',kernel; print''
        
    #calc_aux will give |deltaF1|**2#
    calc_aux1=[x**2 for x in first_calc] 
    #print 'calc_1',calc_aux1
    calc_aux2=sum(calc_aux1)
    #print 'calc_2',calc_aux2
    it+=1 

    while it<iterations:
        if it%(len(hyperparms)+1)!=0: 
            print 'iteration number:',it+1
            hyperparms=[] #initial values of the hyperparameters 
            for k in range(len(kernel.__dict__['pars'])):
                hyperparms.append(kernel.__dict__['pars'][k])
            #hyperparms=[np.log(x) for x in hyperparms]
            hyperparms = hyperparms
            
            #this calc_aux will give |deltaF1|**2
            calc_aux1=[x**2 for x in first_calc] 
            #print 'calc_1',calc_aux1
            calc_aux2=sum(calc_aux1)
            #print 'calc_2',calc_aux2           
            
            #second_calc gives -deltaF2
            second_calc= opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #gradient likelihood
            
            #this calc_aux will give |deltaF2|**2
            calc_aux3=[x**2 for x in second_calc] #
            #print   calc_aux3
            calc_aux4=sum(calc_aux3)
            #print 'delta_F2 ->', calc_aux4;print''     
            
            #this will be deltaF1/deltaF2
            deltas=calc_aux4/calc_aux2 
            #print deltas        
            
            hyperparms=[] #initial values of the hyperparameters 
            for k in range(len(kernel.__dict__['pars'])):
                hyperparms.append(kernel.__dict__['pars'][k])
            #hyperparms=[np.log(x) for x in hyperparms]
            hyperparms = hyperparms
         
            #this will first be deltas*S1
            new_hyperparms = [x*deltas for x in first_calc]
            #then it will be the sum of -deltaF2 with deltas*S1      
            new_hyperparms = [sum(x) for x in zip(second_calc, new_hyperparms)]
            #new_hyperparms now gives lambda*S2
            new_hyperparms = [x*step for x in new_hyperparms]
            #new_hyperparms will finally give X2 + lambda*S2
            new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
            kernel.__dict__['pars'][:]=new_hyperparms 
            a = kernel.__dict__['pars']
                  
            b=[]    
            for ij in range(len(a)):
                b.append(a[ij])         
            kernel=new_kernel(kernelFIRST,b) #new kernel with hyperparams updated
            print 'kernel nova ->',kernel; print''
            it+=1
            
            #para o ciclo continuar
            first_calc=opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
            #first_calc=first_calc
            calc_aux1=[x**2 for x in first_calc] #|deltaF1|**2#
            #calc_aux1=calc_aux1
            calc_aux2=sum(calc_aux1)
            #calc_aux2=calc_aux2
        else:
            print 'iteration number:',it+1
            hyperparms=[] #initial values of the hyperparameters 
            for k in range(len(kernel.__dict__['pars'])):
                hyperparms.append(kernel.__dict__['pars'][k])
            #hyperparms=[np.log(x) for x in hyperparms]
            hyperparms = hyperparms

            second_calc= opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #gradient likelihood
            #print 'second -gradient ->', second_calc; print ''
        
            new_hyperparms = [x*step for x in second_calc]
            new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
            kernel.__dict__['pars'][:]=new_hyperparms 
            a = kernel.__dict__['pars']
            
            b=[]    
            for ij in range(len(a)):
                b.append(a[ij])         
            kernel=new_kernel(kernelFIRST,b) #new kernel with hyperparams updated
            print 'nova kernel ->',kernel; print''
            it+=1
            
            #para o ciclo continuar
            first_calc=opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
            calc_aux1=[x**2 for x in first_calc] #|deltaF1|**2#
            calc_aux2=sum(calc_aux1)
            #print 'delta_F1 ->', calc_aux2;print''

    final_log= opt_likelihood(kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #likelihood    
    print 'total iterations ->', it
    print 'final log likelihood ->',  final_log
    print 'final kernel ->', kernel  
                

# Steepest descent Algorithm - DOES IT WORK? 
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
    
    final_log= opt_likelihood(kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #likelihood    
    print 'total iterations ->', it
    print 'log likelihood ->', final_log
    print 'new kernel ->', kernel        
        
        
        
import tests
##### trash - do not delete ###################################################

# Conjugate gradient (Fletcher-Reeves) Algorithm 1st version - DOESN'T WORK 
def CGA2(kernel,x,xcalc,y,yerr,step=1e-10,precision = 1e-5,iterations=5):
    kernelFIRST=kernel #just not to loose the original one
    xFIRST=x;xcalcFIRST=xcalc #just not to loose the original data
    yFIRST=y;yerrFIRST=yerr #just not to loose the original data
    it=0

    #First step
    hyperparms=[] #initial values of the hyperparameters 
    for k in range(len(kernel.__dict__['pars'])):
        hyperparms.append(kernel.__dict__['pars'][k])
    #hyperparms=[np.log(x) for x in hyperparms]
    hyperparms = hyperparms
    first_calc= opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #gradient likelihood
    calc_aux1=[x*step for x in first_calc] #|deltaF1|**2#
    calc_aux2=sum(calc_aux1)
    
    it+=1        
    while it<=iterations:
        if it%(len(hyperparms)+1)!=0:        
            print 'iteration number:',it
            new_hyperparms = [x*step for x in first_calc]
            new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
            kernel.__dict__['pars'][:]=new_hyperparms 
            a = kernel.__dict__['pars']        
            b=[]    
            for ij in range(len(a)):
                b.append(a[ij])         
            kernel=new_kernel(kernelFIRST,b) #new kernel with hyperparams updated
            #print 'kernel ->',kernel;print''
    
            second_calc= opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #gradient likelihood
            #print 'second -gradient ->', second_calc; print ''
    
            calc_aux3=[x**2 for x in second_calc] #|deltaF2|**2#
            #print   calc_aux3
            calc_aux4=sum(calc_aux3)
            #print 'delta_F2 ->', calc_aux4;print''
        
            deltas=calc_aux4/calc_aux2 #deltaF1/deltaF2
            #print deltas
            
            hyperparms=[] #initial values of the hyperparameters 
            for k in range(len(kernel.__dict__['pars'])):
                hyperparms.append(kernel.__dict__['pars'][k])
            #hyperparms=[np.log(x) for x in hyperparms]
            hyperparms = hyperparms
         
            new_hyperparms = [x*deltas for x in first_calc]
            #print 'deltas*S1 ->',new_hyperparms        
            new_hyperparms = [sum(x) for x in zip(second_calc, new_hyperparms)]
            #print 'S2+deltas*S1 ->',new_hyperparms
            
            kernel.__dict__['pars'][:]=new_hyperparms 
            a = kernel.__dict__['pars']        
            b=[]    
            for ij in range(len(a)):
                b.append(a[ij])         
            kernel=new_kernel(kernelFIRST,b) #new kernel with hyperparams updated
            print 'kernel nova ->',kernel; print''
            it+=1
            
            #para o ciclo continuar
            first_calc=opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
            calc_aux1=[x**2 for x in first_calc] #|deltaF1|**2#
            calc_aux2=sum(calc_aux1)
            #print 'delta_F1 ->', calc_aux2; print''
        else:
            print 'iteration number:',it
            hyperparms=[] #initial values of the hyperparameters 
            for k in range(len(kernel.__dict__['pars'])):
                hyperparms.append(kernel.__dict__['pars'][k])
            #hyperparms=[np.log(x) for x in hyperparms]
            hyperparms = hyperparms

            second_calc= opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #gradient likelihood
            #print 'second -gradient ->', second_calc; print ''
        
            new_hyperparms = [x*step for x in second_calc]
            new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
            kernel.__dict__['pars'][:]=new_hyperparms 
            a = kernel.__dict__['pars']
            
            b=[]    
            for ij in range(len(a)):
                b.append(a[ij])         
            kernel=new_kernel(kernelFIRST,b) #new kernel with hyperparams updated
            print 'nova kernel ->',kernel; print''
            it+=1
            
            #para o ciclo continuar
            first_calc=opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST)
            calc_aux1=[x**2 for x in first_calc] #|deltaF1|**2#
            calc_aux2=sum(calc_aux1)
            #print 'delta_F1 ->', calc_aux2;print''

    final_log= opt_likelihood(kernel,xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #likelihood    
    print 'total iterations ->', it-1
    print 'final log likelihood ->',  final_log
    print 'final kernel ->', kernel  

       
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