# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:28:12 2016

@author: camacho
"""

# Conjugate gradient (Fletcher-Reeves) Algorithm - DOESN'T WORK 
def CGA(kernel,x,xcalc,y,yerr,step=1e-10,precision = 1e-5,iterations=5):
    kernelFIRST=kernel #just not to loose the original one
    xFIRST=x;xcalcFIRST=xcalc #just not to loose the original data
    yFIRST=y;yerrFIRST=yerr #just not to loose the original data
    it=0

    hyperparms=[] #initial values of the hyperparameters 
    for k in range(len(kernel.__dict__['pars'])):
        hyperparms.append(kernel.__dict__['pars'][k])
    hyperparms = hyperparms

    first_calc= opt_gradlike(kernel, xFIRST,xcalcFIRST,yFIRST,yerrFIRST) #gradient likelihood
    new_hyperparms = [x*step for x in first_calc]
    new_hyperparms = [sum(x) for x in zip(hyperparms, new_hyperparms)]
    kernel.__dict__['pars'][:]=new_hyperparms 
    a = kernel.__dict__['pars']
    b=[]    
    for ij in range(len(a)):
        b.append(a[ij])         
    kernel=new_kernel(kernelFIRST,b) #new kernel with hyperparams updated    

    calc_aux1=[x**2 for x in first_calc] #|deltaF1|**2#
    calc_aux2=sum(calc_aux1)
    
    it+=1        
    while it<iterations:
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
    print 'iterations ->', it-1
    print 'final log likelihood ->',  final_log
    print 'final kernel ->', kernel               