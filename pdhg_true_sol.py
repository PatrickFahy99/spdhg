import numpy as np
import os.path
import odl
import odl.contrib.solvers.spdhg as spdhg
import operators as ops
import matplotlib.pyplot as plt
import sigpy as sp
import sigpy.mri as mri
import functionals as fctls


'''####  solving                                                      ######## 
#######                                                               ########
#######        x = argmin Sum_{i=1}^n  1/2*||Ai(x)-bi||^2 + g(x)      ########
#######                                                               ########
#######   using PDHG with real MRI data and TV-L2 reg, i.e.           ########
#######                                                               ########
#######               g = l1||grad(x)||_1 + l2/2||x||^2_2             #####'''


redo = True
save_png = False
overwrite = True

niter = 10**4
prox_niter = 10
rho = 0.98

vmin=0
vmax=1.2*10**7
#vmax=None
title_on = True
show_plot= False

# Choose Data
filenames = ['cartesian_ksp','mri_data3','mri_data4','mri_data5','mri_data6','mri_data7',
            'fastmri_1','fastmri_2','fastmri_3','fastmri_4']
filename = filenames[2]

# Regularization
# l1= 10**2
# l = 10**-2

# for l in [10**-3,10**-2,10**-1]:
#     for l1 in [10**1,10**2,10**3]:
for l in [10**-2]:
    for l1 in [10**2]:
    
        ##### Read Groundtruth
        ksp = np.load('data/'+filename+'.npy')
        
        ##### Coil sensitivities estimated with E-Spirit
        if os.path.isfile('data/'+filename+'_coilsens.npy'):
            mps = np.load('data/'+filename+'_coilsens.npy')   
        else:
            mps = mri.app.EspiritCalib(ksp).run()           # ESpirit Calibration
            np.save('data/'+filename+'_coilsens.npy', mps) 
        
        ##### Discrete Space
        n,d1,d2 = ksp.shape                             # n signals of size d1*d2
        
        C = odl.uniform_discr([-1, -1], [1, 1], [d1,d2], dtype='complex') # Complex space
        X = C.real_space**2                                               # Real space
        M = ops.Magnitude(X)     
        R = ops.Complex2Real(C)           
        
        # Save Parameters
        def string_parameters(l1,l):
            paramTV, paramL2 = int(np.log10(l1)), int(np.log10(l))
            strTV = str(abs(paramTV))
            strL2 = str(abs(paramL2))
            if paramTV < 0:
                strTV = 'minus'+strTV
            if paramL2 < 0:
                strL2 = 'minus'+strL2
            strTV = 'TV'+strTV
            strL2 = 'L2'+strL2
            return strTV, strL2
        strTV, strL2 = string_parameters(l1,l)
        aux_redo = False
        
        
        #### Load Image
        if os.path.isfile('true_solutions/'+filename+'_'+strL2+'_'+strTV+'.png') and not redo:
            im = plt.imread('true_solutions/'+filename+'_'+strL2+'_'+strTV+'.png')
            
        #### Load solution
        elif os.path.isfile('true_solutions/'+filename+'_'+strL2+'_'+strTV+'_'+str(prox_niter)+'.npy') and not redo:
            x = np.load('true_solutions/'+filename+'_'+strL2+'_'+strTV+'_'+str(prox_niter)+'.npy')
            im = M(x)
    
        #### PDHG
        else:
            aux_redo = True
            
            ##### Forward Operator
            F = ops.RealFourierTransform(X)      
            weights = (sp.rss(ksp,axes=(0,)) > 0).astype(ksp.dtype)
            S = ops.RealMultiplyOperator(X,C.element(weights**0.5))
                       
            Ai = []                                        
            for i in range(n):                              
                ci = C.element(mps[i])                    
                Ci = ops.RealMultiplyOperator(X,ci)         # Ci(x) = x.*ci (element-wise)
                Ai.append(S*F*Ci)                           # Ai = S*F*Ci
            A = odl.BroadcastOperator(*Ai)             
            
            ##### Convex Functionals
            ksp = ksp * weights**0.5                        # Apply weighting to data y 
            fi = []                                         
            for i in range(n):
                bi = R(C.element(ksp[i]))                   # Data points
                fi.append(0.5*odl.solvers.functional.L2NormSquared(A[i].range).translated(bi))
            
            f = odl.solvers.SeparableSum(*fi)               # Data fidelity 
            
            g = fctls.resetTV(X, reg_term=l1, prox_niter=prox_niter)
            g = odl.solvers.FunctionalQuadraticPerturb(g,l/2)   # TV + str_conv
            
            ##### Callback 
            def callback(step,n,f,g,A):
                cb = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=step, end=', ') &
                   odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s', cumulative=True, step=step)
                   & ops.CallbackStore(n,f,g,A))       # output every n iterations
                return cb
            
            # print('lambda = ',l)
            # norm_A = A.norm(True)
            # tau = rho / gamma 
            # sigma = rho * gamma / norm_A**2  
            # print('tau = ',tau,' sigma= ',sigma)
            
            norm_A = A.norm(True)
            stki = np.sqrt(1+norm_A**2/l/rho**2)  
            theta = 1-2/(1+stki)          
            tau = l**-1/(-1+stki)
            sigma = 1/(stki-1)
            prob = (1+stki)/(1+stki)
            
            x = X.zero()
            cb = callback(niter//5, 1, f, g, A)
            spdhg.pdhg(x, f, g, A, tau, sigma, niter, callback=cb)
            s = np.array(cb.callbacks[1].out)
            im = M(x)
            
            
        plt.imshow(im,cmap='gray',vmin=vmin,vmax=vmax), plt.axis('off')
        if aux_redo and title_on: plt.title('MRI-TV with $\lambda_1=${}, $\lambda_2=${}'.format(l1,l))
        if aux_redo and save_png: plt.savefig('true_solutions/'+filename+'_'+strL2+'_'+strTV+'.png',bbox_inches='tight', pad_inches=0)
        plt.show()
        
        if aux_redo and overwrite:
            # Save Solution
            np.save('true_solutions/'+filename+'_'+strL2+'_'+strTV+'_'+str(prox_niter)+'.npy',np.array(x))
        
        if aux_redo and show_plot:
            # Plot Objective Functional Phi(x^k)
            plt.plot(s)
            plt.title('$\Phi(x^k)$ for MRI Reconstruction $\lambda1=${}, $\lambda2=${}'.format(l1,l))
            plt.xlabel('epochs')
            plt.yscale('log')
            plt.xscale('log')
            plt.show()

        
