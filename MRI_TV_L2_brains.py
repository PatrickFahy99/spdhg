import numpy as np
import sigpy as sp
import sigpy.mri as mri
import sigpy.plot as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
import odl
#import odl.contrib.solvers.spdhg as spdhg
import os.path
import operators as ops
import functionals as fctls
from functools import partial


'''####  solving                                                      ######## 
#######                                                               ########
#######        x = argmin Sum_{i=1}^n  1/2*||Ai(x)-bi||^2 + g(x)      ########
#######                                                               ########
#######   using SPDHG with real MRI data, with TV-L2 regularizer      ########
#######                                                               ########
#######           g =  l/2 ||x||^2_2  +  l1 ||grad(x)||_1             ########
#######                                                               ########
#######   with serial sampling, optimal serial sampling (L2)          ########
#######        b-nice sampling, and optimal b-nice sampling (L2)      #####'''


# Data
filename = ['cartesian_ksp','mri_data3','mri_data4','mri_data5','mri_data6','mri_data7',
            'fastmri_1','fastmri_2','fastmri_3','fastmri_4']
filename = filename[2]

# Regularization parameters
l = 10**-2                                      # L2                                      
l1= 10**2                                       # TV

# Instructions
redo_sigpy = False
redo_spdhg = False
overwrite_sigpy_recon = False
overwrite_figures = False

# Sampling
samplings = ['bserial','bnice','bfair']
sampling = samplings[0]

# Epochs
nepoch = 8                                    # Epochs
nruns = 1                                       # Repeats
rho = 0.98                                          
prox_niter = 1

#b_list = [1,2,3,4,6,12]                         # Values of b
b_list=[1,12]
lb = len(b_list)
I = range(lb)                                   
#I = [0,1,3,4,5]                                # Select values to plot

dpi = 780
ylog_on = False
titles_on = True
contrast = 1.2*10**7
#contrast = None

# Prox_Niter scheme
#for Prox_Niter in ['fixed','balanced']:
for Prox_Niter in ['balanced']:
#for Prox_Niter in ['fixed']:
    
    # Partitions
    #for choice in ['best','worst']:  
    for choice in ['best']:
        
        
        '''#########################  Get Data  #########################'''
        
        ##### Read Groundtruth
        ksp = np.load('../data/'+filename+'.npy')
        #pl.ImagePlot(sp.ifft(ksp, axes=(-1,-2)),mode='l',z=0,title='Inverse Fourier of data')
        
        ##### Coil sensitivities estimated with E-Spirit
        if os.path.isfile('../data/'+filename+'_coilsens.npy'):
            mps = np.load('../data/'+filename+'_coilsens.npy')   
        else:
            mps = mri.app.EspiritCalib(ksp).run()           # ESpirit Calibration
            np.save('../data/'+filename+'_coilsens.npy', mps) 
        # pl.ImagePlot(mps, z=0, title='Sensitivity Maps by ESPIRiT')
        
        ##### String Parameters
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
        
        ##### Load Solution
        if l == 0 and os.path.isfile('../9_plots_IOP_paper/new_results_TV/'+filename+'_truesol.npy'):
            sol = np.load('../9_plots_IOP_paper/new_results_TV/'+filename+'_truesol.npy')
            print('TV true solution')
            
        elif l1 == 0 and os.path.isfile('true_solutions/'+filename+'_truesol.npy') and not redo_sigpy:
            sol = np.load('true_solutions/'+filename+'_truesol.npy')
            print('L2 true solution')

        elif os.path.isfile('true_solutions/'+filename+'_'+strL2+'_'+strTV+'.npy'):
            sol = np.load('true_solutions/'+filename+'_'+strL2+'_'+strTV+'.npy')
            #sol = np.load('true_solutions/'+filename+'_'+strL2+'_'+strTV+'_10.npy')
            print('TVL2 true solution')
        
        else: raise SystemExit("No true solution found")
        
        
        '''#########################  Define Model  #########################'''
        
        ##### Discrete Space
        n,d1,d2 = ksp.shape                             # n signals of size d1*d2
        
        C = odl.uniform_discr([-1, -1], [1, 1], [d1,d2], dtype='complex') # Complex space
        X = C.real_space**2                                               # Real space
        M = ops.Magnitude(X)     
        R = ops.Complex2Real(C)                                                                             
        
        
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
        
        g = fctls.resetTV(X, reg_term=l1)
        g = odl.solvers.FunctionalQuadraticPerturb(g,l/2)
        
        # if l1==0:
        #     g = l/2*odl.solvers.L2NormSquared(X)        #L2 
        # elif l==0:
        #     g = fctls.resetTV(X, reg_term=l1)           #TV 
        # else:
        #     g = fctls.resetTV(X, reg_term=l1) 
        #     g = odl.solvers.FunctionalQuadraticPerturb(g,l/2)
        
        
        '''##########################  Define SPDHG  ###########################'''
        
        x_0 = X.zero()                                  
        #Phi_0 = f(A(x_0)) + g(x_0)                      # Initial value
        sol = X.element(sol)
        #sol = R(C.element(sol)) 
        
        # plt.imshow(M(sol),cmap='gray', vmax=contrast), plt.axis('off')
        # if titles_on: plt.title('PDHG Reconstruction with $\lambda_1=${}, $\lambda_2=${}'.format(l1,l))
        # plt.show()   
        #Phi_s = f(A(sol)) + g(sol)                      # Target value
        
        
        ##### Callback 
        def callback(step,n,f,g,A,sol,b):
            cb = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=step, end=', ') &
               odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s', cumulative=True, step=step)
               & ops.CallbackStore2brains(n,f,g,A,sol,b,contrast,overwrite_figures,filename,Prox_Niter,prox_niter))       # output every n iterations
            return cb
        
        
        ##### Samplings
        def serial_batch(x,b,n,nbatches,prob=None):
            if prob is None:
                prob = [1/nbatches]*nbatches
            i = np.random.choice(nbatches,p=prob)            
            return list(range( b*i, min(b*(i+1),n) ))
        
        def arb_batch(x,b,n,prob=None):
            if prob is None:
                prob = [1/n]*n
            return np.random.choice(n,b,replace=False,p=prob) 
        
        
        ##### SPDHG
        def spdhg_batch0(sampling, b, gamma, model, l,  n, f, g , A, nepoch, nruns, prox_niter):
            
            nbatches = int(np.ceil(n/b))
            niter = nepoch*nbatches
            
            step = max(niter//5,1)
            
            ##### GET STEP SIZE CONDITION
            #b-nice sampling
            if sampling == 'bnice':
                prob = [b/n]*n   
                fun_select = partial(arb_batch, b=b, n=n, prob=prob)
                
                AA = A*A.adjoint
                D = [Ai*Ai.adjoint for Ai in A]
                D = odl.DiagonalOperator(*D)
                PAA = b*(b-1)/(n-1)/n * AA + b*(n-b)/(n-1)/n * D
                norm_PAA = PAA.norm(True)
                
                #Strongly convex g,f*
                if l>0:
                    stki = np.sqrt(1+norm_PAA/rho**2/l/np.array(prob))
                    theta = 1 - 2*b/(n+sum(stki))
                    print('theta =',theta**(n//b))
                    tau = b/l/(n-2*b+sum(stki))
                    sigma = 1/(stki-1)
        
                #convex
                else:
                    theta = 1
                    tau = rho/gamma
                    sigma = rho*gamma*np.array(prob)**2/np.sqrt(norm_PAA)**2
                
            #b-serial sampling
            else:
                norm_AI = []                        
                for i in range(nbatches):
                    AI = odl.BroadcastOperator(*A[i*b:(i+1)*b])
                    norm_AI.append(AI.norm(True))
                
                #Strongly convex g,f*
                if l>0:
                    
                    #uniform
                    if sampling == 'bfair':
                        prob = np.array([1/nbatches]*nbatches)
                        stki = np.sqrt(1+np.array(norm_AI)**2/l/rho**2)  
                        theta = 1-2/(nbatches+nbatches*max(stki))        
                        tau = l**-1/(nbatches-2+nbatches*max(stki))
                        sigma = 1/max(stki-1)
                        sigma = np.array([sigma]*len(stki))
        
                    #optimal
                    else:
                        stki = np.sqrt(1+np.array(norm_AI)**2/l/rho**2)  
                        theta = 1-2/(nbatches+sum(stki))          
                        tau = l**-1/(nbatches-2+sum(stki))
                        sigma = 1/(stki-1)
                        prob = (1+stki)/(nbatches+sum(stki))
                        
                    print('theta =',theta**(n//b))
                    print('prob =',prob)
        
                #Convex g,f*
                else:
                    theta = 1
                    prob = np.array([1/nbatches]*nbatches)
                    tau = rho / gamma
                    sigma = rho*prob*gamma/np.array(norm_AI)**2   
                
                fun_select = partial(serial_batch, b=b, n=n, nbatches=nbatches, prob=prob)
                
                if len(sigma) < n:
                    sigma = [sigma[int(i/b)] for i in range(n)]
                    prob = [prob[int(i/b)] for i in range(n)]
            
            print('tau = ',tau,' sigma= ',min(sigma))
        
            ##### Run algorithm
            if b==n: nruns=1
            temp = np.zeros([4,nruns,nepoch+1])
            for i in range(nruns):
                x = x_0.copy()
                if 1==1:
                    g = fctls.resetTV(X, reg_term=l1, prox_niter=prox_niter)                # TV Regularizer
                    g = odl.solvers.FunctionalQuadraticPerturb(g,l/2)   
                cb = callback(step,nbatches,f,g,A,sol,b)
                ops.spdhg2(x, f, g, A, tau, sigma, niter, theta=theta, 
                            prob=prob, fun_select=fun_select, callback=cb)
                temp[0,i] = cb.callbacks[1].out              # Ob.Fun. f(Ax) +g(x)
                temp[1,i] = cb.callbacks[1].err              # error x^k - x* 
                temp[2,i] = cb.callbacks[1].dis              # distance x^k - x^k-1
                temp[3,i] = cb.callbacks[1].time             # computational time
        
            # Plot image
            # im = M(x)
            # plt.imshow(im,cmap='gray', vmin=0, vmax=contrast), plt.axis('off')
            # if titles_on: plt.title(sampling+' sampling, $b={}$, $\lambda_1=${}, $\lambda_2=${}'.format(b,l1,l))
            # plt.show()
            
            # Mean and Variance
            s = np.sum(temp[0],0)/nruns
            e = np.sum(temp[1],0)/nruns
            d = np.sum(temp[2],0)/nruns
            time = np.sum(temp[3],0)/nruns
            s_var = np.sqrt(np.sum((temp[0]-s)**2,0)/max(nruns-1,1))
            e_var = np.sqrt(np.sum((temp[1]-e)**2,0)/max(nruns-1,1))
            d_var = np.sqrt(np.sum((temp[2]-d)**2,0)/max(nruns-1,1))
            
            return s,e,d,s_var,e_var,d_var,time,theta
        
        
        '''#########################  Run SPDHG  #########################'''
        
        if l1==0: model='L2'
        elif l==0: model='TV'
        else: model='TVL2'
        
        lb = len(b_list)
        if os.path.isfile('brains/'+filename+'_'+Prox_Niter+str(prox_niter)+'_b'+str(1)+'_'+str(1)+'.png') and not redo_spdhg:

            #### Show Brains at every epoch             
            
            nepoch_list = [0,1,3,7]
            fontsize = 12
            
            plt.rcParams['figure.dpi'] = dpi
            plt.rcParams['savefig.dpi'] = dpi
            
            # Brains 
            Im = []
            for index,b in enumerate(b_list):
                Im.append([])
                for e in nepoch_list:
                    im = plt.imread('brains/'+filename+'_'+Prox_Niter+str(prox_niter)+'_b'+str(b)+'_'+str(e+1)+'.png')
                    Im[index].append(im)
                    ##
                #     plt.imshow(im,cmap='gray',vmax=contrast), plt.axis('off')
                #     plt.title('$b={}$, epoch {}'.format(b,e))
                #     plt.show()
                # plt.figure()
                # plt.imshow(M(sol),cmap='gray', vmax=contrast), plt.axis('off')
                # plt.title('PDHG Reconstruction')
                # plt.show()
            
            f = plt.figure(layout='constrained', figsize=(8,3))
            subf = f.subfigures(1,2, wspace=0.0, width_ratios=[4,1])
            ax = subf[0].subplots(2,4)
            for index,b in enumerate(b_list):
                for e,e2 in enumerate(nepoch_list):
                    ax[index,e].imshow(Im[index][e])
                    #ax[index,e].get_xaxis().set_visible(False)
                    ax[index,e].set_xticks([])
                    ax[index,e].set_yticks([])
                    if index==1 and e==0: ax[index,e].set_xlabel(str(e2+1)+' epoch', fontsize=fontsize)
                    if index==1 and e!=0: ax[index,e].set_xlabel(str(e2+1)+' epochs', fontsize=fontsize)
                    if e==0 and index==0: ax[index,e].set_ylabel('SPDHG', fontsize=fontsize)
                    if e==0 and index==1: ax[index,e].set_ylabel('PDHG', fontsize=fontsize)
            ax1 = subf[1].subplots(1,1)
            ax1.imshow(M(sol),cmap='gray', vmax=contrast)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_xlabel('Target', fontsize=fontsize)
            f.supylabel(' Primal\nvariable  ',rotation=0)
            plt.show()
            
            # Error with colorbar
            Im2 = []
            for index,b in enumerate(b_list):
                Im2.append([])
                for e in nepoch_list:
                    im = np.load('brains/'+filename+'_'+Prox_Niter+str(prox_niter)+'_b'+str(b)+'_'+str(e+1)+'.npy')
                    Im2[index].append(im-sol)
                    
            vmax = max([ np.amax(im[0]) for im in Im2[-1] ])
            # = min([np.amin(np.array(xr2)) for xr2 in x_recon2])
            vmin = -vmax
            cmap = 'seismic'

            f = plt.figure(layout='constrained', figsize=(8,3))
            subf = f.subfigures(1,2, wspace=0.0, width_ratios=[4,1])
            ax = subf[0].subplots(2,4)
            for index,b in enumerate(b_list):
                for e,e2 in enumerate(nepoch_list):
                    ax[index,e].imshow(Im2[index][e][0], cmap=cmap, vmin=vmin, vmax=vmax)
                    ax[index,e].set_xticks([])
                    ax[index,e].set_yticks([])
                    if index==1 and e==0: ax[index,e].set_xlabel(str(e2+1)+' epoch', fontsize=fontsize)
                    if index==1 and e!=0: ax[index,e].set_xlabel(str(e2+1)+' epochs', fontsize=fontsize)
                    if e==0 and index==0: ax[index,e].set_ylabel('SPDHG', fontsize=fontsize)
                    if e==0 and index==1: ax[index,e].set_ylabel('PDHG', fontsize=fontsize)
            ax1 = subf[1].subplots(1,1)
            #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            norm = mpl.colors.Normalize(vmin=-1, vmax=1)
            fraction=1
            cbar = ax1.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax1, pad=.05, extend='both', fraction=fraction)
            ax1.axis('off')
            f.supylabel('Primal     \n error  ',rotation=0)
            plt.show()

            
        else:
            s = np.zeros([lb,nepoch+1])
            e = np.zeros([lb,nepoch+1])
            d = np.zeros([lb,nepoch+1])
            s_var = np.zeros([lb,nepoch+1])
            e_var = np.zeros([lb,nepoch+1])
            d_var = np.zeros([lb,nepoch+1])
            time  = np.zeros([lb,nepoch+1])
            theta = np.zeros(lb)
            
            #gamma = [-0.3,-0.5,-0.7,-0.8,-0.8,-1.1]     # for bserial-TV only
            #gamma = [-0.6,-0.9,-1.1,-1.2,-1.3,-1.6]
            gamma = [-0.7,-0.9,-1.1,-1.2,-1.3,-1.5]
        
            '''#### Get the partitions and their convegence rates #####''' 
            for j,b in enumerate(b_list):
                
                print('b=',b)
                nbatches = n//b
                
                if Prox_Niter == 'fixed':
                    prox_niter_b = prox_niter
                else:
                    prox_niter_b = prox_niter*b  #### Cuestionable parameters
                print('prox_niter =',prox_niter_b)
                
                if j!=0 and j!=lb-1:
                    results = np.load('results_theta/'+filename+'_b{}.npy'.format(b))
                    
                    amin = int(results[nbatches+1,0])
                    if len(results[0])>1: amax = int(results[nbatches+1,1])
                    amin_u = int(results[nbatches+2,0])
                    if len(results[0])>1: amax_u = int(results[nbatches+2,1])
                    B = (results[nbatches+3:].T).astype(int)
                    
                    #### Get Best & Worst Partition
                    if sampling=='bserial':
                        order_min = B[amin]
                        order_max = B[amax]
                    if sampling=='bfair':
                        order_min = B[amin_u]
                        order_max = B[amax_u]
                        
                    if choice == 'best':
                        order = order_min
                    else: 
                        order = order_max
                    
                    #### Order Ai and fi
                    Bi = [Ai[i] for i in order]
                    Fi = [fi[i] for i in order]
                    A = odl.BroadcastOperator(*Bi)
                    f = odl.solvers.SeparableSum(*Fi)
                    
                    print(order)
                
                spdhg_batch = partial(spdhg_batch0, model=model, l=l, n=n, 
                                      f=f, g=g, A=A, nepoch=nepoch, nruns=nruns, prox_niter=prox_niter_b)
                
                i = j
                s[i],e[i],d[i],s_var[i],e_var[i],d_var[i],time[i],theta[i] = spdhg_batch(sampling,b,10**gamma[i])
                print('---------------')
                
                #if titles_on: plt.title('PDHG Reconstruction with $\lambda_1=${}, $\lambda_2=${}'.format(l1,l))
                
                plt.figure()
                plt.imshow(M(sol),cmap='gray', vmax=contrast), plt.axis('off')
                plt.title('PDHG Reconstruction')
                plt.show()
            
            
        

        # '''#########################  Plot Results  #############################'''
        
        # # Style choices
        # plt.style.use('default')
        # prop_cycle = plt.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        # plt.style.use('seaborn')
        # lw = 2.5
        # al = 0.5
        
        # # Legend
        # lgnd = []  #Legend for each b
        # for i,b in enumerate(b_list):
        #     lgnd.append(r'$b={}$'.format(b))
        
        
        # #I = [0,1,2,3,4,5] 
        # ##### Plot computational time
        # for j,i in enumerate(I):
        #     plt.plot( time[i], color=colors[j], linewidth=lw, label=lgnd[i] )
        # plt.legend(fontsize=18, frameon=True, facecolor='white')
        # if titles_on: plt.title('$time$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
        # plt.xlabel('epochs',fontsize=14)
        # #plt.xlim(0,nepoch+1)
        # #plt.ylim(7*10**-5,2)
        # #plt.yscale('log')
        # #plt.xscale('log')
        # plt.show()

   
        # ##### Plot Relative Distance to Solution ||x^k - x*||
        # #J=[0,2,5]
        # for j,i in enumerate(I):
        #     plt.plot( e[i], color=colors[j], linewidth=lw, label=lgnd[i] )
        # plt.legend(fontsize=18, frameon=True, facecolor='white')
        # if titles_on: plt.title('$||x^k-x^*||/||x^*||$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
        # plt.xlabel('epochs',fontsize=14)
        # #plt.ylim(0.2*10**-6,1.2)
        # #plt.xlim(0,50)
        # plt.yscale('log')
        # #plt.xscale('log')
        # plt.show()
        
        
        # ##### Plot Relative Distance to Solution ||x^k - x*|| over time
        # for j,i in enumerate(I):
        #     plt.plot( time[i], e[i], color=colors[j], linewidth=lw, label=lgnd[i] )
        # plt.legend(fontsize=18, frameon=True, facecolor='white')
        # if titles_on: plt.title('Error $||x^k-x^*||/||x^*||$ over time \n for MRI with $n=${}, $\lambda=${}'.format(n,l))
        # plt.xlabel('seconds',fontsize=14)
        # #plt.xlim(0,20)
        # #plt.ylim(0.2*10**-6,1.2)
        # plt.yscale('log')
        # #plt.xscale('log')
        # plt.show()
        
        
        # if nruns>1:
        #     ##### Plot Distance to Solution ||x^k - x*|| (with variance)
        #     #J = [0,3,5]
        #     J = I
        #     for j,i in enumerate(J):
        #         plt.plot( e[i], color=colors[j], linewidth=lw+2, label=lgnd[i] )
        #         std = np.sqrt(e_var[i])
        #         aux = np.linspace(0,nepoch,nepoch+1)
        #         plt.fill_between( aux, e[i]-std, e[i]+std, color=colors[j], alpha=al )
        #     plt.legend(fontsize=18, frameon=True, facecolor='white')
        #     if titles_on: plt.title('$||x^k-x^*||/||x^*||$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
        #     plt.xlabel('epochs',fontsize=14)
        #     if ylog_on: plt.yscale('log')
        #     #plt.ylim(0,1.05)
        #     #plt.xlim(0,50)
        #     #plt.yscale('log')
        #     plt.show()
        
        
        # if l>0:
        # ##### Plot Distance to Solution ||x^k - x*|| vs theorical rate theta^n
        #     #J = [0,2,5]
        #     J=I
        #     for j,i in enumerate(J):
        #         plt.plot( e[i]**2, color=colors[j], linewidth=lw, label=lgnd[i] )
        #         b = b_list[i]
        #         aux = np.linspace(0,nepoch,nepoch+1)
        #         rate = 10**-2*theta[i]**(aux*n//b)
        #         plt.plot( rate, color=colors[j], linewidth=lw, linestyle='dashed')
        #     #plt.ylim(10**-8,0.05)
        #     plt.legend(fontsize=18, frameon=True, facecolor='white')
        #     plt.title('$||x^k-x^*||^2/||x^*||^2$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
        #     plt.xlabel('epochs',fontsize=14)
        #     plt.yscale('log')
        #     #plt.xscale('log')
        #     #plt.xlim(0,nepoch)
        #     plt.show()
        
        # #%%
        
        # ##### Plot Distance between iterates ||x^k - x^{k-1}||
        # #epochs = np.linspace(2,nepoch,nepoch-2)
        # for j,i in enumerate(I):
        #     plt.plot( d[i], color=colors[j], linewidth=lw, label=lgnd[i] )
        # plt.legend(fontsize=18, frameon=True, facecolor='white')
        # #if titles_on: plt.title(r'$||x^{k}-x^{k-1}||\frac{1}{||x^{k-1}||}$'+' for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
        # if titles_on: plt.title(r'$||x^{k}-x^{k-1}||$'+' for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
        # plt.xlabel('epochs',fontsize=14)
        # plt.yscale('log')
        # #plt.xscale('log')
        # #plt.ylim(10**-3,10**-1)
        # #plt.xlim(0,nepoch)
        # plt.show()
       
        
        # # ##### Plot Distance between iterates ||x^k - x^{k-1}|| (with variance)
        # # aux = np.linspace(2,nepoch,nepoch-2)
        # # for j,i in enumerate(I):
        # #     plt.plot( aux, d[i,2:], color=colors[j], linewidth=lw, label=lgnd[i] )
        # #     std = np.sqrt(d_var[i,2:])
        # #     plt.fill_between( aux, d[i,2:]-std, d[i,2:]+std, color=colors[j] )
        # # plt.legend(fontsize=18, frameon=True, facecolor='white')
        # # if titles_on: plt.title(r'$||x^{k}-x^{k-1}||\frac{1}{||x^{k-1}||}$'+' for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
        # # plt.xlabel('epochs',fontsize=14)
        # # if ylog_on: plt.yscale('log')
        # # #plt.xscale('log')
        # # plt.xlim(0,nepoch)
        # # plt.show()
        
        
        # # ##### Plot Relative Objective Functional Phi(x^k)
        # # #epochs = np.linspace(1, nepoch, nepoch)
        # # for j,i in enumerate(I):
        # #     plt.plot( s[i], color=colors[j], linewidth=lw, label=lgnd[i] )
        # # plt.legend(fontsize=18, frameon=True, facecolor='white')
        # # if titles_on: plt.title('$\Phi_r(x^k)$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
        # # plt.xlabel('epochs',fontsize=14)
        # # plt.yscale('log')
        # # #plt.xscale('log')
        # # #plt.xlim(0,nepoch)
        # # plt.show()
        
        
        # # ##### Plot Objective Functional Phi(x^k) (with variance)
        # # epochs = np.linspace(1, nepoch, nepoch)
        # # for j,i in enumerate(I):
        # #     plt.plot( epochs, s[i], color=colors[j], linewidth=lw, label=lgnd[i] )
        # #     std = np.sqrt(s_var[i])
        # #     plt.fill_between( epochs, s[i]-std, s[i]+std, color=colors[j] )
        # # plt.legend(fontsize=18, frameon=True, facecolor='white')
        # # if titles_on: plt.title('$\Phi(x^k)$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
        # # plt.xlabel('epochs',fontsize=14)
        # # if ylog_on: plt.yscale('log')
        # # plt.xscale('log')
        # # plt.xlim(0,nepoch)
        # # plt.show()
        
        
        # # ##### Plot Relative Objective Functional Phi_r(x^k)
        # # Phi_s = f(A(sol)) + g(sol)
        # # for j,i in enumerate(I):
        # #     sr = abs((s[i] - Phi_s)/(Phi_0 - Phi_s))
        # #     plt.plot( sr, color=colors[j], linewidth=lw, label=lgnd[i] )
        # # plt.legend(fontsize=18, frameon=True, facecolor='white')
        # # if titles_on: plt.title('$\Phi_r(x^k)$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
        # # plt.xlabel('epochs',fontsize=14)
        # # plt.yscale('log')
        # # #plt.xscale('log')
        # # plt.xlim(0,nepoch)
        # # plt.show()
        
        
        # # ##### Plot Relative Objective Functional Phi_r(x^k) (with variance)
        # # aux = np.linspace(0,nepoch-1,nepoch)
        # # for j,i in enumerate(I):
        # #     sr = abs((s[i] - Phi_s)/(Phi_0 - Phi_s))
        # #     plt.plot( sr, color=colors[j], linewidth=lw, label=lgnd[i] )
        # #     std = np.sqrt(s_var[i])/(Phi_0 - Phi_s)
        # #     plt.fill_between( aux, sr-std, sr+std, color=colors[j] )
        # # plt.legend(fontsize=18, frameon=True, facecolor='white')
        # # if titles_on: plt.title('$\Phi_r(x^k)$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
        # # plt.xlabel('epochs',fontsize=14)
        # # if ylog_on: plt.yscale('log')
        # # #plt.xscale('log')
        # # plt.xlim(0,nepoch)
        # # plt.show()
