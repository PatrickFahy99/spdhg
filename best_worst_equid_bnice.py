import numpy as np
import sigpy as sp
import sigpy.mri as mri
import sigpy.plot as pl
import matplotlib.pyplot as plt
import odl
import odl.contrib.solvers.spdhg as spdhg
import os.path
import operators as ops
import functionals as fctls
from functools import partial


'''####  solving                                                      ######## 
#######                                                               ########
#######        x = argmin Sum_{i=1}^n  1/2*||Ai(x)-bi||^2 + g(x)      ########
#######                                                               ########
#######   using SPDHG with real MRI data, with TV or L2 reg, i.e.     ########
#######                                                               ########
#######         g = l||grad(x)||_1    or    g = l/2*||x||^2           ########
#######                                                               ########
#######   with serial sampling, optimal serial sampling (L2)          ########
#######        b-nice sampling, and optimal b-nice sampling (L2)      #####'''


# Choose Data
filename = ['cartesian_ksp','mri_data3','mri_data4','mri_data5','mri_data6','mri_data7',
            'fastmri_1','fastmri_2','fastmri_3','fastmri_4']
filename = filename[2]

# Choose Regularizer
model = ['L2','TV','TVL2']
model = model[2] 

if model=='TV': l = 10**2                       # Regularization parameter
else: l = 10**-2

'''Sampling ['bserial','bfair','bnice']'''
'''    choice ['best','worst',None]    '''

nepoch = 40                                    # Epochs
                                        
Prox_Niter = 'balanced'
prox_niter = 10

contrast = None

#b_list = [1,2,3,4,6,12]
b_list = [4]
lb = len(b_list)
I = range(lb)                                   # Plot results for all b
#I = [0,1,2,3,5]                                 # Values to plot (ignore b=12)

ylog_on = False
titles_on = False

###### 5 curves to compare
sampling = 'bserial'
#Choices = ['best','equid','bnice','naive','worst']
Choices = ['best','equid','naive','worst']
ncurves = len(Choices)

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
#pl.ImagePlot(mps, z=0, title='Sensitivity Maps by ESPIRiT')

##### PDHG solution
if model == 'TV' and os.path.isfile('new_results_TV/'+filename+'_truesol.npy'):
    sol = np.load('new_results_TV/'+filename+'_truesol.npy')
    
##### Sigpy solution
elif model == 'L2' and os.path.isfile('../data/'+filename+'_sigpy.npy'):
    sol = np.load('../data/'+filename+'_sigpy.npy')


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

if model=='L2':
    g = l/2*odl.solvers.L2NormSquared(X)        # L2 Regularizer
else:
    g = fctls.resetTV(X, reg_term=l)            # TV Regularizer


'''##########################  Load results  ###########################'''

# x_0 = X.zero()                                  

# if model == 'L2':
#     sol = R(C.element(sol))               # SIGPY solution
#     # plt.imshow(M(sol),cmap='gray', vmax=contrast), plt.axis('off')
#     # if titles_on: plt.title('SIGPY Reconstruction, $\lambda=${}'.format(l))
#     # plt.show()
# else:
#     sol = X.element(sol)
#     # plt.imshow(M(sol),cmap='gray', vmax=contrast), plt.axis('off')
#     # if titles_on: plt.title('PDHG Reconstruction, $\lambda=${}'.format(l))
#     # plt.show()   
    

### Recover data
s = []
e = []
d = []
s_var=[]
e_var=[]
d_var=[]
time=[]
theta = []

for choice in Choices:
    if choice == 'bnice':
        results = np.load('results_'+model+'_best_worst/'+filename+'_bnice_'+Prox_Niter+str(prox_niter)+'_bnice.npz')
    else:
        results = np.load('results_'+model+'_best_worst/'+filename+'_'+sampling+'_'+Prox_Niter+str(prox_niter)+'_'+choice+'.npz')
    Results = results['r']
    s.append(Results[0*lb:1*lb])
    e.append(Results[1*lb:2*lb])
    d = (Results[2*lb:3*lb])
    s_var.append(Results[3*lb:4*lb])
    e_var.append(Results[4*lb:5*lb])
    d_var.append(Results[5*lb:6*lb])
    time.append(Results[6*lb:7*lb])
    theta.append(results['t'])


'''#########################  Plot Results  #############################'''


# Style choices
plt.style.use('default')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.style.use('seaborn')
lw = 3
al = 0.5
ylog_on = False

# Batch size
b = 4

# Legend

Choices = ['best   ','equid ','cons  ','worst ']

lgnd = []
for j,choice in enumerate(Choices):
    if choice == 'bnice':
        str_t = r'$b$-nice $\vartheta={}$'.format(np.format_float_positional(theta[j]**(n//b),4,unique=False))
    else:
        str_t =  choice + r' $\vartheta={}$'.format(np.format_float_positional(theta[j]**(n//b),4,unique=False))
    lgnd.append(str_t)

##### Plot Distance to Solution ||x^k - x*||
#plt.figure(figsize = [8,5] )
for j,choice in enumerate(Choices):
    plt.plot( e[j][0]**2, color=colors[j], linewidth=lw, label=lgnd[j] )
plt.legend(fontsize=17, frameon=True, facecolor='white', ncol=1, loc='upper right')
if titles_on: plt.title('$||x^k-x^*||/||x^*||$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
plt.xlabel('epochs',fontsize=18)
plt.ylabel(r'$e^2_b$'+'\t\t',rotation=0, fontsize=18)
plt.xlim(0,nepoch)
plt.hlines(10**-4, 0, nepoch, color = 'k', linestyle = ':', alpha =0.5, linewidth = 2)
plt.ylim(10**-6,10**0)
plt.yscale('log')
plt.show()



# ####### Write the values of theta  (best vs worst only)
# if model=='L2':
#     colors = prop_cycle.by_key()['color']
#     #colors = colors[0:1]+colors[2:4]
#     lgnd = []           #Legend for each b
#     for i,b in zip([0,4,5],[1,6,12]):
#         if b==1 or b==n:
#             lgnd.append(r'$b={},\;'.format(b) + r'\vartheta_{os}='+r'{}$'.format(np.format_float_positional(t[i]**(n//b),4)))
#         else:
#             lgnd.append(r'$b={},\;'.format(b) + r'\vartheta_{os}^{best}='+r'{}$'.format(np.format_float_positional(t[i]**(n//b),4)))
#             lgnd.append(r'$b={},\;'.format(b) + r'\vartheta_{os}^{worst}='+r'{}$'.format(np.format_float_positional(t2[i]**(n//b),4)))
    
#     ##### Plot Distance to Solution ||x^k - x*||
#     I=[0,4,5]
#     index=0
#     for j,i in enumerate(I):
#         plt.plot( e[i], color=colors[j], linewidth=lw, label=lgnd[index] )
#         index+=1
#         if i!=I[0] and i!=I[-1]:
#             plt.plot( e2[i], color=colors[j], linewidth=lw, linestyle='dashed', label=lgnd[index] )
#             index+=1
#     plt.legend(fontsize=18, frameon=True, facecolor='white')
#     if titles_on: plt.title('$||x^k-x^*||/||x^*||$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
#     plt.xlabel('epochs',fontsize=14)
#     plt.xlim(0,80)
#     plt.ylim(10**-4/1.5,2*10**-1)
#     plt.yscale('log')
#     #plt.xscale('log')
#     plt.show()




# ##### Plot Relative Objective Functional Phi_r(x^k)
# Phi_0 = f(A(x_0)) + g(x_0)                      # Initial value
# Phi_s = f(A(sol)) + g(sol)
# if case == 1:
#     for j,i in enumerate(I):
#         sr = abs((s[i] - Phi_s)/(Phi_0 - Phi_s))
#         if i==0:
#             plt.plot( epochs, sr, color=colors[j], linewidth=lw, label=r'$b=${}'.format(b_list[i]) )
#         elif i==5:
#             plt.plot( epochs, sr, color=colors[j], linewidth=lw, label=r'$b=${}'.format(b_list[i]) )
#         else:
#             sr2 = abs((s2[i] - Phi_s)/(Phi_0 - Phi_s))
#             plt.plot( epochs, sr, color=colors[j], linewidth=lw, label=r'$b=${}'.format(b_list[i])+r' best' )
#             plt.plot( epochs, sr2, color=colors[j], linewidth=lw, linestyle='dashed', label=r'$b=${}'.format(b_list[i])+r' worst' )
# #
# if case == 2:
#     for j,i in enumerate(I):
#         if i==0:
#             plt.plot( epochs, e[i,2:], color=colors[j], linewidth=lw, label=r'$b=${}'.format(b_list[i])+r' bserial' )
#             plt.plot( epochs, e2[i,2:], color=colors[j], linewidth=lw, linestyle='dashed', label=r'$b=${}'.format(b_list[i])+r' bnice' )
#         elif i==5:
#             plt.plot( epochs, e[i,2:], color=colors[j], linewidth=lw, label=r'$b=${}'.format(b_list[i]) )
#         else:
#             plt.plot( epochs, e[i,2:], color=colors[j], linewidth=lw, label=r'$b=${}'.format(b_list[i])+r' bserial' )
#             plt.plot( epochs, e2[i,2:], color=colors[j], linewidth=lw, linestyle='dashed', label=r'$b=${}'.format(b_list[i])+r' bnice' )
# # for j,i in enumerate(I):
# #     sr = abs((s[i] - Phi_s)/(Phi_0 - Phi_s))
# #     sr2 = abs((s2[i] - Phi_s)/(Phi_0 - Phi_s))
    
# #     plt.plot( sr, color=colors[j], linewidth=lw, label=lgnd[i] )
# #     plt.plot( sr2, color=colors[j], linewidth=lw, linestyle='dashed' )
# plt.legend(fontsize=18, frameon=True, facecolor='white')
# if titles_on: plt.title('$\Phi_r(x^k)$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
# plt.xlabel('epochs',fontsize=14)
# plt.yscale('log')
# #plt.xscale('log')
# plt.xlim(0,100)
# plt.show()






# ##### Plot Distance between iterates ||x^k - x^{k-1}||
# epochs = np.linspace(2,nepoch,nepoch-2)
# for j,i in enumerate(I):
#     plt.plot( epochs, d[i,2:], color=colors[j], linewidth=lw, label=lgnd[i] )
#     plt.plot( epochs, d2[i,2:], color=colors[j], linewidth=lw, linestyle='dashed' )
# plt.legend(fontsize=18, frameon=True, facecolor='white')
# if titles_on: plt.title(r'$||x^{k}-x^{k-1}||\frac{1}{||x^{k-1}||}$'+' for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
# plt.xlabel('epochs',fontsize=14)
# #plt.ylim(10**-4,10**-3)
# plt.yscale('log')
# #plt.xscale('log')
# plt.xlim(0,100)
# plt.show()


# #%%
# ##### Plot Distance between iterates ||x^k - x^{k-1}||
# epochs = np.linspace(2,nepoch,nepoch-2)
# for j,i in enumerate(I):
#     if i==0:
#         plt.plot( epochs, d2[i,2:], color=colors[j], linewidth=lw, label=r'$b=${}'.format(b_list[i])+r' bserial' )
#         plt.plot( epochs, d[i,2:], color=colors[j], linewidth=lw, linestyle='dashed', label=r'$b=${}'.format(b_list[i])+r' bnice' )
#     elif i==5:
#         plt.plot( epochs, d[i,2:], color=colors[j], linewidth=lw, label=r'$b=${}'.format(b_list[i]) )
#     else:
#         plt.plot( epochs, d[i,2:], color=colors[j], linewidth=lw, label=r'$b=${}'.format(b_list[i])+r' bserial' )
#         plt.plot( epochs, d2[i,2:], color=colors[j], linewidth=lw, linestyle='dashed', label=r'$b=${}'.format(b_list[i])+r' bnice' )
# #plt.legend(fontsize=19, frameon=True, facecolor='white', ncol=1)
# if titles_on: plt.title(r'$||x^{k}-x^{k-1}||\frac{1}{||x^{k-1}||}$'+' for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
# plt.xlabel('epochs',fontsize=14)
# plt.yscale('log')
# #plt.ylim(10**-4,0.2)
# #plt.xlim(0,100)
# plt.ylim(10**-4,10**-3)
# plt.xlim(60,100)
# plt.show()
# #%%


# ##### Plot Objective Functional Phi(x^k)
# epochs = np.linspace(1, nepoch, nepoch)
# for j,i in enumerate(I):
#     plt.plot( epochs, s[i], color=colors[j], linewidth=lw, label=lgnd[i] )
#     plt.plot( epochs, s2[i], color=colors[j], linewidth=lw, linestyle='dashed' )
# plt.legend(fontsize=18, frameon=True, facecolor='white')
# if titles_on: plt.title('$\Phi(x^k)$ for MRI Reconstruction \n $n=${}, $\lambda=${}'.format(n,l))
# plt.xlabel('epochs',fontsize=14)
# plt.yscale('log')
# plt.xscale('log')
# plt.xlim(0,100)
# plt.show()




