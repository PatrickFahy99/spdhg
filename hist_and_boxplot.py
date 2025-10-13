import numpy as np
import os.path
import odl
import operators as ops
import matplotlib.pyplot as plt
import sigpy as sp
import sigpy.mri as mri
#import sigpy.plot as pl
import itertools as it
import random
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines


'''###                                                        #######
###### SPDHG for different grouping of A_i with real MRI data #######
######               for different values of lambda           ####'''


##### Choose Data
filename = ['cartesian_ksp','mri_data3','mri_data4','mri_data5','mri_data6','mri_data7',
            'fastmri_1','fastmri_2','fastmri_3','fastmri_4']
filename = filename[2]

rho = 0.98
l = 10**-2                              # Regularization 

limit = 20000

redo = False
overwrite = False

titles_on = 0

##### Read Groundtruth
ksp = np.load('../data/'+filename+'.npy')


##### Coil sensitivities estimated with E-Spirit
if os.path.isfile('../data/'+filename+'_coilsens.npy'):
    mps = np.load('../data/'+filename+'_coilsens.npy')   
else:
    mps = mri.app.EspiritCalib(ksp).run()           # ESpirit Calibration
    np.save('../data/'+filename+'_coilsens.npy', mps) 
#pl.ImagePlot(mps, z=0, title='Sensitivity Maps by ESPIRiT')


##### SENSE Reconstruction with E-Spirit
if os.path.isfile('results_orth/'+filename+'_sigpy.npy'):
    img_sigpy = np.load('results_orth/'+filename+'_sigpy.npy')   
    plt.imshow(abs(img_sigpy),cmap='gray'), plt.axis('off')
    plt.title('Solution')
    plt.show()

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


##### Get all divisors of n
def divisors(n):
    return set(it.chain.from_iterable((i,n//i) for i in range(1,int(np.sqrt(n))+1) if n%i == 0))

b_list = divisors(n)

##### Naive partition
def naive(n):
    if n==12:
        B = [0,2,11,3,1,6,10,7,8,4,9,5]    # Order of the coils by position, starting by 0 = 3o'clock
        return B
    else: 
        print('error: n!=12')

##### Equidistant partition
def equidistant(n,b):
    B0 = naive(n)          
    nbatches = n//b
    B = []
    for i in range(nbatches):
        for j in range(b):
            B.append(B0[i+j*nbatches])
    return B

##### List all possible partitions, assumes n|b 
def partitions(N,b,B=[],temp=[]):
    if b<= 1:
        B = [N]
    elif len(N)<= b:
        B.append(temp + N)
    else:
        for comb in it.combinations(N[1:],b-1):
            temp_new = temp + [N[0]] + list(comb)
            N_new = list(set(N)-set(temp_new))
            B = partitions(N_new,b,B,temp_new)
    return B


def show_coils(b,Bmin,Bmax,mps,normalize=False):
    cmap = 'viridis'
    fig = plt.figure(figsize=(7.3,5))
    nrows = 5
    ncolm = 7
    vmax = 0.4
    vmin = 0
    gs = GridSpec(nrows,ncolm, hspace=0, wspace=0,
                  width_ratios=(1,1,1,0.1,1,1,1),height_ratios=(1,1,0.1,1,1))
    for (j,i) in enumerate(Bmin[:b]):
        ax0 = fig.add_subplot(gs[int(j/int(ncolm/2)),j%int(ncolm/2)])
        ax0.imshow(abs(mps[i]), vmin=vmin, vmax=vmax, cmap=cmap), ax0.axis('off')
    
    for (j,i) in enumerate(Bmin[b:]):
        ax1 = fig.add_subplot(gs[int(j/int(ncolm/2))+3,j%int(ncolm/2)])
        ax1.imshow(abs(mps[i]), vmin=vmin,vmax=vmax, cmap=cmap), ax1.axis('off')
    
    for (j,i) in enumerate(Bmax[:b]):
        ax2 = fig.add_subplot(gs[int(j/int(ncolm/2)),j%int(ncolm/2)+4])
        ax2.imshow(abs(mps[i]), vmin=vmin, vmax=vmax, cmap=cmap), ax2.axis('off')
        
    for (j,i) in enumerate(Bmax[b:]):
        ax3 = fig.add_subplot(gs[int(j/int(ncolm/2))+3,j%int(ncolm/2)+4])
        ax3.imshow(abs(mps[i]), vmin=vmin, vmax=vmax, cmap=cmap), ax3.axis('off')
    plt.show()
    
    # add the magnitudes
    # find the local maximum
    # colorbar//side by side
    # norms of the Ai
    # 
    MPS = sum([mps[i]/np.amax(abs(mps[i])) for i in Bmin[:b]])
    if normalize: MPS=MPS/np.amax(abs(MPS))
    plt.imshow(abs(MPS), cmap=cmap), plt.axis('off')
    plt.show()
    MPS = sum([mps[i]/np.amax(abs(mps[i])) for i in Bmin[b:]])
    if normalize: MPS=MPS/np.amax(abs(MPS))
    plt.imshow(abs(MPS), cmap=cmap), plt.axis('off')
    plt.show()
    
    MPS = sum([mps[i] for i in Bmax[:b]])
    if normalize: MPS=MPS/np.amax(abs(MPS))
    plt.imshow(abs(MPS), cmap=cmap), plt.axis('off')
    plt.show()
    MPS = sum([mps[i] for i in Bmax[b:]])
    if normalize: MPS=MPS/np.amax(abs(MPS))
    plt.imshow(abs(MPS), cmap=cmap), plt.axis('off')
    plt.show()

def showcoils_b6():
    angle = np.linspace(0,2*np.pi,n+1)[:-1]
    radius = 0.6
    Pbest = [0,2,4,6,8,10,1,3,5,7,9,11]
    Pworst = [2,3,4,5,6,7,8,9,10,11,0,1]
    for partition in [Pbest,Pworst]:
        xx = radius*np.cos(angle[partition])
        yy = radius*np.sin(angle[partition])
        
        plt.figure(figsize=(5,5))
        plt.scatter(xx[:n//2], yy[:n//2], s=2500, color='blue' )
        plt.scatter(xx[n//2:], yy[n//2:], s=2500, color='orange' )
        plt.xlim( -0.75 , 0.75 ) 
        plt.ylim( -0.75 , 0.75 ) 
        plt.axis('off')
        plt.show()

def show_colors(b, Pbest, Pworst, sampling='', theta=['',''], A_i=['',''], names=['Best','Worst'], fontsize=12):
    if b==1 or b==n: return
    P = [0,4,1,3,9,11,5,7,8,10,6,2] # Rearrenge the coils by their actual location
    Pbest = [P[i] for i in Pbest]
    Pworst= [P[i] for i in Pworst]
    ##
    angle = np.linspace(0,2*np.pi,n+1)[:-1]
    radius = 0.6
    for j,partition in enumerate([Pbest,Pworst]):
        if len(theta[j])>1:
            rate = '\n' + r'$\vartheta =$' + theta[j]
        if len(A_i[j])>1:
            norm = '\n' + r'$||A_j||_j$ =' + A_i[j]
        xx = radius*np.cos(angle[partition])
        yy = radius*np.sin(angle[partition])
        ##
        plt.style.use('default')
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        ##
        plt.figure(figsize=(5,5))
        for i in range(n//b):
            plt.scatter(xx[i*b:(i+1)*b], yy[i*b:(i+1)*b], s=2500, color=colors[i] )    
        plt.xlim( -0.75 , 0.75 ) 
        plt.ylim( -0.75 , 0.75 ) 
        plt.axis('off')
        #plt.title(sampling +r' $b=$' +str(b) +' ' +names[j] +rate +norm , fontsize=fontsize)
        plt.title(names[j] +' partition' +rate, fontsize=fontsize )
        plt.show()


'''#########  Compute histogram for each b  #########'''
tt = []                                     # optimal serial sampling
tt_unif = []                                # uniform serial sampling
ttheta  = []                                # uniform b-nice sampling

naive_t = []                                # optimal serial sampling with naive partition
naive_t_unif = []                           # uniform serial sampling with naive partition
equid_t = []                                # optimal serial sampling with equidistant partition
equid_t_unif = []                           # uniform serial sampling with equidistant partition

K_naive = []
K_equid = []

for index,b in enumerate([4]):
#for index,b in enumerate([1,2,3,4,6,12]):
    nbatches = n//b
    N = list(range(n))
    
    if os.path.isfile('results_theta/'+filename+'_b{}.npy'.format(b)) and not redo:
        results = np.load('results_theta/'+filename+'_b{}.npy'.format(b))
        K = np.transpose(results[:nbatches])
        norm_PAA = results[nbatches,0]
        amin = results[nbatches+1,0]
        if len(K)>1: amax = results[nbatches+1,1]
        amin_u = results[nbatches+2,0]
        if len(K)>1: amax_u = results[nbatches+2,1]
        B = (results[nbatches+3:].T).astype(int)
    
    else:
        #### Compute ||Ai||^2 for all combinations
        B = partitions(N,b,[],[])
        npar = len(B)
        print('----')
        print('# of partitions of size {}='.format(b),npar)
    
        if npar >= limit:
            npar = limit
            B = B[:limit]
            for i in range(limit): 
                random.shuffle(N)
                B[i] = N[:]
        
        K = np.zeros([npar,nbatches])
        for j in range(npar):
            Bi = [Ai[i] for i in B[j]]
            norm_AI = []                        
            for i in range(nbatches):
                AI = odl.BroadcastOperator(*Bi[i*b:(i+1)*b])
                norm_AI.append(AI.norm(True))
            print(j,norm_AI)
            K[j] = norm_AI
        
        # Compute ||P.AA|| for b-nice
        AA = A*A.adjoint
        D = [Ai*Ai.adjoint for Ai in A]
        D = odl.DiagonalOperator(*D)
        PAA = b*(b-1)/(n-1)/n * AA + b*(n-b)/(n-1)/n * D
        norm_PAA = PAA.norm(True)
        
        # Save Results
        if overwrite: 
            aux = np.zeros([1,len(K)])
            aux[0,0] = norm_PAA
            results = np.concatenate((np.array(K).T, aux, np.array(B).T))
            np.save('results_theta/'+filename+'_b{}.npy'.format(b),results)


    ##### Compute theta for all partitions #####
    tt_unif.append(np.zeros(len(K)))
    tt.append(np.zeros(len(K)))
    
    for j in range(len(K)):
        stki = np.sqrt(1+np.array(K[j])**2/l/rho**2)  # sqrt(tilde_kappa_i)
        tt_unif[index][j] = ( 1-2/(nbatches+nbatches*max(stki)) )**(n//b)
        tt[index][j] = ( 1-2/(nbatches+sum(stki)) )**(n//b)
    
    ##### Compute theta for b-nice sampling
    prob = [b/n]*n
    stbi = np.sqrt(1+norm_PAA/rho**2/l/np.array(prob))
    ttheta.append( [( 1 - 2*b/(n+sum(stbi)) )**(n//b)] )    
    
    
    ##### Compute theta for naive partition
    norm_AI = []
    Bi = [Ai[i] for i in naive(n)]
    for i in range(nbatches):
        AI = odl.BroadcastOperator(*Bi[i*b:(i+1)*b])
        norm_AI.append(AI.norm(True))
    K_naive.append(norm_AI)
    stki = np.sqrt(1+np.array(norm_AI)**2/l/rho**2)
    naive_t_unif.append( ( 1-2/(nbatches+nbatches*max(stki)) )**(n//b) )
    naive_t.append( ( 1-2/(nbatches+sum(stki)) )**(n//b) )
    
    
    ##### Compute theta for equidistant partition
    norm_AI = []   
    Bi = [Ai[i] for i in equidistant(n,b)]                     
    for i in range(nbatches):
        AI = odl.BroadcastOperator(*Bi[i*b:(i+1)*b])
        norm_AI.append(AI.norm(True))
    K_equid.append(norm_AI)
    stki = np.sqrt(1+np.array(norm_AI)**2/l/rho**2)
    equid_t_unif.append( ( 1-2/(nbatches+nbatches*max(stki)) )**(n//b) )
    equid_t.append( ( 1-2/(nbatches+sum(stki)) )**(n//b) )
    
    
    '''#########  Show Results (serial)  #########'''
    print('--')
    print('b =',b)
    print('optimal')
    npar = len(B)
    print('# of partitions of size {} ='.format(b),npar)
    
    #best
    t = tt[index]
    #t = t**(n//b)
    amin = np.argmin(t)                                   
    Bmin = B[amin]
    
    print('theta min =',t[amin])
    print('||A_i||_i =',K[amin])
    print('order min =',Bmin)
    
    #worst
    amax = np.argmax(t)
    Bmax = B[amax]
    
    print('theta max =',t[amax])
    print('||A_i||_i =',K[amax])
    print('order max =',Bmax)
    
    # b-nice
    theta = ttheta[index][0]
    print('thetabnice=',theta)
    
    #naive
    print('naive_theta =',naive_t[index])
    print('||A_i||_i =',K_naive[index])
    print('order naive =',naive(n))
    
    #equidistant
    print('equid_theta =',equid_t[index])
    print('||A_i||_i =',K_equid[index])
    print('order equid =',equidistant(n,b))
    
    
    ##### Style choices
    plt.style.use('default')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.style.use('seaborn')
    lw = 2.5
    al = 0.5
    
    ##### Histogram (serial) #####
    plt.hist(t, color='red', alpha=0.4)                  
    plt.xticks(fontsize=14)
    if titles_on: plt.title(filename+', b={}'.format(b)) 
    #plt.axvline(theta,color='blue')          
    plt.axvline(np.median(t), color='r', linewidth= 3)    
    # handles = [mlines.Line2D([0], [0], label='average', color='gray', linestyle='--'),
    #             mlines.Line2D([0], [0], label='b-nice', color='b')]
    handles = [mlines.Line2D([0], [0], label='median', color='r', linewidth= 2)]
    plt.legend(handles=handles, fontsize=20, frameon=True, facecolor='white')
    #plt.xlabel(r'$\vartheta$', fontsize=20)
    plt.show()

    ##### Color coils (serial)
    fontsize = 20
    tminmax = [np.format_float_positional(t[amin],4),
                np.format_float_positional(t[amax],4)]
    Ajminmax = [ str([ np.format_float_positional(kk,4) for kk in K[amin] ]),
                  str([ np.format_float_positional(kk,4) for kk in K[amax] ]) ]
    show_colors(b, Bmin, Bmax, 'opt b-serial', tminmax, Ajminmax, fontsize=fontsize)

    ##### Color coils (adjacent & equidistant)
    tminmax = [np.format_float_positional(naive_t[index],4),
                np.format_float_positional(equid_t[index],4)]
    Ajminmax = [ str([ np.format_float_positional(kk,4) for kk in K_naive[index] ]),
                  str([ np.format_float_positional(kk,4) for kk in K_equid[index] ]) ]
    show_colors(b, naive(n), equidistant(n,b), 'opt b-serial', tminmax, Ajminmax, ['Consecutive','Equidistant'], fontsize=fontsize)


    '''#########  Show results (fair)  #########'''
    print('--')
    print('b =',b)
    print('uniform')
    npar = len(B)
    print('# of partitions of size {}='.format(b),npar)
    
    #best
    t_unif = tt_unif[index]
    #t_unif = t_unif**(n//b)
    amin_u = np.argmin(t_unif)                                   
    Bmin_u = B[amin_u] 
    
    print('theta min =',t_unif[amin_u])
    print('||A_i||_i =',K[amin_u])
    print('order min =',Bmin_u)
    
    #worst
    amax_u = np.argmax(t_unif)
    Bmax_u = B[amax_u]
    
    print('theta max =',t_unif[amax_u])
    print('||A_i||_i =',K[amax_u])
    print('order max =',Bmax_u)
    
    # b-nice
    print('thetabnice =',theta)
    
    #naive
    print('naive_theta =',naive_t_unif[index])
    print('order naive =',naive(n))
    
    #equidistant
    print('equid_theta =',equid_t_unif[index])
    print('order equid =',equidistant(n,b))
    
    #%%
    ##### Style choices
    plt.style.use('default')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.style.use('seaborn')
    lw = 2.5
    al = 0.5
    
    ##### Histogram (fair) #####
    plt.hist(t_unif, color='g', alpha=0.3)                  
    plt.xticks(fontsize=14)
    if titles_on: plt.title(filename+', b={}'.format(b)) 
    plt.axvline(theta,color='blue', linewidth=2)          
    plt.axvline(np.median(t_unif),label='median', color='green', linewidth=2)    
    handles = [mlines.Line2D([0], [0], label='median', color='green', linewidth=2),
               mlines.Line2D([0], [0], label='b-nice', color='blue', linewidth=2)]    
    plt.legend(handles=handles, fontsize=20, frameon=True, facecolor='white')
    #plt.xlabel(r'$\vartheta$', fontsize=20)
    plt.show()

#%%
    #### Color coils (fair)
    tminmax = [np.format_float_positional(t_unif[amin_u],4),
                np.format_float_positional(t_unif[amax_u],4)]
    Ajminmax = [ str([ np.format_float_positional(kk,4) for kk in K[amin_u] ]),
                  str([ np.format_float_positional(kk,4) for kk in K[amax_u] ]) ]
    show_colors(b, Bmin_u, Bmax_u, 'unif b-serial', tminmax, Ajminmax)


    ##### Color coils (adjacent & equidistant)
    tminmax = [np.format_float_positional(naive_t_unif[index],4),
                np.format_float_positional(equid_t_unif[index],4)]
    Ajminmax = [ str([ np.format_float_positional(kk,4) for kk in K_naive[index] ]),
                  str([ np.format_float_positional(kk,4) for kk in K_equid[index] ]) ]
    show_colors(b, naive(n), equidistant(n,b), 'unif b-serial', tminmax, Ajminmax, ['adjacent','equidistant'] )    


    ##### Coils (only for b=6)
    if b==6:
        show_coils(b,Bmin,Bmax,mps,False)

    ##### Save min and max #####
    if overwrite: 
        aux = np.zeros([3,len(K)])
        aux[0,0] = norm_PAA
        aux[1,0] = amin
        if len(K)>1: aux[1,1] = amax
        aux[2,0] = amin_u
        if len(K)>1: aux[2,1] = amax_u
        results = np.concatenate((np.array(K).T,aux,np.array(B).T))
        np.save('results_theta/'+filename+'_b{}.npy'.format(b),results)
        
#%%
##### Boxplot #2 

fig1, ax1 = plt.subplots(figsize=(8,6))
ax1.grid(axis='x')
if titles_on: ax1.set_title(r'Distribution of $\vartheta$ for each $b$-sampling')
#labels = [ r'$~$', 1 , r'$~$', r'$~$', 2, r'$~$', r'$~$', 3, r'$~$',r'$~$', 4, r'$~$',r'$~$', 6, r'$~$',r'$~$', 12]

alpha = 0.9

tt_copy = []
tt_unif_copy = []
tdud = [np.NaN]
for i,b in enumerate(b_list):
    tt_copy.append(tt[i])
    tt_copy.append(tdud)
    tt_unif_copy.append(tdud)
    tt_unif_copy.append(tt_unif[i])
    if i!= len(b_list)-1:
        tt_copy.append(tdud)
        tt_unif_copy.append(tdud)

outlier_sz = 1.5

# c = 'red'
# bplot1 = ax1.boxplot(tt_copy, patch_artist=True, showcaps=True, #widths=0.5,
#             boxprops=dict(facecolor=c, color=c,alpha=0.65),
#             capprops=dict(color=c),
#             whiskerprops=dict(color=c),
#             flierprops=dict(color=c, marker='o', markeredgecolor=c, alpha=0.5, markersize=outlier_sz),
#             medianprops=dict(color=c, linewidth=2) )
c = 'g'
bplot2 = ax1.boxplot(tt_unif, patch_artist=True, showcaps=True, widths=0.4,
            boxprops=dict(facecolor='lightgreen', color=c, alpha = alpha),
            capprops=dict(color=c,linewidth=2,alpha=0.75),
            whiskerprops=dict(color=c,linewidth=2,alpha=0.75),
            flierprops=dict(color=c, markeredgecolor=c, alpha=1, markersize=outlier_sz),
            medianprops=dict(color=c,alpha=1,linewidth=2) )

# c = 'blue'
# bplot3 = ax1.boxplot(ttheta, patch_artist=True, 
#             boxprops=dict(facecolor=c, color=c),
#             medianprops=dict(color=c, alpha=0.9, linewidth=3), labels=labels )

dx=0.14
# caps = bplot1['caps']
caps2= bplot2['caps']
# for cap in caps:
#     cap.set(xdata=cap.get_xdata() + (-dx,+dx))
for cap in caps2:
    cap.set(xdata=cap.get_xdata() + (-dx,+dx))

dx = 0.24
sz = 20
c2 = 'darkviolet'
c3 = 'dodgerblue'
sz2= 16
mrk= '$-$'
shft =-0.0007
shft2= 0 
for i,b in enumerate(b_list):
    if i!=0 and i!=len(b_list)-1:
        #plt.plot([3*i+1],[min(tt[i])], marker=".", markersize=sz ,markeredgecolor="r", markerfacecolor="r", zorder=5)
        #plt.plot([3*i+1],[max(tt[i])], marker=".", markersize=sz ,markeredgecolor="r", markerfacecolor="r", zorder=5)
        plt.plot([i+1-dx,i+1+dx],[max(ttheta[i]),max(ttheta[i])], linewidth=2, color='blue', alpha=0.9)
        #plt.plot([i+1-dx,i+1+dx],[naive_t[i],naive_t[i]], linewidth=2, color='r', alpha=0.9)
        plt.plot([i+1],[min(tt_unif[i])], marker=".", markersize=sz ,markeredgecolor="g", markerfacecolor="g", zorder=5)
        plt.plot([i+1],[max(tt_unif[i])], marker=".", markersize=sz ,markeredgecolor="g", markerfacecolor="g", zorder=5)
        #naive
        #plt.plot([3*i+1],[naive_t[i]+shft], marker=mrk, markersize=sz2 ,markeredgecolor=c2, markerfacecolor=c2, zorder=4)
        #plt.plot([3*i+1],[equid_t[i]+shft], marker=mrk, markersize=sz2 ,markeredgecolor=c3, markerfacecolor=c3, zorder=4)
        #plt.plot([3*i+2],[naive_t_unif[i]+shft], marker=mrk, markersize=sz2 ,markeredgecolor=c2, markerfacecolor=c2, zorder=4)
        #plt.plot([3*i+2],[equid_t_unif[i]+shft], marker=mrk, markersize=sz2 ,markeredgecolor=c3, markerfacecolor=c3, zorder=4)
    # else:
        # plt.plot([i+1],[max(tt_unif[i])+shft2], marker='$-$', markersize=25 ,markeredgecolor="b", markerfacecolor="b", zorder=4)
        

# plt.plot([1], [ttheta[0]], marker="D", markersize=8, markeredgecolor="g", markerfacecolor="green")
# plt.plot([6], [ttheta[-1]], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="r")
# plt.plot([6], [ttheta[-1]], marker="D", markersize=7, markeredgecolor="g", markerfacecolor="green")

# eight = mlines.Line2D([], [], color='red', marker='$-$', markersize=25, ls='', label=r'$b$-serial (optimized)')
# nine = mlines.Line2D([], [], color='green', marker='$-$', markersize=25, ls='', label=r'$b$-serial')
# ten = mlines.Line2D([], [], color='blue', marker='$-$', markersize=25, ls='', label=r'$b$-nice')

# eight = mlines.Line2D([], [], color='red', marker='s', markersize=12, ls='', label=r'$\vartheta_{os}$')

nine = mlines.Line2D([], [], color='green', marker='s', markersize=12, ls='', label=r'$\vartheta_{us}$')
ten = mlines.Line2D([], [], color='blue', marker='$-$', markersize=25, ls='', label=r'$\vartheta_{un}$')
eleven = mlines.Line2D([], [], color='black', marker='.', markersize=sz, ls='', label='best/worst')
#twelve = mlines.Line2D([], [], color='black', marker='v', markersize=13, ls='', label='worst')
#naive = mlines.Line2D([], [], color=c2, marker='$-$', markersize=sz2, ls='', label='ordered')
#equid = mlines.Line2D([], [], color=c3, marker='$-$', markersize=sz2, ls='', label='equidistant')

ax1.legend([#bplot1["boxes"][0], 
            bplot2["boxes"][0], ten, eleven], [#r'$\vartheta_{os}$ (optimal $b$-serial)',
                                               r'$\vartheta_{us}$'+'   '+'$b$-serial',r'$\vartheta_{un}$'+'  '+' $b$-nice'
                                               #,'best/worst partition'
                                               ],fontsize=15, frameon=True,
            facecolor='white', ncol=1)
# ax1.legend(['b-serial','b-fair','b-nice','best','worst','~','~'],fontsize=18, frameon=True,
#             facecolor='white',handles=[eight, nine, ten, eleven, equid], ncol=2)
plt.ylim(0.81,0.83)
yticks = [round(r,3) for r in np.linspace(0.81,0.83,6)]
plt.yticks(yticks, fontsize=12)
plt.xticks([1,2,3,4,5,6],
           ['1 \n (1 partition)','2 \n (10395 p)','3 \n (15400 p)','4 \n (5775 p)','6 \n (462 p)','12 \n (1 partition)'],
           #[1,2,3,4,6,12]
           fontsize=13)
plt.xlabel(r'$b$',fontsize=24)
#plt.legend()
plt.ylabel(r'$\vartheta$' +'\t  ', fontsize=24,rotation=0)

#plt.ylabel(r'$\vartheta$', fontsize=20, rotation=0)

# for color,bplot in zip(['pink','lightgreen'],[bplot1,bplot2]):
#     for patch in bplot['boxes']:
#         patch.set_facecolor(color)

#%%
##### Boxplot #3

fig1, ax1 = plt.subplots(figsize=(10,6))
ax1.grid(axis='x')
if titles_on: ax1.set_title(r'Distribution of $\vartheta$ for each $b$-sampling')
#labels = [ r'$~$', 1 , r'$~$', r'$~$', 2, r'$~$', r'$~$', 3, r'$~$',r'$~$', 4, r'$~$',r'$~$', 6, r'$~$',r'$~$', 12]

alpha = 0.55

tt_copy = []
tt_unif_copy = []
tdud = [np.NaN]
for i,b in enumerate(b_list):
    tt_copy.append(tt[i])
    tt_copy.append(tdud)
    tt_unif_copy.append(tdud)
    tt_unif_copy.append(tt_unif[i])
    if i!= len(b_list)-1:
        tt_copy.append(tdud)
        tt_unif_copy.append(tdud)

outlier_sz = 0.5

c = 'red'
bplot1 = ax1.boxplot(tt_copy, patch_artist=True, showcaps=True, #widths=0.5,
            boxprops=dict(facecolor=c, color=c,alpha=0.65),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, marker='o', markeredgecolor=c, alpha=0.5, markersize=outlier_sz),
            medianprops=dict(color=c, linewidth=2) )
c = 'g'
bplot2 = ax1.boxplot(tt_unif_copy, patch_artist=True, showcaps=True, #widths=0.5,
            boxprops=dict(facecolor=c, color=c,alpha = alpha),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c, alpha=0.5, markersize=outlier_sz),
            medianprops=dict(color=c,alpha=1,linewidth=2) )

ax1.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ['A', 'B'], loc='upper right')