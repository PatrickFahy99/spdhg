from __future__ import print_function, division
import numpy as np
import odl
from scipy.ndimage import convolve as sp_convolve
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
#from timeit import default_timer as timer
import time

###########################################
#################OPERATORS#################
###########################################

class RealFourierTransform(odl.Operator):
    
    def __init__(self, domain):
        """TBC
        
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.uniform_discr(0, 1, 10) ** 2
        >>> F = myOperators.RealFourierTransform(X)
        >>> x = X.one()
        >>> y = F(x)
        """
        #domain_complex = domain[0].complex_space
        #self.fourier = odl.trafos.DiscreteFourierTransform(domain_complex)
        
        
        #range = self.fourier.range.real_space ** 2
        range = domain
        
        super(RealFourierTransform, self).__init__(
                domain=domain, range=range, linear=True)
    
    def _call(self, x, out):
        #Fx = self.fourier(x[0].asarray() + 1j * x[1].asarray())
        Fx = np.fft.fftn(np.fft.fftshift(x[0].asarray() + 1j * x[1].asarray()),norm="ortho")
        Fx = np.fft.fftshift(Fx)
        out[0][:] = np.real(Fx)
        out[1][:] = np.imag(Fx)
        
        #out *= self.domain[0].cell_volume
                            
    @property
    def adjoint(self):
        op = self
        
        class RealFourierTransformAdjoint(odl.Operator):
    
            def __init__(self, op):        
                """TBC
                
                Parameters
                ----------
                TBC
                
                Examples
                --------
                >>> import odl
                >>> import operators as myOperators
                >>> X = odl.uniform_discr(0, 2, 10) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = odl.phantom.white_noise(A.range)
                >>> t1 = A(x).inner(y)
                >>> t2 = x.inner(A.adjoint(y))
                >>> t1 / t2
                
                >>> import odl
                >>> import operators as myOperators
                >>> X = odl.uniform_discr([-1, -1], [2, 1], [10, 30]) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = odl.phantom.white_noise(A.range)
                >>> t1 = A(x).inner(y)
                >>> t2 = x.inner(A.adjoint(y))
                >>> t1 / t2
                """
                self.op = op
                
                super(RealFourierTransformAdjoint, self).__init__(
                        domain=op.range, range=op.domain, linear=True)
            
            def _call(self, x, out):
                y = x[0].asarray() + 1j * x[1].asarray()
                #Fadjy = self.op.fourier.adjoint(y)
                y = np.fft.ifftshift(y)
                Fadjy = np.fft.ifftshift(np.fft.ifftn(y,norm="ortho"))
                out[0][:] = np.real(Fadjy)
                out[1][:] = np.imag(Fadjy)
                
                #out *= self.op.fourier.domain.size
        
            @property
            def adjoint(self):
                return op
        
        return RealFourierTransformAdjoint(op)
    
    @property
    def inverse(self):
        op = self
        
        class RealFourierTransformInverse(odl.Operator):
    
            def __init__(self, op):        
                """TBC
                
                Parameters
                ----------
                TBC
                
                Examples
                --------
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr(0, 2, 10) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = A(x)
                >>> (A.inverse(y)-x).norm()
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1], [2, 1], [10, 30]) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = A(x)
                >>> (A.inverse(y)-x).norm()
                """
                self.op = op
                
                super(RealFourierTransformInverse, self).__init__(
                        domain=op.range, range=op.domain, linear=True)
            
            def _call(self, x, out):
                y = x[0].asarray() + 1j * x[1].asarray()
                Fadjy = self.op.fourier.inverse(y)
                out[0][:] = np.real(Fadjy)
                out[1][:] = np.imag(Fadjy)
                
                out /= self.op.fourier.domain.cell_volume
        
            @property
            def inverse(self):
                return op
        
        return RealFourierTransformInverse(op)



class RealMultiplyOperator(odl.Operator):    # Matthias, Claire and I for MRI recon
    
    def __init__(self, domain, element):
        """TBC
        
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.uniform_discr(0, 1, 10) ** 2
        >>> F = myOperators.RealFourierTransform(X)
        >>> x = X.one()
        >>> y = F(x)
        """
        self.element = element
        
        super(RealMultiplyOperator, self).__init__(
                domain=domain, range=domain, linear=True)
    
    def _call(self, x, out):
        Y = x[0].asarray() + 1j * x[1].asarray()
        Y *= self.element    # element  is a complex numpy array
        out[0][:] = np.real(Y)
        out[1][:] = np.imag(Y)
        
        #out *= self.domain[0].cell_volume
                            
    @property
    def adjoint(self):
        """TBC
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import realmultop
        >>> X = odl.uniform_discr(0, 2, 10) ** 2
        >>> z = np.array(X[0].one()) + 1j*np.array(X[1].one())
        >>> A = realmultop.RealMultiplyOperator(X,z)
        >>> x = odl.phantom.white_noise(A.domain)
        >>> y = odl.phantom.white_noise(A.range)
        >>> t1 = A(x).inner(y)
        >>> t2 = x.inner(A.adjoint(y))
        >>> t1 / t2
        
        >>> import odl
        >>> import myOperators
        >>> X = odl.uniform_discr([-1, -1], [2, 1], [10, 30]) ** 2
        >>> A = myOperators.RealFourierTransform(X)
        >>> x = odl.phantom.white_noise(A.domain)
        >>> y = odl.phantom.white_noise(A.range)
        >>> t1 = A(x).inner(y)
        >>> t2 = x.inner(A.adjoint(y))
        >>> t1 / t2
        """
        return RealMultiplyOperator(self.domain, np.conj(self.element)) 



class UnitaryRealFourierTransform(odl.Operator):
    
    def __init__(self, domain):
        """TBC
        
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.uniform_discr(0, 1, 10) ** 2
        >>> F = myOperators.RealFourierTransform(X)
        >>> x = X.one()
        >>> y = F(x)
        """
        domain_complex = domain[0].complex_space
        self.fourier = odl.trafos.DiscreteFourierTransform(domain_complex)
        
        range = self.fourier.range.real_space ** 2
        
        super(UnitaryRealFourierTransform, self).__init__(
                domain=domain, range=range, linear=True)
        
        self.factor = domain.one().norm() / 2
        self.factor = np.sqrt(self.domain[0].cell_volume / self.fourier.domain.size)
    
    def _call(self, x, out):
        Fx = self.fourier(x[0].asarray() + 1j * x[1].asarray())
        out[0][:] = np.real(Fx)
        out[1][:] = np.imag(Fx)
        
        out *= self.factor
                            
    @property
    def adjoint(self):
        op = self
        
        class RealFourierTransformAdjoint(odl.Operator):
    
            def __init__(self, op):        
                """TBC
                
                Parameters
                ----------
                TBC
                
                Examples
                --------
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr(0, 2, 10) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = odl.phantom.white_noise(A.range)
                >>> t1 = A(x).inner(y)
                >>> t2 = x.inner(A.adjoint(y))
                >>> t1 / t2
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1], [2, 1], [10, 30]) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = odl.phantom.white_noise(A.range)
                >>> t1 = A(x).inner(y)
                >>> t2 = x.inner(A.adjoint(y))
                >>> t1 / t2
                """
                self.op = op
                
                super(RealFourierTransformAdjoint, self).__init__(
                        domain=op.range, range=op.domain, linear=True)
            
            def _call(self, x, out):
                y = x[0].asarray() + 1j * x[1].asarray()
                Fadjy = self.op.fourier.adjoint(y)
                out[0][:] = np.real(Fadjy)
                out[1][:] = np.imag(Fadjy)
                
                out /= self.op.factor
        
            @property
            def adjoint(self):
                return op
        
        return RealFourierTransformAdjoint(op)

    @property
    def inverse(self):
        """Inverse Fourier transform
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.uniform_discr(0, 2, 10) ** 2 
        >>> A = myOperators.RealFourierTransform(X)
        >>> x = odl.phantom.white_noise(A.domain)
        >>> (A.inverse(A(x)) - x).norm()
        
        >>> import odl
        >>> import myOperators
        >>> X = odl.uniform_discr([-1, -1], [2, 1], [10, 30]) ** 2
        >>> A = myOperators.RealFourierTransform(X)
        >>> x = odl.phantom.white_noise(A.domain)
        >>> (A.inverse(A(x)) - x).norm()
        """
        return self.adjoint

    
class Complex2Real(odl.Operator):
    
    def __init__(self, domain):
        """TBC
        
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.cn(3)
        >>> J = myOperators.Complex2Real(X)
        >>> x = X.one()
        >>> y = J(x)
        
        >>> import odl
        >>> import myOperators
        >>> X = odl.cn(3)
        >>> A = myOperators.Complex2Real(X)
        >>> x = odl.phantom.white_noise(A.domain)
        >>> y = odl.phantom.white_noise(A.range)
        >>> t1 = A(x).inner(y)
        >>> t2 = x.inner(A.adjoint(y))
        """
        
        super(Complex2Real, self).__init__(domain=domain, 
                                           range=domain.real_space ** 2, 
                                           linear=True)
        
    def _call(self, x, out):
        out[0][:] = np.real(x)
        out[1][:] = np.imag(x)
                            
    @property
    def adjoint(self):
        return Real2Complex(self.range)
    
    
class Real2Complex(odl.Operator):
            
    def __init__(self, domain):
        """TBC
        Parameters
        ----------
        TBC
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.rn(3) ** 2
        >>> J = myOperators.Real2Complex(X)
        >>> x = X.one()
        >>> y = J(x)
        """
        
        super(Real2Complex, self).__init__(domain=domain, 
             range=domain[0].complex_space, linear=True)
            
    def _call(self, x, out):
        out[:] = x[0].asarray() + 1j * x[1].asarray()
 
    @property
    def adjoint(self):
        return Complex2Real(self.range)    
    
    
def Magnitude(space):
    return odl.PointwiseNorm(space)


class Subsampling(odl.Operator):
    '''  '''
    def __init__(self, domain, range, margin=None):
        """TBC
        
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.rn((8, 8, 8))
        >>> Y = odl.rn((2, 2))  
        >>> S = myOperators.Subsampling(X, Y)
        >>> x = X.one()
        >>> y = S(x)
        """
        domain_shape = np.array(domain.shape)
        range_shape = np.array(range.shape)
        
        len_domain = len(domain_shape)
        len_range = len(range_shape)
        
        if margin is None:
            margin = 0
                
        if np.isscalar(margin):
            margin = [(margin, margin)] * len_domain

        self.margin = np.array(margin).astype('int')

        self.margin_index = []
        for m in self.margin:
            m0 = m[0]
            m1 = m[1]
            
            if m0 == 0:
                m0 = None
            
            if m1 == 0:
                m1 = None
            else:
                m1 = -m1

            self.margin_index.append((m0, m1))
                        
        if len_domain < len_range:
            ValueError('TBC')
        else:
            if len_domain > len_range:
                range_shape = np.append(range_shape, np.ones(len_domain - len_range))
                
            self.block_size = tuple(((domain_shape - np.sum(self.margin, 1)) / range_shape).astype('int'))

        super(Subsampling, self).__init__(domain=domain, range=range, 
                                          linear=True)
        
    def _call(self, x, out):
        m = self.margin_index
        if m is not None:
            if len(m) == 1:
                x = x[m[0][0]:m[0][1]]
            elif len(m) == 2:
                x = x[m[0][0]:m[0][1], m[1][0]:m[1][1]]
            elif len(m) == 3:
                x = x[m[0][0]:m[0][1], m[1][0]:m[1][1], m[2][0]:m[2][1]]
            else:
                ValueError('TBC')
            
        out[:] = np.squeeze(block_reduce(x, block_size=self.block_size, func=np.mean))
                # block_reduce: returns Down-sampled image with same number of dimensions as input image.
                            
    @property
    def adjoint(self):
        op = self
            
        class SubsamplingAdjoint(odl.Operator):
            
            def __init__(self, op):
                """TBC
        
                Parameters
                ----------
                TBC
        
                Examples
                --------
                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 8))
                >>> Y = odl.rn((2, 2))  
                >>> S = myOperators.Subsampling(X, Y)
                >>> x = odl.phantom.white_noise(X)
                >>> y = odl.phantom.white_noise(Y)
                >>> S(x).inner(y) / x.inner(S.adjoint(y))
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 15))
                >>> Y = odl.rn((2, 2))  
                >>> S = myOperators.Subsampling(X, Y)
                >>> x = odl.phantom.white_noise(X)
                >>> y = odl.phantom.white_noise(Y)
                >>> S(x).inner(y) / x.inner(S.adjoint(y))
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1, -.1], [1, 1, .1], (160, 160, 15))
                >>> Y = odl.uniform_discr([-1, -1], [1, 1], (40, 40))
                >>> S = myOperators.Subsampling(X, Y)
                >>> x = odl.phantom.white_noise(X)
                >>> y = odl.phantom.white_noise(Y)
                >>> S(x).inner(y) / x.inner(S.adjoint(y))
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 8))
                >>> Y = odl.rn((2, 2))  
                >>> S = myOperators.Subsampling(X, Y, margin=1)
                >>> x = odl.phantom.white_noise(X)
                >>> y = odl.phantom.white_noise(Y)
                >>> S(x).inner(y) / x.inner(S.adjoint(y))
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1, -.1], [1, 1, .1], (160, 160, 21))
                >>> Y = odl.uniform_discr([-1, -1], [1, 1], (40, 40))
                >>> S = myOperators.Subsampling(X, Y, margin=((0, 0),(0, 0),(3, 3)))
                >>> x = odl.phantom.white_noise(X)
                >>> y = odl.phantom.white_noise(Y)
                >>> S(x).inner(y) / x.inner(S.adjoint(y))
                """
                domain = op.range
                range = op.domain
                self.block_size = op.block_size
                self.margin = op.margin
                self.margin_index = op.margin_index
                
                x = range.zero()
                m = self.margin_index
                if m is not None:
                    if len(m) == 1:
                        x[m[0][0]:m[0][1]] = 1
                    elif len(m) == 2:
                        x[m[0][0]:m[0][1], m[1][0]:m[1][1]] = 1
                    elif len(m) == 3:
                        x[m[0][0]:m[0][1], m[1][0]:m[1][1], m[2][0]:m[2][1]] = 1
                    else:
                        ValueError('TBC')
                else:
                    x = range.one()
                
                self.factor = x.inner(range.one()) / domain.one().inner(domain.one())
                
                super(SubsamplingAdjoint, self).__init__(
                        domain=domain, range=range, linear=True)
                    
            def _call(self, x, out):
                for i in range(len(x.shape), len(self.block_size)):     
                    x = np.expand_dims(x, axis=i)
                                    
                if self.margin is None:
                    out[:] = np.kron(x, np.ones(self.block_size)) / self.factor                     
                else:      
                    y = np.kron(x, np.ones(self.block_size)) / self.factor                     
                    out[:] = np.pad(y, self.margin, mode='constant')

            @property
            def adjoint(self):
                return op
                    
        return SubsamplingAdjoint(self)

            
##################################################
#################HELPER FUNCTIONS#################
##################################################

def get_cartesian_sampling(X, ufactor):
    sampling_array = np.zeros(X.shape)
    sampling_array[::ufactor, :] = 1
    sampling_array_vis = sampling_array.copy()
    sampling_array  = np.fft.fftshift(sampling_array) 
    sampling_points = np.nonzero(sampling_array)
    return odl.SamplingOperator(X, sampling_points), sampling_array_vis

def get_random_sampling(X, prob):
    sampling_array = np.random.choice([0,1], size=X.shape,p=[1-prob,prob])
    sampling_array_vis = sampling_array.copy()
    sampling_array  = np.fft.fftshift(sampling_array)               
    sampling_points = np.nonzero(sampling_array)
    return odl.SamplingOperator(X, sampling_points), sampling_array_vis

def get_radial_sampling(X, num_angles=10, block=0):
    sampling_array = np.zeros(X.shape)
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    center = [int(np.round(i/2)) for i in X.shape]
    for r in range(int(np.sqrt(2)*int(np.max(X.shape)/2))):
        for i in range(num_angles):
            phi = angles[i]
            idx = np.ceil([r*np.cos(phi), r*np.sin(phi)])
            idx = [int(i) for i in idx]
            glob_idx0 = center[0] + idx[0]
            glob_idx1 = center[1] + idx[1]
            if 0 <= glob_idx0 < X.shape[0] and 0 <= glob_idx1 < X.shape[1]:
                sampling_array[glob_idx0, glob_idx1] = 1
    sampling_array[center[0] - int(block/2) + 1 : center[0] + int(block/2) + 1, 
                   center[1] - int(block/2) + 1 : center[1] + int(block/2) + 1] = 1
    sampling_array_vis = sampling_array.copy()                   
    sampling_array  = np.fft.fftshift(sampling_array)               
    sampling_points = np.nonzero(sampling_array)            
    return odl.SamplingOperator(X, sampling_points), sampling_array_vis

def ChooseSampling(name, space):
    sampling_parts = name.split('-')
    if sampling_parts[0] == 'cartesian':
        r = int(sampling_parts[1])
        sampling_op, mask = get_cartesian_sampling(space, r)
    elif sampling_parts[0] == 'random':
        p = float(sampling_parts[1])
        sampling_op, mask = get_random_sampling(space, p)
    elif sampling_parts[0] == 'radial':
        num_angles = int(sampling_parts[1])
        sampling_op, mask = get_radial_sampling(space, num_angles)                
    elif sampling_parts[0] == 'radialbl':
        num_angles = int(sampling_parts[1])
        sampling_op, mask = get_radial_sampling(space, num_angles, block=10)                       
    return sampling_op, mask                

def GenerateSensitivityMap(FOV=0.24, size=128, Nc=1, R=0.04, D=0.15):
    resolution = FOV/size
    x = np.arange(0,FOV,resolution)
    x = x - x[int(np.ceil(len(x)/2))]
    y = np.arange(0,FOV,resolution)
    y = y - y[int(np.ceil(len(y)/2))]    
    if isinstance(Nc,int):
        dalpha = 2*np.pi/Nc
        alpha = np.arange(0,2*np.pi,dalpha)
    else:
        alpha = Nc
    Nc = len(alpha)
    S = np.zeros([Nc,len(y),len(x)],dtype=complex)
    Nangles = 60
    dtheta = 2*np.pi/Nangles
    theta = np.arange(-np.pi,np.pi,dtheta)
    [Y,T,X] = np.meshgrid(y,theta,x)
    for i in range(Nc):
        x = X*np.cos(alpha[i]) - Y*np.sin(alpha[i])
        y = X*np.sin(alpha[i]) + Y*np.cos(alpha[i])
        s = np.exp(1j*alpha[i])*(-R + y*np.cos(T) - 1j*(D-x)*np.cos(T))/((D-x)**2 + y**2 + R**2 - 2*R*y*np.cos(T))**(3/2)
        S[i,:,:] = dtheta*sum(s)
    S = S/np.amax(abs(S))
    return S


    #baruch
def GenerateSensitivityMap2(FOV=0.24, size=[180,230], Nc=1, R=0.04, D=0.15):
    resolutionx = FOV/size[1]
    resolutiony = FOV/size[0]
    x = np.arange(0,FOV,resolutionx)
    x = x - x[int(np.ceil(len(x)/2))]
    y = np.arange(0,FOV,resolutiony)
    y = y - y[int(np.ceil(len(y)/2))]    
    if isinstance(Nc,int):
        dalpha = 2*np.pi/Nc
        alpha = np.arange(0,2*np.pi,dalpha)
    else:
        alpha = Nc
    Nc = len(alpha)
    S = np.zeros([Nc,len(y),len(x)],dtype=complex)
    Nangles = 60
    dtheta = 2*np.pi/Nangles
    theta = np.arange(-np.pi,np.pi,dtheta)
    [Y,T,X] = np.meshgrid(y,theta,x)
    for i in range(Nc):
        x = X*np.cos(alpha[i]) - Y*np.sin(alpha[i])
        y = X*np.sin(alpha[i]) + Y*np.cos(alpha[i])
        s = np.exp(1j*alpha[i])*(-R + y*np.cos(T) - 1j*(D-x)*np.cos(T))/((D-x)**2 + y**2 + R**2 - 2*R*y*np.cos(T))**(3/2)
        S[i,:,:] = dtheta*sum(s)
    S = S/np.amax(abs(S))
    return S


class CallbackStore(odl.solvers.Callback):
    """Callback to store function values"""
    def __init__(self, n, f, g, A):
        self.n = n
        self.f = f
        self.g = g
        self.A = A
        self.iter = 0
        self.out = []
    def __call__(self, w, **kwargs):
        if np.remainder(self.iter,self.n)==0:
            x = w[0].copy()
            obj_fun = self.f(self.A(x)) + self.g(x)
            self.out.append(obj_fun)
        self.iter += 1


########## Baruch 2 ##########
class CallbackStore2(odl.solvers.Callback):
    """Callback to store Phi(x), ||x-x*||, ||xk-xk-1|| and time """
    def __init__(self, n, f, g, A, sol):
        self.n = n
        self.f = f
        self.g = g
        self.A = A
        self.sol = sol
        xzero = A.domain.zero()
        self.Phi0 = f(A(xzero)) + g(xzero)
        self.PhiS = f(A(sol)) + g(sol)
        self.norm_s = sol.norm()
        self.xold = xzero.copy()
        self.norm_xone = 0
        self.iter = 1
        self.out = [1.0]                           # Obj functional
        self.err = [1.0]                           # Error w.r.t. solution
        self.dis = [1.0]                           # Distance between iterates
        self.time= [0.0]
        self.ep_time = 0.0
    def __call__(self, w, **kwargs):
        it_time = w[2]
        self.ep_time = self.ep_time + it_time
        if np.remainder(self.iter,self.n)==0:
            
            x = w[0].copy()
            
            #### Objective
            #obj_fun = self.f(self.A(x)) + self.g(x)
            #obj_fun = (obj_fun - self.PhiS)/(self.Phi0 - self.PhiS)
            obj_fun = 1
            self.out.append(abs(obj_fun))
          
            #### Error
            err = (x - self.sol).norm()/self.norm_s
            self.err.append(err)
            
            #### Distance
            # xold_norm = self.xold.norm()
            # if xold_norm == 0:
            #     dis = 1 
            # else:
            #     dis = (x - self.xold).norm()/self.xold.norm() 
            # self.dis.append(dis)
            # self.xold = x.copy()
            
            # if self.norm_xone == 0:
            #     norm_x = x.copy().norm()
            #     if norm_x > 0:
            #         self.norm_xone = norm_x
            #     dis = 1
            # else:
            #     dis = (x - self.xold).norm()/self.norm_xone 
            dis = 1
            self.dis.append(dis)
            self.xold = x.copy()
            
            #### Time
            new_time = self.time[-1] + self.ep_time
            self.time.append( new_time )
            self.ep_time = 0
        self.iter += 1
        
        
########## Baruch 2 brains ##########
class CallbackStore2brains(odl.solvers.Callback):
    """Callback to store Phi(x), ||x-x*||, ||xk-xk-1|| and time """
    def __init__(self, n, f, g, A, sol, b, contrast, overwrite, filename, Prox_Niter, prox_niter):
        self.n = n
        self.f = f
        self.g = g
        self.b = b
        self.A = A
        self.M = Magnitude(A.domain)
        self.sol = sol
        self.contrast = contrast
        self.overwrite = overwrite
        self.filename = filename
        self.Prox_Niter = Prox_Niter
        self.prox_niter = prox_niter
        xzero = A.domain.zero()
        self.Phi0 = f(A(xzero)) + g(xzero)
        self.PhiS = f(A(sol)) + g(sol)
        self.norm_s = sol.norm()
        self.xold = xzero.copy()
        self.norm_xone = 0
        self.iter = 1
        self.out = [1.0]                           # Obj functional
        self.err = [1.0]                           # Error w.r.t. solution
        self.dis = [1.0]                           # Distance between iterates
        self.time= [0.0]
        self.ep_time = 0.0
    def __call__(self, w, **kwargs):
        it_time = w[2]
        self.ep_time = self.ep_time + it_time
        if np.remainder(self.iter,self.n)==0:
            
            x = w[0].copy()
            
            #### Objective
            obj_fun = self.f(self.A(x)) + self.g(x)
            obj_fun = (obj_fun - self.PhiS)/(self.Phi0 - self.PhiS)
            self.out.append(abs(obj_fun))
          
            #### Error
            err = (x - self.sol).norm()/self.norm_s
            self.err.append(err)
            
            #### Distance
            if self.norm_xone == 0:
                norm_x = x.copy().norm()
                if norm_x > 0:
                    self.norm_xone = norm_x
                dis = 1
            else:
                dis = (x - self.xold).norm()/self.norm_xone 
            self.dis.append(dis)
            self.xold = x.copy()
            
            #### Print Image
            plt.rcParams['figure.dpi'] = 780
            plt.rcParams['savefig.dpi'] = 780
            
            im = self.M(x)
            epoch = self.iter//self.n
            np.save('brains/'+self.filename+'_'+self.Prox_Niter+str(self.prox_niter)+'_b'+str(self.b)+'_'+str(epoch)+'.npy',x)
            if self.overwrite:
                plt.imshow(im, cmap='gray', vmin=0, vmax=self.contrast), plt.axis('off')
                plt.savefig('brains/'+self.filename+'_'+self.Prox_Niter+str(self.prox_niter)+'_b'+str(self.b)+'_'+str(epoch)+'.png', bbox_inches='tight', pad_inches=0)
                plt.close()
            plt.figure(epoch)
            plt.imshow(im, cmap='gray', vmin=0, vmax=self.contrast), plt.axis('off')
            plt.title('$b={}$, epoch {}'.format(self.b, epoch))
            plt.show()
            
            #### Save Results
            
            
            #### Time
            new_time = self.time[-1] + self.ep_time
            self.time.append( new_time )
            self.ep_time = 0
        self.iter += 1


########## Baruch 3 ##########
class CallbackStore3(odl.solvers.Callback):
    """Callback to store Phi(x) and ||x-x*|| """
    def __init__(self, n, f, g, A, sol):
        self.n = n
        self.f = f
        self.g = g
        self.A = A
        self.sol = sol
        self.norm_s = sol.norm()
        self.xold = A.domain.zero()
        self.iter = 0
        self.out = []                           # Obj functional
        self.err = []                           # Error w.r.t. solution
        self.dis = []                           # Distance between iterates
    def __call__(self, w, **kwargs):
        if np.remainder(self.iter,self.n)==0:
            x = w[0].copy()
            obj_fun = self.f(self.A(x)) + self.g(x)
            self.out.append(obj_fun)
            err = (x - self.sol).norm()/self.norm_s
            self.err.append(err)
            dis = (x - self.xold).norm()/max(self.xold.norm(),1)
            self.dis.append(dis)
            self.xold = x.copy()
        self.iter += 1


class Convolution(odl.Operator):

    def __init__(self, space, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'constant'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(space, space, linear=True)

    def _call(self, x, out):
        sp_convolve(x, self.kernel, output=out.asarray(),
                    mode=self.boundary_condition, origin=self.origin)

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            if self.domain.ndim == 2:
                kernel = np.fliplr(np.flipud(self.kernel.copy().conj()))
                kernel = self.kernel.space.element(kernel)
            else:
                raise NotImplementedError('"adjoint_kernel" only defined for '
                                          '2d kernels')

            origin = [0, ] * self.domain.ndim
            for i in range(self.domain.ndim):
                if np.mod(self.kernel.shape[i], 2) == 0:
                    origin[i] = -1

            for i in range(self.domain.ndim):
                if kernel.shape[i] < 3 and origin[i] == -1:
                    NotImplementedError('Shifted origins are only implemented '
                                        'for kernels of size 3 or larger.')

            self.__adjoint = Convolution(self.domain, kernel, origin, self)

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)


class ConvolutionEmbedding(odl.Operator):

    def __init__(self, domain, range, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'constant'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(domain, range, linear=True)

    def _call(self, x, out):
        sp_convolve(self.kernel, x, output=out.asarray(),
                    mode=self.boundary_condition, origin=self.origin)

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            origin = [0, ] * self.domain.ndim
            for i in range(self.domain.ndim):
                if np.mod(self.kernel.shape[i], 2) == 0:
                    origin[i] = -1

            for i in range(self.domain.ndim):
                if self.kernel.shape[i] < 3 and origin[i] == -1:
                    NotImplementedError('Shifted origins are only implemented '
                                        'for kernels of size 3 or larger.')

            self.__adjoint = ConvolutionEmbeddingAdjoint(self.range,
                                                         self.domain,
                                                         self.kernel,
                                                         origin, self)

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)


class ConvolutionEmbeddingAdjoint(odl.Operator):

    def __init__(self, domain, range, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'constant'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(domain, range, linear=True)

    def _call(self, x, out):
        if not self.domain.ndim == 2:
            raise NotImplementedError('adjoint only defined for 2d domains')

        out_a = out.asarray()
        x_a = x.asarray()
        k_a = self.kernel.asarray()

        n = x.shape
        s = out.shape[0] // 2, out.shape[1] // 2

        for i in range(out_a.shape[0]):

            if n[0] > 1:
                ix1, ix2 = max(i - s[0], 0), min(n[0] + i - s[0], n[0])
                ik1, ik2 = max(s[0] - i, 0), min(n[0] - i + s[0], n[0])
            else:
                ix1, ix2 = 0, 1
                ik1, ik2 = 0, 1

            for j in range(out_a.shape[1]):
                if n[1] > 1:
                    jx1, jx2 = max(j - s[1], 0), min(n[1] + j - s[1], n[1])
                    jk1, jk2 = max(s[1] - j, 0), min(n[1] - j + s[1], n[1])
                else:
                    jx1, jx2 = 0, 1
                    jk1, jk2 = 0, 1

                out_a[i, j] = np.sum(x_a[ix1:ix2, jx1:jx2] *
                                     k_a[ik1:ik2, jk1:jk2])

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            NotImplementedError('Can only be called as an "adjoint" of '
                                '"ConvolutionEmbedding".')

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)
    
    
##########################################################
#################### Baruch ##############################


def spdhg2(x, f, g, A, tau, sigma, niter, **kwargs):
    """ calls spdhg_generic2 which uses Callback2 """

    # Probabilities
    prob = kwargs.pop('prob', None)
    if prob is None:
        prob = [1 / len(A)] * len(A)

    # Selection function
    fun_select = kwargs.pop('fun_select', None)
    if fun_select is None:
        def fun_select(x):
            return [int(np.random.choice(len(A), 1, p=prob))]

    # Dual variable
    y = kwargs.pop('y', None)
    extra = [1 / p for p in prob]

    spdhg_generic2(x, f, g, A, tau, sigma, niter, fun_select=fun_select, y=y,
                  extra=extra, **kwargs)


def spdhg_generic2(x, f, g, A, tau, sigma, niter, **kwargs):
    """ Callback2 stores computational time """

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    # Dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = A.range.zero()

    # Adjoint of dual variable
    z = kwargs.pop('z', None)
    if z is None:
        if y.norm() == 0:
            z = A.domain.zero()
        else:
            z = A.adjoint(y)

    # Strong convexity of g
    mu_g = kwargs.pop('mu_g', None)
    if mu_g is None:
        update_proximal_primal = False
    else:
        update_proximal_primal = True

    # Global extrapolation factor theta
    theta = kwargs.pop('theta', 1)

    # Second extrapolation factor
    extra = kwargs.pop('extra', None)
    if extra is None:
        extra = [1] * len(sigma)

    # Selection function
    fun_select = kwargs.pop('fun_select', None)
    if fun_select is None:
        def fun_select(x):
            return [int(np.random.choice(len(A), 1, p=1 / len(A)))]

    # Initialize variables
    z_relax = z.copy()
    dz = A.domain.element()
    y_old = A.range.element()

    # Save proximal operators
    proximal_dual_sigma = [fi.convex_conj.proximal(si)
                           for fi, si in zip(f, sigma)]
    proximal_primal_tau = g.proximal(tau)

    # run the iterations
    for k in range(niter):
        
        # measure time
        #start_time = timer()
        start_time = time.time()
        
        # select block
        selected = fun_select(k)
        
        # end_time = time.process_time()
        # it_time = end_time - start_time
        # print(it_time)
        # start_time = time.process_time()

        # update primal variable
        # tmp = x - tau * z_relax; z_relax used as tmp variable
        z_relax.lincomb(1, x, -tau, z_relax)
        
        # end_time = time.process_time()
        # it_time = end_time - start_time
        # print(it_time)
        # start_time = time.process_time()
        
        # x = prox(tmp)
        proximal_primal_tau(z_relax, out=x)
        
        # end_time = time.process_time()
        # it_time = end_time - start_time
        # print('prox= ',it_time)
        # start_time = time.time()

        # # update extrapolation parameter theta
        # if update_proximal_primal:
        #     theta = float(1 / np.sqrt(1 + 2 * mu_g * tau))

        # update dual variable and z, z_relax
        z_relax.assign(z)
        for i in selected:

            # save old yi
            y_old[i].assign(y[i])

            # end_time = time.time()
            # it_time = end_time - start_time
            # print('assign=', it_time)
            # start_time = time.time()

            # y[i] = Ai(x)
            A[i](x, out=y[i])

            # end_time = time.time()
            # it_time = end_time - start_time
            # print('A= ',it_time)
            # start_time = time.time()

            # y[i] = y_old + sigma_i * Ai(x)
            y[i].lincomb(1, y_old[i], sigma[i], y[i])

            # end_time = time.time()
            # it_time = end_time - start_time
            # print(it_time)
            # start_time = time.time()

            # y[i] = prox(y[i])
            proximal_dual_sigma[i](y[i], out=y[i])

            # end_time = time.time()
            # it_time = end_time - start_time
            # print('prox_y=', it_time)
            # start_time = time.time()

            # update adjoint of dual variable
            y_old[i].lincomb(-1, y_old[i], 1, y[i])
            A[i].adjoint(y_old[i], out=dz)
            z += dz

            # end_time = time.time()
            # it_time = end_time - start_time
            # print('A*= ', it_time)
            #start_time = time.time()

            # compute extrapolation
            z_relax.lincomb(1, z_relax, 1 + theta * extra[i], dz)

            # end_time = time.time()
            # it_time = end_time - start_time
            # print('lincomb=', it_time)
            #start_time = time.process_time()

        # update the step sizes tau and sigma for acceleration
        if update_proximal_primal:
            for i in range(len(sigma)):
                sigma[i] /= theta
            tau *= theta

            proximal_dual_sigma = [fi.convex_conj.proximal(si)
                                    for fi, si in zip(f, sigma)]
            proximal_primal_tau = g.proximal(tau)
        
        # measure time
        end_time = time.time()
        it_time = end_time - start_time
        # print(it_time)
        
        # Callback2 which stores computational time
        if callback is not None:
            callback([x, y, it_time])
            
            
            
