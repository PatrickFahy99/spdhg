import numpy as np
import odl
from odl.solvers.functional.functional import Functional
from odl.solvers import (GroupL1Norm, ZeroFunctional, IndicatorBox)
from odl.operator.operator import Operator
from odl.operator.default_ops import IdentityOperator
from algorithms import fgp_dual


###############    Functionals    ###############

def resetTV(domain, reg_term=1., prox_niter=5):
    prox_options = {}
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    prox_options['tol'] = None
    prox_options['niter'] = prox_niter  #100
    #g = TV(domain, alpha=reg_term, prox_options=prox_options.copy())
    g = TV_new(domain, alpha=reg_term, prox_options=prox_options.copy())
    return g

def total_variation(domain, grad): 
    L1 = GroupL1Norm(grad.range, exponent=2)
    return L1 * grad


def generate_vfield_from_sinfo(sinfo, grad, eta=1e-2):
    sinfo_grad = grad(sinfo)
    grad_space = grad.range
    norm = odl.PointwiseNorm(grad_space, 2)
    norm_sinfo_grad = norm(sinfo_grad)
    max_norm = np.max(norm_sinfo_grad)
    eta_scaled = eta * max(max_norm, 1e-4)
    norm_eta_sinfo_grad = np.sqrt(norm_sinfo_grad ** 2 +
                                  eta_scaled ** 2)
    xi = grad_space.element([g / norm_eta_sinfo_grad for g in sinfo_grad])
    return xi

    
def project_on_fixed_vfield(domain, vfield):
        class OrthProj(Operator):
            def __init__(self):
                super(OrthProj, self).__init__(domain, domain, linear=True)
            def _call(self, x, out):
                xi = vfield
                Id = IdentityOperator(domain)
                xiT = odl.PointwiseInner(domain, xi)
                xixiT = odl.BroadcastOperator(*[x*xiT for x in xi])
                gamma = 1
                P = (Id - gamma * xixiT)
                out.assign(P(x))
            @property
            def adjoint(self):
                return self
            @property
            def norm(self):
                return 1.
        return OrthProj()


########################    Total Variation     #######################

class TV(Functional):

    def __init__(self, domain, alpha=1., sinfo=None, NonNeg=False,
                 prox_options={}, gamma=1, eta=1e-2):
                 
        if isinstance(domain, odl.ProductSpace):
            grad_basic = odl.Gradient(
                    domain[0], method='forward', pad_mode='symmetric')
            
            pd = [odl.discr.diff_ops.PartialDerivative(
                    domain[0], i, method='forward', pad_mode='symmetric') 
                  for i in range(2)]
            cp = [odl.operator.ComponentProjection(domain, i) 
                  for i in range(2)]
                
            if sinfo is None:
                self.grad = odl.BroadcastOperator(
                        *[pd[i] * cp[j]
                          for i in range(2) for j in range(2)])
                
            else:
                vfield = gamma * generate_vfield_from_sinfo(sinfo, grad_basic, eta)
                inner = odl.PointwiseInner(domain, vfield) * grad_basic
                self.grad = odl.BroadcastOperator(
                        *[pd[i] * cp[j] - vfield[i] * inner * cp[j] 
                          for i in range(2) for j in range(2)])
            
            self.grad.norm = self.grad.norm(estimate=True)
            
        else:
            grad_basic = odl.Gradient(
                    domain, method='forward', pad_mode='symmetric')
            
            if sinfo is None:
                self.grad = grad_basic
            else:
                vfield = gamma * generate_vfield_from_sinfo(sinfo, grad_basic, eta)
                P = project_on_fixed_vfield(grad_basic.range, vfield)
                self.grad = P * grad_basic
                
            grad_norm = 2 * np.sqrt(sum(1 / grad_basic.domain.cell_sides**2))
            self.grad.norm = grad_norm
        
        self.tv = total_variation(domain, grad=self.grad)

        if NonNeg is True:
            self.nn = IndicatorBox(domain, 0, np.inf)
        else:
            self.nn = ZeroFunctional(domain)            

        if 'name' not in prox_options:
            prox_options['name'] = 'FGP'
        if 'warmstart' not in prox_options:
            prox_options['warmstart'] = True
        if 'niter' not in prox_options:
            prox_options['niter'] = 5
        if 'p' not in prox_options:
            prox_options['p'] = None
        if 'tol' not in prox_options:
            prox_options['tol'] = None

        prox_options['grad'] = self.grad
        prox_options['proj_P'] = self.tv.left.convex_conj.proximal(0)
        prox_options['proj_C'] = self.nn.proximal(1)

        self.prox_options = prox_options
        self.alpha = alpha

        super().__init__(space=domain, linear=False, grad_lipschitz=0)

    def __call__(self, x):
        if self.alpha == 0:
            return 0
        else:
            nn = self.nn(x)
            out = self.alpha * self.tv(x) + nn
            return out

    @property
    def proximal(self):
        alpha = self.alpha
        prox_options = self.prox_options
        space = self.domain
        
        class ProximalTV(Operator):
            def __init__(self, sigma):
                self.sigma = float(sigma)
                self.prox_options = prox_options
                self.alpha = float(alpha)
                super(ProximalTV, self).__init__(
                    domain=space, range=space, linear=False)
    
            def _call(self, z, out):
                sigma = self.sigma * self.alpha
                if sigma == 0:
                    out.assign(z)
                else:
                    opts = self.prox_options
    
                    grad = opts['grad']
                    proj_C = opts['proj_C']
                    proj_P = opts['proj_P']
    
                    if opts['name'] == 'FGP':
                        if opts['warmstart']:
                            if opts['p'] is None:
                                opts['p'] = grad.range.zero()
    
                            p = opts['p']
                        else:
                            p = grad.range.zero()
    
                        niter = opts['niter']
                        out.assign(fgp_dual(p, z, sigma, niter, grad, proj_C,
                                          proj_P, tol=opts['tol']))
    
                    else:
                        raise NotImplementedError('Not yet implemented')
                    
        return ProximalTV
    
    
########################    Total Variation new    #######################

class TV_new(Functional):

    def __init__(self, domain, alpha=1., prox_options={}):
                 
        if isinstance(domain, odl.ProductSpace):
            grad_basic = odl.Gradient(
                    domain[0], method='forward', pad_mode='symmetric')
            
            pd = [odl.discr.diff_ops.PartialDerivative(
                    domain[0], i, method='forward', pad_mode='symmetric') 
                  for i in range(2)]
            cp = [odl.operator.ComponentProjection(domain, i) 
                  for i in range(2)]
                
           
            self.grad = odl.BroadcastOperator(
                    *[pd[i] * cp[j]
                      for i in range(2) for j in range(2)])
                
            self.grad.norm = self.grad.norm(estimate=True)
            
        else:
            grad_basic = odl.Gradient(
                    domain, method='forward', pad_mode='symmetric')
            
            self.grad = grad_basic
                
            grad_norm = 2 * np.sqrt(sum(1 / grad_basic.domain.cell_sides**2))
            self.grad.norm = grad_norm
        
        self.tv = total_variation(domain, grad=self.grad)

        if 'name' not in prox_options:
            prox_options['name'] = 'FGP'
        if 'warmstart' not in prox_options:
            prox_options['warmstart'] = True
        if 'niter' not in prox_options:
            prox_options['niter'] = 5
        if 'p' not in prox_options:
            prox_options['p'] = None
        if 'tol' not in prox_options:
            prox_options['tol'] = None

        prox_options['grad'] = self.grad
        prox_options['proj_P'] = self.tv.left.convex_conj.proximal(0)

        self.prox_options = prox_options
        self.alpha = alpha

        super().__init__(space=domain, linear=False, grad_lipschitz=0)

    def __call__(self, x):
        if self.alpha == 0:
            return 0
        else:
            out = self.alpha * self.tv(x) 
            return out

    @property
    def proximal(self):
        alpha = self.alpha
        prox_options = self.prox_options
        space = self.domain
        
        class ProximalTV(Operator):
            def __init__(self, sigma):
                self.sigma = float(sigma)
                self.prox_options = prox_options
                self.alpha = float(alpha)
                super(ProximalTV, self).__init__(
                    domain=space, range=space, linear=False)
    
            def _call(self, z, out):
                sigma = self.sigma * self.alpha
                if sigma == 0:
                    out.assign(z)
                else:
                    opts = self.prox_options
    
                    grad = opts['grad']
                    proj_P = opts['proj_P']
    
                    if opts['name'] == 'FGP':
                        if opts['warmstart']:
                            if opts['p'] is None:
                                opts['p'] = grad.range.zero()
    
                            p = opts['p']
                        else:
                            p = grad.range.zero()
    
                        niter = opts['niter']
                        out.assign(fgp_dual(p, z, sigma, niter, grad,
                                          proj_P, ))
    
                    else:
                        raise NotImplementedError('Not yet implemented')
                    
        return ProximalTV