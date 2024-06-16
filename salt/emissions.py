from functools import partial
from jax.scipy.optimize import minimize
import jax.numpy as np
from jax import vmap, lax, jit
from jax.tree_util import tree_map, register_pytree_node_class
from jax.flatten_util import ravel_pytree
from tensorflow_probability.substrates import jax as tfp

from ssm.hmm.emissions import Emissions
from ssm.hmm.posterior import StationaryHMMPosterior
import ssm.distributions as ssmd
tfd = tfp.distributions

EPS = 1e-2

@register_pytree_node_class
class SALTEmissions(Emissions):
    def __init__(self,
                 num_states: int,
                 mode: str,
                 single_subspace: bool,
                 l2_penalty: float,
                 temporal_penalty: float,
                 separate_diag: bool,
                 output_factors: np.ndarray,            # U: (K, N, D1)
                 input_factors: np.ndarray,             # V: (K, N, D2)
                 lag_factors: np.ndarray,               # W: (K, L, D3)
                 core_tensors: np.ndarray,              # G: (K, D1, D2, D3)
                 lowD_dynamics: np.ndarray,
                 biases: np.ndarray,                    # d: (K, N)
                 lowD_biases: np.ndarray,
                 diag: np.ndarray,                      # diag: (K, N, N)
                 covariance_matrix_sqrts: np.ndarray,   # Sigma: (K, N, N)
                 ):
        r"""Switching Autoregressive Low-Rank Tensor (SALT) Emissions"""
        super(SALTEmissions, self).__init__(num_states)
        self.mode = mode
        self.single_subspace = single_subspace
        self.l2_penalty = l2_penalty
        self.temporal_penalty = temporal_penalty

        # temporal l2 penalty matrix
        L, D3 = lag_factors.shape[1:]
        temporal_penalty_matrix = np.zeros((1,L*D3,L*D3))
        for l in range(L):
            penalty = l2_penalty * temporal_penalty**l
            idx = np.arange(-1-D3*(l+1),-1-D3*l)+1
            temporal_penalty_matrix = temporal_penalty_matrix.at[0,idx,idx].set(penalty)
        self.temporal_penalty_matrix = temporal_penalty_matrix
        
        self.separate_diag = separate_diag
        self.diag = diag
        
        self.output_factors = output_factors
        self.input_factors = input_factors
        self.lag_factors = lag_factors
        self.core_tensors = core_tensors
        self.lowD_dynamics = lowD_dynamics
        self.biases = biases
        self.lowD_biases = lowD_biases
        self.covariance_matrix_sqrts = covariance_matrix_sqrts
        
        # precompute tensors
        if single_subspace:
            self.tensors = np.einsum('kdef,ix,kje,klf,kxd->kijl',
                                      core_tensors,
                                      output_factors,
                                      input_factors,
                                      lag_factors,
                                      lowD_dynamics)

            self.tensors_for_lag_factors = np.einsum('kabc,ix,kjb,kxa->kijc',
                                                      core_tensors,
                                                      output_factors,
                                                      input_factors,
                                                      lowD_dynamics)
        else:
            self.tensors = np.einsum('kdef,kid,kje,klf->kijl',
                                     core_tensors,
                                     output_factors,
                                     input_factors,
                                     lag_factors)

            self.tensors_for_lag_factors = np.einsum('kabc,kia,kjb->kijc',
                                                     core_tensors,
                                                     output_factors,
                                                     input_factors)

    @property
    def emissions_dim(self):
        return self.output_factors.shape[0] if self.single_subspace else self.output_factors.shape[1]

    @property
    def emissions_shape(self):
        return (self.emissions_dim,)

    @property
    def num_lags(self):
        return self.lag_factors.shape[1]
    
    @property
    def core_tensor_dims(self):
        return self.core_tensors.shape[1:]

    def distribution(self, state: int, 
                     covariates=None, 
                     metadata=None, 
                     history: np.ndarray=None) -> tfd.MultivariateNormalTriL:
        """Returns the emissions distribution conditioned on a given state.

        Args:
            state (int): latent state
            covariates (np.ndarray, optional): optional covariates.
                Not yet supported. Defaults to None.

        Returns:
            emissions_distribution (tfd.MultivariateNormalTriL): the emissions distribution
        """
        # multiply history (L, N) with the tensor for given state to get shape (N,) prediction
        mean = np.einsum('ijl,lj->i',
                         self.tensors[state],
                         history)

        if self.single_subspace:
            mean += np.einsum('ia,a->i',
                              self.output_factors,
                              self.lowD_biases[state])
            mean += self.biases
        else:
            mean += self.biases[state]
            
        if self.separate_diag:
            diag_term = np.einsum('ab,b->a', self.diag[state], history[-1])
            mean += diag_term

        return tfd.MultivariateNormalTriL(mean, self.covariance_matrix_sqrts[state])
    
    def log_likelihoods(self, data, covariates=None, metadata=None):
        num_lags = self.num_lags
        num_states = self.num_states
        num_timesteps, emissions_dim = data.shape
        tensors = self.tensors.transpose([0, 1, 3, 2]) # K, N, N, L -> K, N, L, N
        scale_trils = self.covariance_matrix_sqrts
        biases = self.biases

        mean = lax.conv(data[:-1].reshape(1, 1, num_timesteps-1, emissions_dim), # 1, 1, T-1, N
                        tensors.reshape((num_states*emissions_dim, 1, num_lags, emissions_dim)), # K*N, 1, L, N 
                        window_strides=(1,1), 
                        padding='VALID') # 1, K*N, T-L, 1
        mean = mean[0,:,:,0].reshape((num_states,
                                      emissions_dim,
                                      num_timesteps - num_lags)) # K, N, T-L
        mean = mean.transpose([2,0,1]) # T-L, K, N

        if self.separate_diag:
            diag = self.diag
            diag_term = np.einsum('kab,tb->tka', diag, data[num_lags-1:-1])
            mean += diag_term

        if self.single_subspace:
            mean += np.einsum('ia,ka->ki',
                               self.output_factors,
                               self.lowD_biases)
            mean += biases[None]
        else:
            mean += biases
        
        log_probs = tfd.MultivariateNormalTriL(mean, scale_trils).log_prob(data[num_lags:, None, :])
        log_probs = np.vstack([np.zeros((num_lags, num_states)), log_probs])
        return log_probs
    
#     def update_core_tensors(self, dataset, Y, conv, Ez, Qinvs, mode):
#         if mode == 'cp':
#             return self.core_tensors
        
#         num_states = self.num_states
#         D1, D2, D3 = self.core_tensor_dims
#         G, U = self.core_tensors, self.output_factors

#         X = np.einsum('kje,pkftj->pktef', self.input_factors, conv) # (B, K, T-L, D2, D3)
#         X = X.reshape(X.shape[:-2] + (-1,)) # (B, K, T-L, D2*D3)
    
#         XTX = np.einsum('pkti,pktj,ptk->kij', X, X, Ez) # (K, D2*D3, D2*D3)
#         XTX_inv = np.linalg.inv(XTX)
        
#         UTQinvsU = np.einsum('kai,kab,kbj->kij', U, Qinvs, U) # (K, D1, D1)
#         UTQinvsU_inv = np.linalg.inv(UTQinvsU)
        
#         core_tensors = np.einsum('kab,pktb,ptk,pktc,kcd,kde,kef->kfa',
#                                  XTX_inv, X, Ez, Y, Qinvs, U, UTQinvsU_inv) # (K, D1, D2*D3)
        
#         core_tensors = core_tensors.reshape(num_states, D1, D2, D3)
#         return core_tensors

#     def update_core_tensors(self, dataset, Y, conv, Ez, Qinvs, mode):
#         if mode == 'cp':
#             return self.core_tensors
        
#         num_states = self.num_states
#         D1, D2, D3 = self.core_tensor_dims
#         G, U = self.core_tensors, self.output_factors

#         X = np.einsum('kje,pkftj->pktef', self.input_factors, conv) # (B, K, T-L, D2, D3)
#         X = X.reshape(X.shape[:-2] + (-1,)) # (B, K, T-L, D2*D3)
    
#         J = np.einsum('pkti,pktj,ptk->kij', X, X, Ez) # (K, D2*D3, D2*D3)
        
#         UTQinvsU = np.einsum('kai,kab,kbj->kij', U, Qinvs, U) # (K, D1, D1)
#         UTQinvsU_inv = np.linalg.inv(UTQinvsU)
        
#         h = np.einsum('pktb,ptk,pktc,kcd,kde,kef->kbf', X, Ez, Y, Qinvs, U, UTQinvsU_inv) # (K, D2*D3, D1)
        
#         core_tensors = np.linalg.solve(J, h) # (K, D2*D3, D1)
#         core_tensors = core_tensors.transpose([0,2,1]).reshape(num_states, D1, D2, D3)
#         return core_tensors

    def update_core_tensors(self, dataset, Y, conv, Ez, Qinvs, mode):
        if mode == 'cp':
            return self.core_tensors
        
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        G, U = self.core_tensors, self.output_factors
        
        if self.single_subspace:
            A = np.einsum('ia,kab->kib', U, self.lowD_dynamics)
        else:
            A = U
        
        def _get_Xhat_for_core_tensors(Ak, Xk):
            Xhat = np.kron(Ak[None,None], Xk[:,:,None]) # (B,T-L,N,D1*D2*D3)
            return Xhat

        X = np.einsum('kje,pkftj->kptef', self.input_factors, conv) # (K, B, T-L, D2, D3)
        X = X.reshape(X.shape[:-2] + (-1,)) # (K, B, T-L, D2*D3)
        Xhat = vmap(_get_Xhat_for_core_tensors)(A, X) # (K,B,T-L,N,D1*D2*D3)
    
        J = np.einsum('ptk,kptni,knm,kptmj->kij', Ez, Xhat, Qinvs, Xhat) # (K,D1*D2*D3,D1*D2*D3)
        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        
        h = np.einsum('ptk,kptni,knm,pktm->ki', Ez, Xhat, Qinvs, Y) # (K,D1*D2*D3)
        core_tensors = np.linalg.solve(J, h) # (K, D1*D2*D3)
        core_tensors = core_tensors.reshape(num_states, D1, D2, D3)
        
        return core_tensors
    
    def update_core_tensors_scan(self, dataset, Y, conv, Ez, Qinvs, mode):

        if mode == 'cp':
            return self.core_tensors
        
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        G, U = self.core_tensors, self.output_factors
        
        def _step(carry, xs):
            Jt, ht = carry
            Xt, Et, Yt = xs
            Xhat = vmap(lambda Uk, Xtk: np.kron(Uk[None], Xtk[:,None]))(U, Xt) # (K,B,N,D1*D2*D3)
            Jt += np.einsum('pk,kpni,knm,kpmj->kij', Et, Xhat, Qinvs, Xhat)
            ht += np.einsum('pk,kpni,knm,pkm->ki', Et, Xhat, Qinvs, Yt)
            return (Jt, ht), None

        X = np.einsum('kje,pkftj->tkpef', self.input_factors, conv) # (T-L, K, B, D2, D3)
        X = X.reshape(X.shape[:-2] + (-1,)) # (T-L, K, B, D2*D3)
        init_carry = (np.zeros((num_states, D1*D2*D3, D1*D2*D3)), np.zeros((num_states, D1*D2*D3)))
        Jh, _ = lax.scan(_step, init_carry, (X, Ez.transpose(1,0,2), Y.transpose(2,0,1,3)))
    
        J, h = Jh
        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        core_tensors = np.linalg.solve(J, h) # (K, D1*D2*D3)
        core_tensors = core_tensors.reshape(num_states, D1, D2, D3)
        
        return core_tensors
    
    def update_output_factors_and_biases(self, dataset, Y, conv, Ez):
        num_lags = self.num_lags
        xhat = np.einsum('kdef,kje,pkftj->pktd',
                        self.core_tensors,
                        self.input_factors,
                        conv) # (B, K, T-L, D1)
        xhat = np.pad(xhat, ((0,0),(0,0),(0,0),(1,0)), constant_values=1) # (B, K, T-L, 1+D1)

        J = np.einsum('ptk,pkti,pktj->kji', Ez, xhat, xhat) # (K, 1+D1, 1+D1)
        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        
        h = np.einsum('ptk,pktn,pkti->kin', Ez, dataset[:,:,num_lags:], xhat) # (K, 1+D1, N)
        output_factors_and_biases = np.linalg.solve(J, h) # (K, 1+D1, N)
        output_factors_and_biases = np.transpose(output_factors_and_biases, [0,2,1])
        return output_factors_and_biases[:,:,1:], output_factors_and_biases[:,:,0]
    
#     def update_output_factors_and_biases_vmap(self, dataset, Y, conv, Ez):
#         D1, D2, D3 = self.core_tensor_dims
#         num_lags = self.num_lags
        
#         def f(Gk, Vk, convk, Ezk, datasetk):
#             xhat = np.einsum('def,je,pftj->ptd', Gk, Vk, convk) # (B, T-L, D1)
#             xhat = np.pad(xhat, ((0,0),(0,0),(1,0)), constant_values=1) # (B, T-L, 1+D1)
#             Jk = np.einsum('pt,pti,ptj->ji', Ezk, xhat, xhat) # (1+D1, 1+D1)
#             hk = np.einsum('pt,ptn,pti->in', Ezk, datasetk, xhat) # (1+D1, N)
            
#             return Jk, hk

#         J, h = vmap(f, in_axes=(0,0,1,2,1))(self.core_tensors, self.input_factors, 
#                                             conv, Ez, dataset)
#         J += np.eye(J.shape[-1])[None]*self.l2_penalty
#         output_factors_and_biases = np.linalg.solve(J, h) # (K, 1+D1, N)
#         output_factors_and_biases = np.transpose(output_factors_and_biases, [0,2,1])
#         return output_factors_and_biases[:,:,1:], output_factors_and_biases[:,:,0]

    def update_output_factors_and_biases_vmap(self, dataset, Y, conv, Ez):
        D1, D2, D3 = self.core_tensor_dims
        num_lags = self.num_lags
        
        if self.single_subspace:
            def f(Bk, Gk, Vk, lowD_biasesk, convk, Ezk, datasetk):
                xhat = np.einsum('xd,def,je,pftj->ptd', Bk, Gk, Vk, convk) # (B, T-L, D1)
                xhat += lowD_biasesk[None]
                xhat = np.pad(xhat, ((0,0),(0,0),(1,0)), constant_values=1) # (B, T-L, 1+D1)
                Jk = np.einsum('pt,pti,ptj->ji', Ezk, xhat, xhat) # (1+D1, 1+D1)
                hk = np.einsum('pt,ptn,pti->in', Ezk, datasetk, xhat) # (1+D1, N)

                return Jk, hk

            J, h = vmap(f, in_axes=(0,0,0,0,1,2,1))(self.lowD_dynamics, self.core_tensors, 
                                                self.input_factors, self.lowD_biases, 
                                                conv, Ez, dataset)
            
            J, h = J.sum(0), h.sum(0)
            
            J += np.eye(J.shape[-1])*self.l2_penalty
            output_factors_and_biases = np.linalg.solve(J, h) # (1+D1, N)
            output_factors_and_biases = output_factors_and_biases.T
            return output_factors_and_biases[:,1:], output_factors_and_biases[:,0]
        else:
            def f(Gk, Vk, convk, Ezk, datasetk):
                xhat = np.einsum('def,je,pftj->ptd', Gk, Vk, convk) # (B, T-L, D1)
                xhat = np.pad(xhat, ((0,0),(0,0),(1,0)), constant_values=1) # (B, T-L, 1+D1)
                Jk = np.einsum('pt,pti,ptj->ji', Ezk, xhat, xhat) # (1+D1, 1+D1)
                hk = np.einsum('pt,ptn,pti->in', Ezk, datasetk, xhat) # (1+D1, N)

                return Jk, hk

            J, h = vmap(f, in_axes=(0,0,1,2,1))(self.core_tensors, self.input_factors, 
                                                conv, Ez, dataset)
            J += np.eye(J.shape[-1])[None]*self.l2_penalty
            output_factors_and_biases = np.linalg.solve(J, h) # (K, 1+D1, N)
            output_factors_and_biases = np.transpose(output_factors_and_biases, [0,2,1])
            return output_factors_and_biases[:,:,1:], output_factors_and_biases[:,:,0]
    
    def update_output_factors_and_biases_scan(self, dataset, Y, conv, Ez):

        num_lags = self.num_lags
        emissions_dim = self.emissions_dim
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        
        def _step(carry, xs):
            Jt, ht = carry
            convt, Et, datat = xs
            xhat = np.einsum('kdef,kje,pkfj->pkd',
                             self.core_tensors,
                             self.input_factors,
                             convt) # (B, K, D1)
            xhat = np.pad(xhat, ((0,0),(0,0),(1,0)), constant_values=1) # (B, K, 1+D1)
            
            Jt += np.einsum('pk,pki,pkj->kji', Et, xhat, xhat) # (K, 1+D1, 1+D1)
            ht += np.einsum('pk,pn,pki->kin', Et, datat, xhat) # (K, 1+D1, N)
            
            return (Jt, ht), None
        
        init_carry = (np.zeros((num_states, D1+1, D1+1)), np.zeros((num_states, D1+1, emissions_dim)))
        Jh, _ = lax.scan(_step, init_carry, (conv.transpose(3,0,1,2,4), 
                                             Ez.transpose(1,0,2), 
                                             dataset[:,num_lags:].transpose(1,0,2)))
        
        J, h = Jh
        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        output_factors_and_biases = np.linalg.solve(J, h) # (K, 1+D1, N)
        output_factors_and_biases = np.transpose(output_factors_and_biases, [0,2,1])
        return output_factors_and_biases[:,:,1:], output_factors_and_biases[:,:,0]
    
    def update_lowD_biases(self, dataset, Y, conv, Ez, Qinvs):

        num_batches = dataset.shape[0]
        U = self.output_factors
        
        def f(Bk, Gk, Vk, convk):
            xhat = np.einsum('ix,xd,def,je,ftj->ti', self.output_factors, Bk, Gk, Vk, convk)  # (T-L, N)

            return xhat

        _dataset = np.concatenate([(dataset[i] - vmap(f, in_axes=(0, 0, 0, 0))(self.lowD_dynamics,
                                                               self.core_tensors,
                                                               self.input_factors,
                                                               conv[i]))[None] for i in range(len(dataset))])

#         xhat = np.einsum('ix,kxd,kdef,kje,pkftj->pkti', self.output_factors, self.lowD_dynamics, 
#                          self.core_tensors, self.input_factors, conv)  # (B, K, T-L, N)

#         xhat = xhat.transpose(1, 0, 2, 3) # (B, K, T-L, N)

        # _dataset = dataset - xhat # (B, K, T-L, N)

        J = np.einsum('ptk,ia,kij,jb->kab', Ez, U, Qinvs, U)
        h = np.einsum('ptk,ia,kij,pktj->ka', Ez, U, Qinvs, _dataset)

        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        dynamics_biases = np.linalg.solve(J, h) # (K, D1)

        return dynamics_biases
    
    def update_lowD_dynamics(self, dataset, Y, conv, Ez, Qinvs):

        num_batches = dataset.shape[0]
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        G, U = self.core_tensors, self.output_factors

        def _f(Xk):
            Xhat = np.kron(U[None], Xk[:, None])  # (K,T-L,N,D1*D1)
            return Xhat

        X = np.einsum('kje,pkftj,kxef->pktx', self.input_factors, conv, G)  # (B, K, T-L, D1)
        Xhat = np.concatenate([lax.map(_f, X[i])[None] for i in range(num_batches)])  # (B, K, T-L, N, D1*D1)
        
        J = np.einsum('ptk,pktni,knm,pktmj->kij', Ez, Xhat, Qinvs, Xhat) # (K,D1*D1,D1*D1)
        J += np.eye(J.shape[-1])[None] * self.l2_penalty

        h = np.einsum('ptk,pktni,knm,pktm->ki', Ez, Xhat, Qinvs, Y) # (K,D1*D1)
        lowD_dynamics = np.linalg.solve(J, h)  # (K, D1*D1)
        lowD_dynamics = lowD_dynamics.reshape(num_states, D1, D1)

        return lowD_dynamics
    
    def update_input_factors(self, dataset, Y, conv, Ez, Qinvs):
        emissions_dim = self.emissions_dim
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims

        X = np.einsum('kia,kabc,pkctj->pktijb',
                      self.output_factors,
                      self.core_tensors,
                      conv) # (B, K, T, N, N, D2)
        X = X.reshape(X.shape[:-2]+(-1,))  # (B, K, T, N, N*D2)
        
        J = np.einsum('pktab,kac,ptk,pktcd->kbd', X, Qinvs, Ez, X) # (K, N*D2, N*D2)
        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        
        h = np.einsum('pktab,kac,ptk,pktc->kb', X, Qinvs, Ez, Y) # (K, N*D2)
        input_factors = np.linalg.solve(J, h) # (K, N*D2)
        input_factors = input_factors.reshape(num_states, emissions_dim, D2)

        return input_factors
    
    def update_input_factors_vmap(self, dataset, Y, conv, Ez, Qinvs):
        emissions_dim = self.emissions_dim
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims

        def f(Uk, Gk, convk, Qinvsk, Ezk, Yk):
            X = np.einsum('ia,abc,pctj->ptijb', Uk, Gk, convk) # (B, T-L, N, N, D2)
            X = X.reshape(X.shape[:-2]+(-1,))  # (B, T-L, N, N*D2)
            
            Jk = np.einsum('ptab,ac,pt,ptcd->bd', X, Qinvsk, Ezk, X) # (N*D2, N*D2)
            hk = np.einsum('ptab,ac,pt,ptc->b', X, Qinvsk, Ezk, Yk) # (N*D2)
        
            return Jk, hk

        J, h = vmap(f)(self.output_factors, self.core_tensors, 
                       conv.transpose(1,0,2,3,4), Qinvs, 
                       Ez.transpose(2,0,1), Y.transpose(1,0,2,3))
        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        input_factors = np.linalg.solve(J, h) # (K, N*D2)
        input_factors = input_factors.reshape(num_states, emissions_dim, D2)

        return input_factors
    
    def update_input_factors_scan(self, dataset, Y, conv, Ez, Qinvs):
        emissions_dim = self.emissions_dim
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        
        def _step(carry, xs):
            Jt, ht = carry
            convt, Et, Yt = xs
            
            X = np.einsum('kia,kabc,pkcj->pkijb',
                          self.output_factors,
                          self.core_tensors,
                          convt) # (B, K, N, N, D2)
            X = X.reshape(X.shape[:-2]+(-1,))  # (B, K, N, N*D2)
            
            Jt += np.einsum('pkab,kac,pk,pkcd->kbd', X, Qinvs, Et, X) # (K, N*D2, N*D2)
            ht += np.einsum('pkab,kac,pk,pkc->kb', X, Qinvs, Et, Yt) # (K, N*D2)
            
            return (Jt, ht), None

        init_carry = (np.zeros((num_states, emissions_dim*D2, emissions_dim*D2)), 
                      np.zeros((num_states, emissions_dim*D2)))
        Jh, _ = lax.scan(_step, init_carry, (conv.transpose(3,0,1,2,4), 
                                             Ez.transpose(1,0,2), 
                                             Y.transpose(2,0,1,3)))
        
        J, h = Jh
        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        input_factors = np.linalg.solve(J, h) # (K, N*D2)
        input_factors = input_factors.reshape(num_states, emissions_dim, D2)

        return input_factors
    
    def update_input_factors_map_vmap(self, dataset, Y, conv, Ez, Qinvs):
        emissions_dim = self.emissions_dim
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        
        def f(xs):
            Uk, Gk, convk, Qinvsk, Ezk, Yk = xs
            X = np.einsum('ia,abc,pctj->ptijb', Uk, Gk, convk) # (B, T-L, N, N, D2)
            X = X.reshape(X.shape[:-2]+(-1,))  # (B, T-L, N, N*D2)

            #s = vmap(lambda Xb: np.einsum('pta,ac,pt->ptc', Xb, Qinvsk, Ezk), in_axes=3, out_axes=3)(X)
            s = np.einsum('ptab,ac,pt->ptcb', X, Qinvsk, Ezk)

            Jk = np.einsum('ptcb,ptcd->bd', s, X) # (N*D2, N*D2)
            hk = np.einsum('ptcb,ptc->b', s, Yk) # (N*D2)
        
            return Jk, hk
        
        def f_single_subspace(xs):
            Bk, Gk, convk, Qinvsk, Ezk, Yk = xs
            X = np.einsum('ix,xa,abc,pctj->ptijb', self.output_factors, Bk, Gk, convk)  # (T-L, N, N, D2)
            X = X.reshape(X.shape[:-2] + (-1,))  # (T-L, N, N*D2)

            s = np.einsum('ptab,ac,pt->ptcb', X, Qinvsk, Ezk)

            Jk = np.einsum('ptcb,ptcd->bd', s, X)  # (N*D2, N*D2)
            hk = np.einsum('ptcb,ptc->b', s, Yk)  # (N*D2)

            return Jk, hk
        
        
        if self.single_subspace:
            J, h = lax.map(f_single_subspace, (self.lowD_dynamics, self.core_tensors, 
                               conv.transpose(1,0,2,3,4), Qinvs, 
                               Ez.transpose(2,0,1), Y.transpose(1,0,2,3)))
            J += np.eye(J.shape[-1])[None]*self.l2_penalty
            
        else:
            J, h = lax.map(f, (self.output_factors, self.core_tensors, 
                               conv.transpose(1,0,2,3,4), Qinvs, 
                               Ez.transpose(2,0,1), Y.transpose(1,0,2,3)))
            J += np.eye(J.shape[-1])[None]*self.l2_penalty
        
        


        
        def solve(Jk, hk):
            return np.linalg.solve(Jk, hk)
        input_factors = vmap(solve)(J, h) # (K, N*D2)
        input_factors = input_factors.reshape(num_states, emissions_dim, D2)

        return input_factors
    
    def update_lag_factors(self, dataset, Y, Ez, Qinvs):
        num_batches, num_timesteps, emissions_dim = dataset.shape
        num_lags = self.num_lags
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        
        def _get_Xhat_for_lag_factors(t):
            history = lax.dynamic_slice(dataset,
                                        (0,t-num_lags, 0),
                                        (num_batches,num_lags,emissions_dim))

            Xhat = np.einsum('kijc,plj->pkilc',
                             self.tensors_for_lag_factors,
                             history) # (B, K, N, L, D3)
            return Xhat.reshape((num_batches, num_states, emissions_dim, num_lags*D3)) # (B,K,N,L*D3)
        
        Xhat = vmap(_get_Xhat_for_lag_factors)(np.arange(self.num_lags, dataset.shape[1])) # (T-L, B, K, N, L*D3)
        J = np.einsum('ptk,tpknd,kno,tpkoe->kde', Ez, Xhat, Qinvs, Xhat) # (K, L*D3, L*D3)
        J += self.temporal_penalty_matrix
        
        h = np.einsum('ptk,tpknd,kno,pkto->kd', Ez, Xhat, Qinvs, Y) # (K, L*D3)
        lag_factors = np.linalg.solve(J, h) # (K, L*D3)
        
        Yhat = np.einsum('tpkij,kj->pkti', 
                         Xhat, lag_factors) # (B, K, T-L, N)
        Yhat += self.biases[None,:,None]
        lag_factors = lag_factors.reshape(num_states, num_lags, D3) # (K, L, D3)
        
        return lag_factors, Yhat

    def update_lag_factors_scan(self, dataset, Y, Ez, Qinvs):
        num_batches, num_timesteps, emissions_dim = dataset.shape
        num_lags = self.num_lags
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        
        def stack(t):
            return lax.dynamic_slice(dataset, 
                                     (0, t-num_lags, 0), 
                                     (num_batches, num_lags, emissions_dim))
        stacked = vmap(stack)(np.arange(num_lags, num_timesteps)) # (T-L, B, L, N)
        
        def f1(xs):
            k, Ek, Yk = xs

            Xhat = np.einsum('ijc,tplj->ptilc',
                             self.tensors_for_lag_factors[k],
                             stacked) # (B, T-L, N, L, D3)
            Xhat = Xhat.reshape((num_batches, num_timesteps-num_lags, emissions_dim, num_lags*D3)) # (B,T-L,N,L*D3)

            s = vmap(lambda Xhatd: np.einsum('pt,ptn,no->pto', Ek, Xhatd, Qinvs[k]), in_axes=3, out_axes=3)(Xhat)
            Jk = np.einsum('ptod,ptoe->de', s, Xhat) # (K, L*D3, L*D3)
            hk = np.einsum('ptod,pto->d', s, Yk) # (K, L*D3)

            return Jk, hk
            
        J, h = lax.map(f1, (np.arange(num_states), Ez.transpose(2,0,1), Y.transpose(1,0,2,3)))
        
        J += self.temporal_penalty_matrix
        def solve(Jk, hk):
            return np.linalg.solve(Jk, hk)
        lag_factors = vmap(solve)(J, h) # (K, L*D3)
        
        def f2(k):
            Xhat = np.einsum('ijc,tplj->ptilc',
                             self.tensors_for_lag_factors[k],
                             stacked) # (B, T-L, N, L, D3)
            Xhat = Xhat.reshape((num_batches, num_timesteps-num_lags, emissions_dim, num_lags*D3)) # (B,T-L,N,L*D3)
            Yhat = np.einsum('ptij,j->pti', 
                             Xhat, lag_factors[k]) # (B, T-L, N)
            return Yhat

        Yhat = lax.map(f2, np.arange(num_states))
        Yhat = Yhat.transpose(1,0,2,3) # (B, K, T-L, N)
        if self.single_subspace:
            Yhat += np.einsum('ia,ka->ki',
                               self.output_factors,
                               self.lowD_biases)[None, :, None] + self.biases[None,None,None]
        else:
            Yhat += self.biases[None,:,None]
        lag_factors = lag_factors.reshape(num_states, num_lags, D3) # (K, L, D3)

        if self.separate_diag:
            diag_term = np.einsum('kb,ptb->pktb', diag, dataset[:,num_lags-1:-1])
            Yhat += diag_term
        
        return lag_factors, Yhat
    
    def update_diag(self, dataset, Yhat, Ez, Qinvs):
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        num_lags = self.num_lags
        
        residual = dataset[:, None, num_lags:] - Yhat # (B, K, T-L, N)
        
        X = dataset[:, num_lags-1:-1] # B, T-L, N
    
        J = np.einsum('ptk,ptn,knm,ptm->knm', Ez, X, Qinvs, X) # (K,N,N)
        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        
        h = np.einsum('ptk,ptn,knm,pktm->kn', Ez, X, Qinvs, residual) # (K,N)
        diag = np.linalg.solve(J, h) # (K, N)
        
        diag_term = np.einsum('kb,ptb->pktb', diag, dataset[:,num_lags-1:-1])
        Yhat += diag_term

        diag = vmap(lambda diagk: np.diag(diagk))(diag) # (K, N, N)
        
        return diag, Yhat
    
    def update_covariance_matrix_sqrts(self, dataset, Yhat, Ez):
        num_batches, num_timesteps, emissions_dim = dataset.shape
        num_lags = self.num_lags
        num_states = self.num_states
        
        Y = dataset[:,num_lags:].reshape(-1, emissions_dim)
        Yhat_reshaped = np.transpose(Yhat, [1,0,2,3]).reshape(num_states, -1, emissions_dim)
        Ez_reshaped = Ez.reshape(-1, num_states).T
        
        covariance_matrices = vmap(lambda yhatk, Ezk: np.cov(Y - yhatk, 
                                                             rowvar=False, 
                                                             bias=True, 
                                                             aweights=Ezk))(Yhat_reshaped, Ez_reshaped)
        
        covariance_matrices += np.eye(emissions_dim)[None]*EPS
        covariance_matrix_sqrts = np.linalg.cholesky(covariance_matrices)
        
        return covariance_matrix_sqrts

    def convolve_dataset_with_lag_factors(self, dataset):
        num_batches, num_timesteps, emissions_dim = dataset.shape
        num_lags = self.num_lags
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        
        lag_factors = np.transpose(self.lag_factors,[0,2,1]) # K, L, D3 -> K, D3, L
        lag_factors = lag_factors.reshape((num_states*D3, 1, num_lags, 1)) # K, D3, L -> K*D3, L

        conv_output = lax.conv(dataset[:,None,:-1], # B, 1, T-1, N
                               lag_factors, # K*D3, 1, L, 1 
                               window_strides=(1,1), 
                               padding='VALID') # B, K*D3, T-L, N
        conv_output = conv_output.reshape((num_batches,
                                           num_states,
                                           D3,
                                           num_timesteps - num_lags,
                                           emissions_dim)) # B, K, D3, T-L, N
        return conv_output
    
    def m_step(self,
               dataset: np.ndarray,
               posterior, #: StationaryHMMPosterior,
               covariates=None,
               metadata=None):
        r"""Update the distribution with an M step.
        Operates over a batch of data.
        Args:
            dataset (np.ndarray): observed data
                of shape :math:`(\text{batch\_dim}, \text{num\_timesteps}, \text{emissions\_dim})`.
            posteriors (StationaryHMMPosterior): HMM posterior object
                with batch_dim to match dataset.
                
        Returns:
            emissions (SALTEmissions): updated emissions object
        """
        
        num_batches, num_timesteps, emissions_dim = dataset.shape
        num_lags = self.num_lags
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        mode = self.mode
        
        Ez = posterior['expected_states'][:,num_lags:] # B, T-L, K
        Qs = np.einsum('kab,kcb->kac', 
                self.covariance_matrix_sqrts, 
                self.covariance_matrix_sqrts)
        Qinvs = np.linalg.inv(Qs) # K, N, N
        conv = self.convolve_dataset_with_lag_factors(dataset) # B, K, D3, T-L, N
        
        if self.separate_diag:
            diag_term = np.einsum('kab,ptb->pkta', self.diag, dataset[:,num_lags-1:-1]) # B, K, T-L, N
            R = dataset[:,None,num_lags:] - diag_term # B, K, T-L, N
        else:
            R = np.tile(dataset[:,None,num_lags:], (1, num_states, 1, 1)) # B, K, T-L, N
        
        # update output factors and biases
        #self.output_factors, self.biases = self.update_output_factors_and_biases(dataset, None, conv, Ez)
        #self.output_factors, self.biases = self.update_output_factors_and_biases_scan(dataset, None, conv, Ez)
        self.output_factors, self.biases = self.update_output_factors_and_biases_vmap(R, None, conv, Ez)
        
        updated_biases = self.biases[None, None, None] if self.single_subspace else self.biases[None, :,None]
        Y = R - updated_biases # B, K, T-L, N
        
        if self.single_subspace:
            # update lowD biases
            self.lowD_biases = self.update_lowD_biases(Y, None, conv, Ez, Qinvs)
            Y = Y - np.einsum('ia,ka->ki', self.output_factors, self.lowD_biases)[None, :, None]

            # update lowD dynamics
            self.lowD_dynamics = self.update_lowD_dynamics(dataset, Y, conv, Ez, Qinvs)
  
        # update core tensors
        self.core_tensors = self.update_core_tensors(dataset, Y, conv, Ez, Qinvs, mode)
        #self.core_tensors = self.update_core_tensors_scan(dataset, Y, conv, Ez, Qinvs, mode)

        # update input factors
        #self.input_factors = self.update_input_factors(dataset, Y, conv, Ez, Qinvs)
        #self.input_factors = self.update_input_factors_scan(dataset, Y, conv, Ez, Qinvs)
        #self.input_factors = self.update_input_factors_vmap(dataset, Y, conv, Ez, Qinvs)
        self.input_factors = self.update_input_factors_map_vmap(dataset, Y, conv, Ez, Qinvs)
        
        einsum_string1 = 'kabc,ix,kjb,kxa->kijc' if self.single_subspace else 'kabc,kix,kjb,kxa->kijc'
        self.tensors_for_lag_factors = np.einsum(einsum_string1,
                                                 self.core_tensors,
                                                 self.output_factors,
                                                 self.input_factors,
                                                  self.lowD_dynamics)
        
        # update lag factors
        #self.lag_factors, Yhat = self.update_lag_factors(dataset, Y, Ez, Qinvs)
        self.lag_factors, Yhat = self.update_lag_factors_scan(dataset, Y, Ez, Qinvs)
        #self.lag_factors, Yhat = self.update_lag_factors_map_vmap(dataset, Y, Ez, Qinvs)
        
        if self.separate_diag:
            self.diag, Yhat = self.update_diag(dataset, Yhat, Ez, Qinvs)
        
        # update covariance_matrix_sqrts
        self.covariance_matrix_sqrts = self.update_covariance_matrix_sqrts(dataset, Yhat, Ez)
        
        einsum_string2 = 'kdef,ix,kje,klf,kxd->kijl' if self.single_subspace else 'kdef,kix,kje,klf,kxd->kijl'
        self.tensors = np.einsum(einsum_string2,
                                  self.core_tensors,
                                  self.output_factors,
                                  self.input_factors,
                                  self.lag_factors,
                                  self.lowD_dynamics)

        # return updated self
        return self

    def tree_flatten(self):
        children = (self.output_factors,
                    self.input_factors,
                    self.lag_factors,
                    self.core_tensors,
                    self.lowD_dynamics,
                    self.biases,
                    self.lowD_biases,
                    self.diag,
                    self.covariance_matrix_sqrts,
                    )
        aux_data = (self.num_states,
                    self.mode,
                    self.single_subspace,
                    self.l2_penalty,
                    self.temporal_penalty,
                    self.separate_diag,
                    )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)
