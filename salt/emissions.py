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

EPS = 1e-4

@register_pytree_node_class
class SALTEmissions(Emissions):
    def __init__(self,
                 num_states: int,
                 mode: str,
                 single_subspace: bool,
                 diag_cov: bool,
                 l2_penalty: float,
                 temporal_penalty: float,
                 output_factors: np.ndarray,            # U: (K, N, D1)
                 input_factors: np.ndarray,             # V: (K, N, D2)
                 lag_factors: np.ndarray,               # W: (K, L, D3)
                 core_tensors: np.ndarray,              # G: (K, D1, D2, D3)
                 lowD_dynamics: np.ndarray,
                 biases: np.ndarray,                    # d: (K, N)
                 lowD_biases: np.ndarray,
                 covariance_matrix_sqrts: np.ndarray,   # Sigma: (K, N, N)
                 ):
        r"""Switching Autoregressive Low-Rank Tensor (SALT) Emissions"""
        super(SALTEmissions, self).__init__(num_states)
        self.mode = mode
        self.single_subspace = single_subspace
        self.diag_cov = diag_cov
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
        
        self.output_factors = output_factors
        self.input_factors = input_factors
        self.lag_factors = lag_factors
        self.core_tensors = core_tensors
        self.lowD_dynamics = lowD_dynamics
        self.biases = biases
        self.lowD_biases = lowD_biases
        self.covariance_matrix_sqrts = covariance_matrix_sqrts
        
        # precompute tensors
        self.tensors = np.einsum('...def,...ix,...je,...lf,...xd->...ijl',
                                  core_tensors,
                                  output_factors,
                                  input_factors,
                                  lag_factors,
                                  lowD_dynamics)
        self.tensors_for_lag_factors = np.einsum('...abc,...ix,...jb,...xa->...ijc',
                                                  core_tensors,
                                                  output_factors,
                                                  input_factors,
                                                  lowD_dynamics)

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

    def update_core_tensors(self, dataset, Y, conv, Ez, Qinvs, mode):
        if mode == 'cp':
            return self.core_tensors
        
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        A = np.einsum('...ia,...ab->...ib', self.output_factors, self.lowD_dynamics)
        
        def _get_Xhat_for_core_tensors(Ak, Xk):
            Xhat = np.kron(Ak[None,None], Xk[:,:,None]) # (B,T-L,N,D1*D2*D3)
            return Xhat

        X = np.einsum('kje,pkftj->kptef', self.input_factors, conv) # (K, B, T-L, D2, D3)
        X = X.reshape(X.shape[:-2] + (-1,)) # (K, B, T-L, D2*D3)
        Xhat = vmap(_get_Xhat_for_core_tensors)(A, X) # (K, B, T-L, N, D1*D2*D3)
    
        J = np.einsum('ptk,kptni,knm,kptmj->kij', Ez, Xhat, Qinvs, Xhat) # (K,D1*D2*D3,D1*D2*D3)
        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        
        h = np.einsum('ptk,kptni,knm,pktm->ki', Ez, Xhat, Qinvs, Y) # (K,D1*D2*D3)
        core_tensors = np.linalg.solve(J, h[..., None])[..., 0] # (K, D1*D2*D3)
        core_tensors = core_tensors.reshape(num_states, D1, D2, D3)
        
        return core_tensors
    
    def update_core_tensors_scan(self, dataset, Y, conv, Ez, Qinvs, mode):

        if mode == 'cp':
            return self.core_tensors
        
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        U, V = self.output_factors, self.input_factors
        A = np.einsum('...ia,...ab->...ib', U, self.lowD_dynamics)

        conv = conv.transpose(3,0,1,2,4)
        Ez = Ez.transpose(1,0,2)
        Y = Y.transpose(2,0,1,3)
        
        def _step(carry, xs):
            Jt, ht = carry
            convt, Et, Yt = xs
            Xt = np.einsum('kje,pkfj->kpef', V, conv)
            Xt = Xt.reshape(Xt.shape[:-2] + (-1,))
            Xhat = vmap(lambda Ak, Xtk: np.kron(Ak[None], Xtk[:,None]))(A, Xt) # (K,B,N,D1*D2*D3)
            Jt += np.einsum('pk,kpni,knm,kpmj->kij', Et, Xhat, Qinvs, Xhat)
            ht += np.einsum('pk,kpni,knm,pkm->ki', Et, Xhat, Qinvs, Yt)
            return (Jt, ht), None

        init_carry = (np.zeros((num_states, D1*D2*D3, D1*D2*D3)), 
                      np.zeros((num_states, D1*D2*D3)))
        Jh, _ = lax.scan(_step, init_carry, (conv, Ez, Y))
    
        J, h = Jh
        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        
        core_tensors = np.linalg.solve(J, h[..., None])[..., 0] # (K, D1*D2*D3)
        core_tensors = core_tensors.reshape(num_states, D1, D2, D3)
        
        return core_tensors

    def update_output_factors_and_biases_vmap(self, dataset, Y, conv, Ez, Qinvs):
        N = self.emissions_dim
        K = self.num_states
        D1, D2, D3 = self.core_tensor_dims
        num_lags = self.num_lags
        
        def f(Bk, Gk, Vk, lowD_biasesk, Qinvsk, convk, Ezk, datasetk):
            if self.single_subspace:
                xhat = np.einsum('xd,def,je,pftj->ptx', Bk, Gk, Vk, convk) # (B, T-L, D1)
                xhat += lowD_biasesk[None, None]
            else:
                xhat = np.einsum('def,je,pftj->ptd', Gk, Vk, convk) # (B, T-L, D1)
            xhat = np.pad(xhat, ((0,0),(0,0),(1,0)), constant_values=1) # (B, T-L, 1+D1)
            Jk = np.einsum('pt,pti,ab,ptj->aibj', Ezk, xhat, Qinvsk, xhat).reshape(N*(1+D1), N*(1+D1)) # (Nx(1+D1), Nx(1+D1))
            hk = np.einsum('pt,ptn,nb,pti->bi', Ezk, datasetk, Qinvsk, xhat).reshape(-1) # Nx(1+D1)

            return Jk, hk

        J, h = vmap(f, in_axes=(0,0,0,0,0,1,2,1))(self.lowD_dynamics, self.core_tensors, 
                                            self.input_factors, self.lowD_biases, Qinvs,
                                            conv, Ez, dataset)

        if self.single_subspace:
            J, h = J.sum(0), h.sum(0)
            J += np.eye(J.shape[-1])*self.l2_penalty
        else:
            J += np.eye(J.shape[-1])[None]*self.l2_penalty

        output_factors_and_biases = np.linalg.solve(J, h[..., None])[..., 0]

        if self.single_subspace:
            output_factors_and_biases = output_factors_and_biases.reshape(N, D1+1)
            return output_factors_and_biases[:,1:], output_factors_and_biases[:,0]
        else:
            output_factors_and_biases = output_factors_and_biases.reshape(K, N, D1+1)
            return output_factors_and_biases[:,:,1:], output_factors_and_biases[:,:,0]
    
    def update_lowD_biases(self, dataset, Y, conv, Ez, Qinvs):

        num_batches = dataset.shape[0]
        U = self.output_factors
        
        def f(Bk, Gk, Vk, convk):
            xhat = np.einsum('ix,xd,def,je,ftj->ti', 
                             U, Bk, Gk, Vk, convk)  # (T-L, N)
            return xhat

        _dataset = np.concatenate([(dataset[i] - vmap(f, in_axes=(0, 0, 0, 0))(self.lowD_dynamics,
                                                               self.core_tensors,
                                                               self.input_factors,
                                                               conv[i]))[None] for i in range(len(dataset))])

        J = np.einsum('ptk,ia,kij,jb->kab', Ez, U, Qinvs, U)
        h = np.einsum('ptk,ia,kij,pktj->ka', Ez, U, Qinvs, _dataset)

        J += np.eye(J.shape[-1])[None]*self.l2_penalty
        dynamics_biases = np.linalg.solve(J, h[..., None])[..., 0] # (K, D1)

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
        lowD_dynamics = np.linalg.solve(J, h[..., None])[..., 0]  # (K, D1*D1)
        lowD_dynamics = lowD_dynamics.reshape(num_states, D1, D1)

        return lowD_dynamics
    
    def update_input_factors_map_vmap(self, dataset, Y, conv, Ez, Qinvs):
        emissions_dim = self.emissions_dim
        num_states = self.num_states
        D1, D2, D3 = self.core_tensor_dims

        A = np.einsum('...ia,...ab->...ib', self.output_factors, self.lowD_dynamics)
        
        def f(xs):
            Ak, Gk, convk, Qinvsk, Ezk, Yk = xs
            X = np.einsum('ia,abc,pctj->ptijb', Ak, Gk, convk) # (B, T-L, N, N, D2)
            X = X.reshape(X.shape[:-2]+(-1,))  # (B, T-L, N, N*D2)

            #s = vmap(lambda Xb: np.einsum('pta,ac,pt->ptc', Xb, Qinvsk, Ezk), in_axes=3, out_axes=3)(X)
            s = np.einsum('ptab,ac,pt->ptcb', X, Qinvsk, Ezk)

            Jk = np.einsum('ptcb,ptcd->bd', s, X) # (N*D2, N*D2)
            hk = np.einsum('ptcb,ptc->b', s, Yk) # (N*D2)
        
            return Jk, hk

        J, h = lax.map(f, (A, self.core_tensors, 
                           conv.transpose(1,0,2,3,4), Qinvs, 
                           Ez.transpose(2,0,1), Y.transpose(1,0,2,3)))
        J += np.eye(J.shape[-1])[None]*self.l2_penalty

        def solve(Jk, hk):
            return np.linalg.solve(Jk, hk[..., None])[..., 0]
        input_factors = vmap(solve)(J, h) # (K, N*D2)
        input_factors = input_factors.reshape(num_states, emissions_dim, D2)

        return input_factors

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
            return np.linalg.solve(Jk, hk[..., None])[..., 0]
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
        
        return lag_factors, Yhat
    
    def update_covariance_matrix_sqrts(self, dataset, Yhat, Ez):
        num_batches, num_timesteps, emissions_dim = dataset.shape
        num_lags = self.num_lags
        num_states = self.num_states
        
        Y = dataset[:,num_lags:].reshape(-1, emissions_dim)
        Yhat_reshaped = np.transpose(Yhat, [1,0,2,3]).reshape(num_states, -1, emissions_dim)
        Ez_reshaped = Ez.reshape(-1, num_states).T
        
        covariance_matrices = vmap(lambda yhatk, Ezk: np.cov(Y - yhatk, 
                                                             rowvar=False, 
                                                             bias=False, 
                                                             aweights=Ezk))(Yhat_reshaped, Ez_reshaped)

        if self.diag_cov:
            covariance_matrices = vmap(lambda cov_mat: np.diag(np.diag(cov_mat)))(covariance_matrices)
        
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
               posterior,
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
        
        Ez = posterior['expected_states'][:, num_lags:] # B, T-L, K
        Qs = np.einsum('kab,kcb->kac', 
                self.covariance_matrix_sqrts, 
                self.covariance_matrix_sqrts)
        Qinvs = np.linalg.inv(Qs) # K, N, N
        conv = self.convolve_dataset_with_lag_factors(dataset) # B, K, D3, T-L, N
        
        R = np.tile(dataset[:, None, num_lags:], (1, num_states, 1, 1)) # B, K, T-L, N
        
        # update output factors and biases
        self.output_factors, self.biases = self.update_output_factors_and_biases_vmap(R, None, conv, Ez, Qinvs)
        
        updated_biases = self.biases[None, None, None] if self.single_subspace else self.biases[None, :, None]
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
        self.input_factors = self.update_input_factors_map_vmap(dataset, Y, conv, Ez, Qinvs)

        self.tensors_for_lag_factors = np.einsum('...abc,...ix,...jb,...xa->...ijc',
                                                 self.core_tensors,
                                                 self.output_factors,
                                                 self.input_factors,
                                                 self.lowD_dynamics)
        
        # update lag factors
        self.lag_factors, Yhat = self.update_lag_factors_scan(dataset, Y, Ez, Qinvs)
        
        # update covariance_matrix_sqrts
        self.covariance_matrix_sqrts = self.update_covariance_matrix_sqrts(dataset, Yhat, Ez)

        self.tensors = np.einsum('...def,...ix,...je,...lf,...xd->...ijl',
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
                    self.covariance_matrix_sqrts,
                    )
        aux_data = (self.num_states,
                    self.mode,
                    self.single_subspace,
                    self.diag_cov,
                    self.l2_penalty,
                    self.temporal_penalty,
                    )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)
