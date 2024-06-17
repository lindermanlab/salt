"""
Model classes for SALT.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax, jit
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from jax.tree_util import register_pytree_node_class

from ssm.arhmm.base import AutoregressiveHMM
from ssm.hmm.initial import StandardInitialCondition
from ssm.hmm.transitions import StationaryTransitions, StickyTransitions
from ssm.arhmm.emissions import AutoregressiveEmissions

from salt.emissions import SALTEmissions

import numpy as np

supported_modes = ['cp', 'tucker']

@register_pytree_node_class
class SALT(AutoregressiveHMM):
    def __init__(self,
                 num_states: int,
                 num_emission_dims: int=None,
                 num_lags: int=None,
                 core_tensor_dims: tuple=(1, 1, 1), # output, input, lag
                 initial_state_probs: jnp.ndarray=None,
                 transition_matrix: jnp.ndarray=None,
                 emission_output_factors: jnp.ndarray=None,
                 emission_input_factors: jnp.ndarray=None,
                 emission_lag_factors: jnp.ndarray=None,
                 emission_core_tensors: jnp.ndarray=None,
                 emission_lowD_dynamics: jnp.ndarray=None,
                 emission_biases: jnp.ndarray=None,
                 emission_covariance_matrix_sqrts: jnp.ndarray=None,
                 diag_cov: bool=False,
                 lowD_biases: jnp.ndarray=None,
                 seed: jr.PRNGKey=None,
                 mode: str='cp',
                 single_subspace: bool=False,
                 init_data: jnp.ndarray=None,
                 l2_penalty: float=1e-4,
                 temporal_penalty: float=1.0, # should be >=1
                 sticky_params: tuple=None,
                 dtype=jnp.float64):
        r"""Switching Autoregressive Low-rank Tensor Model (SALT).
        ############
        TODO
        ############ 
        """
        
        assert temporal_penalty >= 1.0 and l2_penalty >= 0, "Invalid penalty"
        
        mode = mode.lower()
        if mode not in supported_modes:
            raise ValueError(
                f"'mode' should be from {supported_modes}"
            )
        if mode == "cp":
            if not (core_tensor_dims[0] == core_tensor_dims[1] == core_tensor_dims[2]):
                raise ValueError(
                    f"'core_tensor_dims' should have same dimensions for mode {mode}"
                )

        if initial_state_probs is None:
            initial_state_probs = jnp.ones(num_states).astype(dtype) / num_states

        if transition_matrix is None:
            if sticky_params is None:
                transition_matrix = jnp.ones((num_states, num_states)).astype(dtype) / num_states
            else:
                alpha, kappa = sticky_params
                transition_matrix = kappa * jnp.eye(num_states) + alpha * jnp.ones((num_states, num_states))
                transition_matrix /= (kappa + alpha * num_states)
            
        if init_data is None:
            if emission_output_factors is None:
                this_seed, seed = jr.split(seed, 2)
                if single_subspace:
                    emission_output_factors_shape = (num_emission_dims, core_tensor_dims[0])
                else:
                    emission_output_factors_shape = (num_states, num_emission_dims, core_tensor_dims[0])
                emission_output_factors = tfd.Normal(0, 1).sample(
                    seed=this_seed,
                    sample_shape=emission_output_factors_shape).astype(dtype)
                # emission_output_factors /= jnp.linalg.norm(emission_output_factors, ord=2, axis=-2, keepdims=True)

            if emission_input_factors is None:
                this_seed, seed = jr.split(seed, 2)
                emission_input_factors = tfd.Normal(0, 1).sample(
                    seed=this_seed,
                    sample_shape=(num_states, num_emission_dims, core_tensor_dims[1])).astype(dtype)
                # emission_input_factors /= jnp.linalg.norm(emission_input_factors, ord=2, axis=-2, keepdims=True)

            if emission_lag_factors is None:
                this_seed, seed = jr.split(seed, 2)
                emission_lag_factors = tfd.Normal(0, 1).sample(
                    seed=this_seed,
                    sample_shape=(num_states, num_lags, core_tensor_dims[2])).astype(dtype)

            if emission_core_tensors is None:
                if mode == 'tucker':
                    this_seed, seed = jr.split(seed, 2)
                    emission_core_tensors = tfd.Normal(0, 1).sample(
                        seed=this_seed,
                        sample_shape=(num_states,) + core_tensor_dims).astype(dtype)
                    
                    # stabilize weights
                    intermediate_weights = jnp.einsum('...def,...id,...je->...ijf', 
                                                      emission_core_tensors,
                                                      emission_output_factors, 
                                                      emission_input_factors)
        
                    for k in range(num_states):
                        for f in range(core_tensor_dims[2]):
                            eigdecomp_result = np.linalg.eig(intermediate_weights[k,:,:,f])
                            max_eigval = jnp.max(jnp.abs(eigdecomp_result.eigenvalues))
                            emission_core_tensors = emission_core_tensors.at[k,:,:,f].set(emission_core_tensors[k,:,:,f] / max_eigval)
                
                elif mode == 'cp':
                    idx = jnp.arange(core_tensor_dims[1])
                    emission_core_tensors = jnp.zeros((num_states,) + core_tensor_dims).astype(dtype)
                    emission_core_tensors = emission_core_tensors.at[:,idx,idx,idx].set(1)

            if emission_lowD_dynamics is None:
                if single_subspace:
                    this_seed, seed = jr.split(seed, 2)
                    emission_lowD_dynamics = tfd.Normal(0, 1).sample(seed=this_seed,
                        sample_shape=(num_states, core_tensor_dims[0], core_tensor_dims[0])).astype(dtype)
                else:
                    emission_lowD_dynamics = jnp.tile(jnp.eye(core_tensor_dims[0])[None],
                                                      (num_states, 1, 1)).astype(dtype)

            if emission_biases is None:
                if single_subspace:
                    emission_biases_shape = (num_emission_dims,)
                else:
                    emission_biases_shape = (num_states, num_emission_dims)

                this_seed, seed = jr.split(seed, 2)
                emission_biases = tfd.Normal(0, 1).sample(
                    seed=this_seed,
                    sample_shape=emission_biases_shape).astype(dtype)

            if lowD_biases is None:
                if single_subspace:
                    this_seed, seed = jr.split(seed, 2)
                    lowD_biases = tfd.Normal(0, 1).sample(
                        seed=this_seed,
                        sample_shape=(num_states, core_tensor_dims[0])).astype(dtype)
                else:
                    lowD_biases = jnp.zeros((num_states, core_tensor_dims[0])).astype(dtype)

            if emission_covariance_matrix_sqrts is None:
                emission_covariance_matrix_sqrts = jnp.tile(jnp.eye(num_emission_dims), (num_states, 1, 1)).astype(dtype)
                
        else:
            # needs update (single subspace not supported)
            params_initialized_with_data = self._initialize_with_data(init_data, 
                                                                     num_states, 
                                                                     num_emission_dims, 
                                                                     num_lags, 
                                                                     core_tensor_dims, 
                                                                     constrained,
                                                                     alpha,
                                                                     seed)
            emission_output_factors = params_initialized_with_data[0]
            emission_input_factors = params_initialized_with_data[1]
            emission_lag_factors = params_initialized_with_data[2]
            emission_core_tensors = params_initialized_with_data[3]
            emission_biases = params_initialized_with_data[4]
            emission_covariance_matrix_sqrts = params_initialized_with_data[5]

        initial_condition = StandardInitialCondition(num_states, initial_probs=initial_state_probs)
        if sticky_params is None:
            transitions = StationaryTransitions(num_states, transition_matrix=transition_matrix)
        else:
           transitions = StickyTransitions(num_states, alpha=alpha, kappa=kappa, transition_matrix=transition_matrix)
        emissions = SALTEmissions(num_states,
                                  mode,
                                  single_subspace=single_subspace,
                                  diag_cov=diag_cov,
                                  l2_penalty=l2_penalty,
                                  temporal_penalty=temporal_penalty,
                                  output_factors=emission_output_factors,
                                  input_factors=emission_input_factors,
                                  lag_factors=emission_lag_factors,
                                  core_tensors=emission_core_tensors,
                                  lowD_dynamics=emission_lowD_dynamics,
                                  biases=emission_biases,
                                  lowD_biases=lowD_biases,
                                  covariance_matrix_sqrts=emission_covariance_matrix_sqrts)
        super(SALT, self).__init__(num_states,
                                   initial_condition,
                                   transitions,
                                   emissions)
        
    def _initialize_with_data(self,
                             data, 
                             num_states,
                             num_emission_dims, 
                             num_lags, 
                             core_tensor_dims, 
                             constrained,
                             alpha,
                             seed):
        
        D1, D2, D3 = core_tensor_dims
        
        emission_output_factors = jnp.zeros((num_states, num_emission_dims, core_tensor_dims[0])).astype(jnp.float64)
        emission_input_factors = jnp.zeros((num_states, num_emission_dims, core_tensor_dims[1])).astype(jnp.float64)
        emission_lag_factors = jnp.zeros((num_states, num_lags, core_tensor_dims[2])).astype(jnp.float64)
        emission_core_tensors = jnp.zeros((num_states,) + core_tensor_dims).astype(jnp.float64)
        emission_biases = jnp.zeros((num_states, num_emission_dims)).astype(jnp.float64)
        emission_covariance_matrix_sqrts = jnp.zeros((num_states, num_emission_dims, num_emission_dims)).astype(jnp.float64)
        
        # run K-means
        from sklearn.cluster import KMeans
        from sklearn.linear_model import LinearRegression, Ridge
        km = KMeans(num_states)
        assignments = km.fit_predict(data).reshape(data.shape[:-1])[num_lags:]
        
        def _get_X(t):
            history = lax.dynamic_slice(data,
                                        (t, 0),
                                        (num_lags, num_emission_dims))
            return history
        
        # tensor regression -> tensors + biases
        for k in range(num_states):
            kidx = np.where(assignments == k)[0]
            X = vmap(_get_X)(kidx) # (T_k, L, N)
            X_reshaped = X.reshape(kidx.shape[0], -1) # (T_k, L*N)
            Y = data[kidx+num_lags] # (T_k, N)
            if alpha > 0:
                res = Ridge(alpha).fit(X_reshaped, Y)
            else:
                res = LinearRegression().fit(X_reshaped, Y)
                                                    
            weight = res.coef_.reshape(num_emission_dims, num_lags, num_emission_dims) # N, L, N
            weight = jnp.transpose(weight, [0,2,1]) # N, N, L
            
            this_seed, seed = jr.split(seed, 2)
            weight_noise = tfd.Normal(0, 1).sample(
                                    seed=this_seed,
                                    sample_shape=weight.shape).astype(jnp.float64) / jnp.sqrt(kidx.shape[0])
            
            weight += weight_noise

            if constrained:
                # CP
                from tensorly.decomposition import parafac
                _, factors = parafac(weight, core_tensor_dims[0])

                core = jnp.eye(D1)
                emission_core_tensors = emission_core_tensors.at[k,:,jnp.arange(D2),jnp.arange(D3)].set(core)
                
            else:
                # Tucker
                from tensorly.decomposition import tucker
                core, factors = tucker(weight, core_tensor_dims)
                
                emission_core_tensors = emission_core_tensors.at[k].set(core)

            emission_output_factors = emission_output_factors.at[k].set(factors[0])
            emission_input_factors = emission_input_factors.at[k].set(factors[1])
            emission_lag_factors = emission_lag_factors.at[k].set(factors[2])
            
            this_seed, seed = jr.split(seed, 2)
            bias_noise = tfd.Normal(0, 1).sample(
                                    seed=this_seed,
                                    sample_shape=res.intercept_.shape).astype(jnp.float64) / jnp.sqrt(kidx.shape[0])
            emission_biases = emission_biases.at[k].set(res.intercept_+bias_noise)
            
            yhat = jnp.einsum('abc,ia,jb,kc,tkj->ti',
                             emission_core_tensors[k],
                             emission_output_factors[k],
                             emission_input_factors[k],
                             emission_lag_factors[k],
                             X)
            yhat += emission_biases[k]
            
            covariance_matrix = jnp.cov(Y - yhat, rowvar=False, bias=True)
            covariance_matrix_sqrts = jnp.linalg.cholesky(covariance_matrix)
            emission_covariance_matrix_sqrts = emission_covariance_matrix_sqrts.at[k].set(covariance_matrix_sqrts)
            
        return (emission_output_factors, 
                emission_input_factors, 
                emission_lag_factors, 
                emission_core_tensors, 
                emission_biases, 
                emission_covariance_matrix_sqrts)

    @property
    def num_lags(self):
        return self._emissions.num_lags

    def tree_flatten(self):
        children = (self._initial_condition,
                    self._transitions,
                    self._emissions)
        aux_data = self._num_states
        return children, aux_data

    # directly SALT using parent (HMM) constructor
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        super(cls, obj).__init__(aux_data, *children)
        return obj
