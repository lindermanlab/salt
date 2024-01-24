import wandb

import jax.numpy as np
import jax.random as jr
import scipy
from scipy.linalg import solve_discrete_are, inv
from ssm.utils import random_rotation
from ssm.lds import GaussianLDS

from jax.config import config
config.update("jax_enable_x64", True)
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.85'

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from salt.models import SALT

from ssm.plots import gradient_cmap

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

from ssm.arhmm import GaussianARHMM
from scipy.linalg import solve_discrete_are, inv

from sklearn.metrics import mean_squared_error as mse
from collections import defaultdict

sns.set_style("white")
sns.set_context("talk")

color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange",
    "brown",
    "pink"
    ]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

import logging
logger = logging.getLogger()

class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger.addFilter(CheckTypesFilter())

n_iters = 10000

def get_minimum_cp_rank(w):
    imag = np.imag(w)
    n_cmplx_pairs = np.where(imag != 0)[0].shape[0]//2
    n_reals = np.where(imag == 0)[0].shape[0]
    return n_reals + 3*n_cmplx_pairs

def reconstruct_tensor(output_factors, core_tensors, input_factors, lag_factors):

    reconstructed = np.einsum('kdef,kid,kje,klf->kijl',
                              core_tensors,
                              output_factors,
                              input_factors,
                              lag_factors)

    return reconstructed

def reshape_arhmm_weights(arhmm_weights):
    K, N, L = arhmm_weights.shape
    L = L // N
    reshaped = np.zeros((K,N,N,L))
    for l in range(L):
        reshaped = reshaped.at[:,:,:,l].set(arhmm_weights[:,:,N*l:N*(l+1)])
    return reshaped

def make_lds(K=1, D=7, N=20, L=50, seed=0):

    key = jr.PRNGKey(seed)
    k1, k2, k3, k4, k5, key = jr.split(key, 6)
    # Make parameters
    A1 = random_rotation(k1, D)
    A2 = random_rotation(k2, D)
    A3 = random_rotation(k3, D)
    A4 = random_rotation(k4, D)
    A = A1 @ A2 @ A3 @ A4
    Q = 1e-3 * np.eye(D)
    C = jr.normal(k4, (N, D))
    R = 1e0 * np.eye(N)

    Sigma_s = solve_discrete_are(a=A.T, b=C.T, r=R, q=Q)
    K_s = inv(inv(Sigma_s) + C.T @ R @ C) @ C.T @ inv(R)

    mats = np.zeros((L,N,N))
    for i in range(L-1,-1,-1):
        mat = C
        for j in range(i):
            mat = mat @ A @ (np.eye(D)-K_s @ C)
        mat = mat @ A @ K_s
        mats = mats.at[i].set(mat)
        
    w, v = scipy.linalg.eig(A@(np.eye(D)-K_s@C))
    min_cp_rank = get_minimum_cp_rank(w)
    min_tucker_rank = D
    print(min_tucker_rank, min_cp_rank)
    
    # Make LDS
    true_lds = GaussianLDS(D, N,
                      dynamics_weights=A,
                      dynamics_scale_tril=np.sqrt(Q),
                      emission_weights=C,
                      emission_scale_tril=np.sqrt(R))
    
    return true_lds, mats, min_tucker_rank, min_cp_rank, key

def train_models(true_lds, 
                 mats, 
                 min_tucker_rank, 
                 min_cp_rank, 
                 key, K=1, D=7, N=20, L=50):

    tol=1e-1
    batch_T = 500
    test_batches = 10
    train_batches = [2,4,10,20,60]
    ground_truth_tensor = np.flip(np.transpose(mats, [1,2,0]), axis=-1)
    testkT, key = jr.split(key, 2)
    xs_test, ys_test = true_lds.sample(testkT, batch_T, num_samples=test_batches)

    ground_truth_test_lps = []
    tucker_mses = defaultdict(list)
    tucker_test_lps = defaultdict(list)
    cp_mses = defaultdict(list)
    cp_test_lps = defaultdict(list)
    arhmm_mses = defaultdict(list)
    arhmm_test_lps = defaultdict(list)
    lds_mses = defaultdict(list)
    lds_test_lps = defaultdict(list)
    for n_batches in train_batches:
        kT, key = jr.split(key, 2)
        # Sample
        xs, ys = true_lds.sample(kT, batch_T, num_samples=n_batches)
        if n_batches == 1:
            ys = ys[None]
        true_lp = true_lds.marginal_likelihood(ys[:,L:]).sum()
        true_test_lp = true_lds.marginal_likelihood(ys_test[:,L:]).sum()
        print('true train lp:', true_lp)
        print('true test lp:', true_test_lp)
        ground_truth_test_lps.append(true_test_lp)

        # fit ARHMM
        kARHMM, key = jr.split(key, 2)
        arhmm = GaussianARHMM(K, N, L, seed=kARHMM)
        lps, arhmm, posterior = arhmm.fit(ys, num_iters=n_iters, method='em', tol=tol, initialization_method=None)

        arhmm_weights = arhmm.emission_weights
        arhmm_weights = reshape_arhmm_weights(arhmm_weights)

        arhmm_mse = mse(ground_truth_tensor.flatten(), arhmm_weights.flatten())
        arhmm_mses[n_batches].append(arhmm_mse)

        arhmm_test_lp = arhmm.marginal_likelihood(ys_test).sum()
        arhmm_test_lps[n_batches].append(arhmm_test_lp)

        print(arhmm_mses[n_batches][-1])
        print(arhmm_test_lps[n_batches][-1])

        # fit LDS
        kLDS, key = jr.split(key, 2)
        lds = GaussianLDS(D, N, emission_scale_tril=0.1**2 * np.eye(N), seed=kLDS)
        lps, lds, posterior = lds.fit(ys, num_iters=n_iters, method='em', tol=tol)

        fitted_A = lds._dynamics.weights
        fitted_Q = lds._dynamics.scale_tril @ lds._dynamics.scale_tril.T
        fitted_C = lds._emissions.weights
        fitted_R = lds._emissions.scale_tril @ lds._emissions.scale_tril.T

        Sigma_s = solve_discrete_are(a=fitted_A.T, b=fitted_C.T, r=fitted_R, q=fitted_Q)
        K_s = inv(inv(Sigma_s) + fitted_C.T @ fitted_R @ fitted_C) @ fitted_C.T @ inv(fitted_R)

        fitted_mats = np.zeros((L,N,N))
        for i in range(L-1,-1,-1):
            mat = fitted_C
            for j in range(i):
                mat = mat @ fitted_A @ (np.eye(D)-K_s @ fitted_C)
            mat = mat @ fitted_A @ K_s
            fitted_mats = mats.at[i].set(mat)

        fitted_lds_tensor = np.flip(np.transpose(fitted_mats, [1,2,0]), axis=-1)
        lds_mse = mse(ground_truth_tensor.flatten(), fitted_lds_tensor.flatten())
        lds_mses[n_batches].append(lds_mse)

        lds_test_lp = lds.marginal_likelihood(ys_test[:,L:]).sum()
        lds_test_lps[n_batches].append(lds_test_lp)

        print(lds_mses[n_batches][-1])
        print(lds_test_lps[n_batches][-1])


        for tucker_r in range(D-2,min_tucker_rank+3):

            if tucker_r != min_tucker_rank and n_batches != 60:
                continue
                
            print('tucker_r: ', tucker_r)

            kTucker, key = jr.split(key, 2)
            tucker_core_tensor_dims = (tucker_r,)*3
            print(f'tucker core tensor dimensions: {tucker_core_tensor_dims}')
            tucker_model = SALT(num_states=K,
                                num_emission_dims=N,
                                num_lags=L,
                                core_tensor_dims=tucker_core_tensor_dims,
                                mode='tucker',
                                l2_penalty=1e-4,
                                seed=kTucker)
            lps, tucker_model, posterior = tucker_model.fit(ys, num_iters=n_iters,
                                                            method='em', tol=tol)

            inferred_output_factors = tucker_model._emissions.output_factors
            inferred_core_tensors = tucker_model._emissions.core_tensors
            inferred_input_factors = tucker_model._emissions.input_factors
            inferred_lag_factors = tucker_model._emissions.lag_factors

            tucker_inferred_tensor = reconstruct_tensor(inferred_output_factors,
                                         inferred_core_tensors,
                                         inferred_input_factors,
                                         inferred_lag_factors)

            tucker_mse = mse(ground_truth_tensor.flatten(),
                             tucker_inferred_tensor[0].flatten())
            tucker_mses[n_batches].append(tucker_mse)

            tucker_test_lp = tucker_model.marginal_likelihood(ys_test).sum()
            tucker_test_lps[n_batches].append(tucker_test_lp)

            print(tucker_mses[n_batches][-1])
            print(tucker_test_lps[n_batches][-1])

        for cp_r in range(D-2,min_cp_rank+3):

            if cp_r != min_cp_rank and n_batches != 60:
                continue

            kCP, key = jr.split(key, 2)
            cp_core_tensor_dims = (cp_r,)*3
            print(f'cp core tensor dimensions: {cp_core_tensor_dims}')
            cp_model = SALT(num_states=K,
                            num_emission_dims=N,
                            num_lags=L,
                            core_tensor_dims=cp_core_tensor_dims,
                            mode='cp',
                            l2_penalty=1e-4,
                            seed=kCP)
            lps, cp_model, posterior = cp_model.fit(ys, num_iters=n_iters,
                                                    method='em', tol=tol)

            inferred_output_factors = cp_model._emissions.output_factors
            inferred_core_tensors = cp_model._emissions.core_tensors
            inferred_input_factors = cp_model._emissions.input_factors
            inferred_lag_factors = cp_model._emissions.lag_factors

            cp_inferred_tensor = reconstruct_tensor(inferred_output_factors,
                                         inferred_core_tensors,
                                         inferred_input_factors,
                                         inferred_lag_factors)

            cp_mse = mse(ground_truth_tensor.flatten(), cp_inferred_tensor[0].flatten())
            cp_mses[n_batches].append(cp_mse)

            cp_test_lp = cp_model.marginal_likelihood(ys_test).sum()
            cp_test_lps[n_batches].append(cp_test_lp)

            print(cp_mses[n_batches][-1])
            print(cp_test_lps[n_batches][-1])
        
    return tucker_mses, tucker_test_lps, cp_mses, cp_test_lps, arhmm_mses, arhmm_test_lps, lds_mses, lds_test_lps, ground_truth_test_lps

def plot_result(tucker_mses, tucker_test_lps, cp_mses, cp_test_lps, 
                arhmm_mses, arhmm_test_lps, lds_mses, lds_test_lps, ground_truth_test_lps, min_tucker_rank, min_cp_rank,
                K=1, D=7, N=20, L=50):
    
    batch_T = 500
    test_batches = 10
    train_batches = [2,4,10,20,60]
    
    true_test_lp = ground_truth_test_lps[-1]

    cp_mses_plot = []
    tucker_mses_plot = []
    cp_test_lps_plot = []
    tucker_test_lps_plot = []
    for key in cp_mses:
        if key == 60:
            cp_mses_plot.append(cp_mses[key][-3])
            tucker_mses_plot.append(tucker_mses[key][-3])
            cp_test_lps_plot.append(cp_test_lps[key][-3])
            tucker_test_lps_plot.append(tucker_test_lps[key][-3])
        else:
            cp_mses_plot.append(cp_mses[key][-1])
            tucker_mses_plot.append(tucker_mses[key][-1])
            cp_test_lps_plot.append(cp_test_lps[key][-1])
            tucker_test_lps_plot.append(tucker_test_lps[key][-1])
        
    fig, axs = plt.subplots(1, 4, figsize=(42,6))
    axs[0].plot(np.arange(D-2,min_cp_rank+3), cp_mses[60], color='b', marker='o')
    axs[0].plot(np.arange(D-2,min_tucker_rank+3), tucker_mses[60], color='r', marker='o')
    axs[0].grid(True)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Mean squared error')
    axs[0].set_xlabel('SALT rank, $D$')
    axs[0].set_xticks(np.arange(D-2,min_cp_rank+3))
    axs[0].axvline(7, color='r', linestyle=':')
    axs[0].axvline(10, color='b', linestyle=':')

    axs[1].axhline(true_test_lp / (60*450), color='k', linestyle=':', label='Ground truth LDS')
    axs[1].plot(np.arange(D-2,min_cp_rank+3), np.array(cp_test_lps[60]) / (60*450), color='b', marker='o', label='CP-SALT')
    axs[1].plot(np.arange(D-2,min_tucker_rank+3), np.array(tucker_test_lps[60]) / (60*450), color='r', marker='o', label='Tucker-SALT')
    axs[1].grid(True)
    axs[1].set_ylabel('Test log-likelihood')
    axs[1].set_xlabel('SALT rank, $D$')
    axs[1].set_xticks(np.arange(D-2,min_cp_rank+2))
    axs[1].axvline(7, color='r', linestyle=':')
    axs[1].axvline(10, color='b', linestyle=':')
    axs[1].legend(loc='lower right')

    axs[2].plot(np.array(train_batches)*batch_T, np.array(cp_mses_plot), color='b', marker='o')
    axs[2].plot(np.array(train_batches)*batch_T, np.array(tucker_mses_plot), color='r', marker='o')
    axs[2].plot(np.array(train_batches)*batch_T, np.array([arhmm_mses[key][-1] for key in arhmm_mses]), color='g', marker='o')
    axs[2].plot(np.array(train_batches)*batch_T, np.array([lds_mses[key][-1] for key in lds_mses]), color='k', marker='o')
    axs[2].grid(True)
    axs[2].set_xticks(np.array(train_batches)*batch_T)
    axs[2].set_yscale('log')
    axs[2].set_xscale('log')
    axs[2].set_ylabel('Mean squared error')
    axs[2].set_xlabel('Timesteps of train data')

    axs[3].axhline(true_test_lp / (60*450), color='k', linestyle=':', label='Ground truth LDS')
    axs[3].plot(np.array(train_batches)*batch_T, np.array(cp_test_lps_plot)/ (60*450), color='b', marker='o', label='CP-SALT rank 10')
    axs[3].plot(np.array(train_batches)*batch_T, np.array(tucker_test_lps_plot)/ (60*450), color='r', marker='o', label='Tucker-SALT rank 7')
    axs[3].plot(np.array(train_batches[2:])*batch_T, np.array([arhmm_test_lps[key][-1] for key in arhmm_test_lps])[2:]/ (60*450), color='g', marker='o', label='ARHMM')
    axs[3].plot(np.array(train_batches)*batch_T, np.array([lds_test_lps[key][-1] for key in lds_test_lps])/ (60*450), color='k', marker='o', label='LDS')
    axs[3].grid(True)
    axs[3].set_xticks(np.array(train_batches)*batch_T)
    axs[3].set_xticklabels(np.array(train_batches)*batch_T)
    axs[3].set_xscale('log')
    axs[3].set_ylabel('Test log-likelihood')
    axs[3].set_xlabel('Timesteps of train data')
    axs[3].legend(loc='lower right')
    
    plt.savefig('./salt_figure2.png', dpi=600,  bbox_inches='tight')
    
    wandb.save('./salt_figure2.png')

    # Log some stuff to WandB.
    # wandb.log({
    #     'fig_full': fig,
    # }, commit=False)
    
    return fig