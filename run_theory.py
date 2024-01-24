import wandb

from utils.theory_util import make_lds, train_models, plot_result
from types import SimpleNamespace


# PARAMS -------------------------------------------------------------------------------------------------------------------------------------- PARAMS

env_args = {
    'seed': 0,
    'use_wandb': True,
    'wandb_project': 'salt-theory',
    'wandb_entity': 'jhdlee',
}

# Set up experimental logging.
args = {
    'env_args': env_args,
}
wandb.init(project=env_args['wandb_project'], entity=env_args['wandb_entity'], config=args)

# Make the args easier to use.
env_args = SimpleNamespace(**env_args)


# SCRIPT -------------------------------------------------------------------------------------------------------------------------------------- SCRIPT

# Set up the lds.
true_lds, ground_truth_mats, min_tucker_rank, min_cp_rank, key = make_lds()

# Train models.
tucker_mses, tucker_test_lps, cp_mses, cp_test_lps, arhmm_mses, arhmm_test_lps, lds_mses, lds_test_lps, ground_truth_test_lps = train_models(true_lds, 
                                                                                                                      ground_truth_mats, 
                                                                                                                      min_tucker_rank, 
                                                                                                                      min_cp_rank, 
                                                                                                                      key)

# Plot the comparison of results of SALT and DSARF.
fig = plot_result(tucker_mses, tucker_test_lps, cp_mses, cp_test_lps, 
                  arhmm_mses, arhmm_test_lps, lds_mses, lds_test_lps, ground_truth_test_lps,
                  min_tucker_rank, min_cp_rank)


print('Done, exiting!')