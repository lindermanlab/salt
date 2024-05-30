pip install cython
pip install git+https://github.com/lindermanlab/ssm.git@hyun.salt#egg=ssmv0
pip install git+https://github.com/lindermanlab/ssm-jax.git@salt#egg=ssm
pip install tensorflow==2.8.0 tensorflow_probability==0.16.0
pip install chex==0.1.5 orbax==0.1.0 orbax-checkpoint==0.1.1
pip install --upgrade "jax[cuda12]"
pip install wandb

sudo apt install cm-super dvipng texlive-latex-extra texlive-latex-recommended
