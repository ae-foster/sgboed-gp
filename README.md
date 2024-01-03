# Stochastic-gradient Bayesian Optimal Experimental Design with Gaussian Processes

This is a standalone repo containing the code for the blogs posts:
0. [Stochastic-gradient Bayesian Optimal Experimental Design with Gaussian Processes](url.tbd.com)


## Environment set-up
The `environment.yml` files contains the specifications for the environment that I used, which uses CUDA=11.8.

Alternatively, you can create an environment on a different platform via

```bash
conda create -n sgboed python=3.11
conda activate sgboed
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia  # Adjust depending on the platform
conda install numpy matplotlib tqdm
pip install pyro-ppl
```


