"""
This module is an implementation of the monte carlo pricing of the payoff (S_T - I_T)_+ in torch using gpu.
This Implementatio is mainly used to debug the cuda implementation and to generate data for MLP training.
"""

import torch

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g_cuda = torch.Generator(device="cuda")
g_cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def asian_option(
    n_paths: int,
    n_steps: int,
    spot: float,
    path_integral: float,
    t: float,
    dt: float,
    r: float,
    sigma: float,
) -> torch.Tensor:
    """Sample n_paths of asian option payoffs
    Args:
        - n_paths: number of simulations
        - n_steps: number of steps in the euler scheme
        - spot: initial value of the asset
        - path_integral: mean of the value of the asset at time t
        - t: time of pricing
        - dt: square root of time step
        - r: interest rate
        - sigma: volatility
    Returns:
        - torch.Tensor: sample payoffs
    """
    T = t + dt * dt * n_steps  # Total time as t + time to maturity
    # Sample independant normal increments
    sample = torch.normal(
        mean=torch.zeros(n_paths, n_steps).to(device), std=1, generator=g_cuda
    )
    sample = (r - 0.5 * sigma**2) * dt * dt + dt * sigma * sample
    # Compute scheme
    sample = spot * torch.exp(sample.cumsum(axis=1))
    # Compute payoff
    sample = torch.exp(-r * dt * dt * n_steps) * torch.maximum(
        sample[:, -1]
        - t * path_integral / T
        - dt * dt * n_steps * sample.mean(axis=1) / T,
        torch.zeros(1).to(device),
    )
    # We use the same name variable sample to free up gpu memory of unecessary data
    return sample


# vectorizing the payoffs sampler for batch operations
vmap_asian_option = torch.vmap(
    asian_option,
    in_dims=(None, None, 0, 0, 0, 0, 0, 0),
    out_dims=0,
    randomness="different",
)
