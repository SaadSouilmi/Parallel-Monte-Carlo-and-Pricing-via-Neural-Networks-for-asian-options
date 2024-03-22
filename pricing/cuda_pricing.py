"""
This module is an implementation of the monte carlo pricing of the payoff (S_T - I_T)_+ in numba.cuda.
"""

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import math
import numpy as np


################## Array reduction using numba.cuda.reduce API ####################################


@cuda.reduce
def sum_reduce(a, b):
    return a + b


# With this kernel each thread generates a path for the same initial conditions
@cuda.jit
def asian_option_kernel_1(
    rng_states, S, I, X, X_square, n_paths, n_steps, t, dt, r, sigma
):
    """Cuda kernel that computes samples n_paths payoffs of the asian option
    Args:
        - rng_states: array of cuda random states
        - S: array of size n_paths containing the initial value of S_t
        - I: array of size n_paths containing the initial value of I_t
        - X: array of size n_paths where we will save the sampled payoffs
        - X_square: array of size n_paths where we will save the sampled payoffs squared
        - n_paths: number of sampled paths
        - n_steps: number of time steps per path
        - t: time of pricing
        - dt: square root of time to maturity / n_steps
        - r: interest rate
        - sigma: volatility
    Returns:
        - Device.array: array containing the sampled payoffs"""

    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if tid >= n_paths:
        return  # Exit kernel to avoid index out of bounds error

    total_time = t + n_steps * dt * dt  # T = t + time to maturity
    path_integral = 0.0  # Variable where we store the path average of S
    for i in range(n_steps):
        # Generate a random normal increment
        rand = xoroshiro128p_normal_float32(rng_states, tid)
        # Increment the euler scheme
        S[tid] = S[tid] * (1.0 + r * dt * dt + sigma * dt * rand)
        # Increment the path integral
        path_integral += S[tid]

    # Update I_t with I_T
    I[tid] = dt * dt * path_integral / total_time + t * I[tid] / total_time

    # Compute payoff (S_T - I_T)_+ and it's square
    X[tid] = math.exp(-r * dt * dt * n_steps) * max(S[tid] - I[tid], 0.0)
    X_square[tid] = X[tid] * X[tid]


# Convenience function that calls the kernel 1
def asian_option_1(
    S: float,
    I: float,
    n_paths: int,
    n_steps: int,
    t: float,
    ttm: float,
    r: float,
    sigma: float,
    threadsperblock: int = 1024,
    seed: int = 42,
    rng_states=None,
) -> float:
    """Monte Carlo estimation of an Asian option's price using cuda
    Args:
        - S: initial value S_t
        - I: initial value I_t
        - n_paths: number of sampled paths
        - n_steps: number of time steps per path
        - t: time of pricing
        - ttm: time to maturity
        - r: interest_rate
        - sigma: volatility
    Returns:
        - tuple[float, float]: mc estimator and corresponding variance"""

    blockspergrid = blockspergrid = (n_paths + (threadsperblock - 1)) // threadsperblock
    dt = np.sqrt(ttm / n_steps)

    # Initializing initial conditions on gpu memory
    dtype = np.float32
    S_gpu = cuda.to_device(S * np.ones(n_paths, dtype=dtype))
    I_gpu = cuda.to_device(I * np.ones(n_paths, dtype=dtype))
    X_gpu = cuda.to_device(np.zeros(n_paths, dtype=dtype))
    X_square_gpu = cuda.to_device(np.zeros(n_paths, dtype=dtype))

    # Initializing random states
    if not rng_states:
        rng_states = create_xoroshiro128p_states(
            threadsperblock * blockspergrid, seed=seed
        )

    # Running the CUDA Kernel
    asian_option_kernel_1[blockspergrid, threadsperblock](
        rng_states, S_gpu, I_gpu, X_gpu, X_square_gpu, n_paths, n_steps, t, dt, r, sigma
    )

    mean = sum_reduce(X_gpu) / n_paths
    var = sum_reduce(X_square_gpu) / n_paths - mean**2

    return mean, var


#######################################################################################################


################## Array reduction without using numba.cuda.reduce ####################################


# With this kernel each thread generates a path for the same initial conditions
@cuda.jit
def asian_option_kernel_2(rng_states, S, I, X, n_paths, n_steps, t, dt, r, sigma):
    """Cuda kernel that computes samples n_paths payoffs of the asian option
    Args:
        - rng_states: array of cuda random states
        - S: array of size n_paths containing the initial value of S_t
        - I: array of size n_paths containing the initial value of I_t
        - X: array of size 2 where we will save the mean and variance
        - n_paths: number of sampled paths
        - n_steps: number of time steps per path
        - t: time of pricing
        - dt: square root of time to maturity / n_steps
        - r: interest rate
        - sigma: volatility
    Returns:
        - Device.array: array containing the sampled payoffs"""

    tid = cuda.threadIdx.x
    blocksize = cuda.blockDim.x

    idx = tid + cuda.blockIdx.x * blocksize

    if idx >= n_paths:
        return  # Exit kernel to avoid index out of bounds error

    total_time = t + n_steps * dt * dt  # T = t + time to maturity
    path_integral = 0.0  # Variable where we store the path average of S
    for i in range(n_steps):
        # Generate a random normal increment
        rand = xoroshiro128p_normal_float32(rng_states, idx)
        # Increment the euler scheme
        S[idx] = S[idx] * (1.0 + r * dt * dt + sigma * dt * rand)
        # Increment the path integral
        path_integral += S[idx]

    # Update I_t with I_T
    I[idx] = dt * dt * path_integral / total_time + t * I[idx] / total_time

    # Initializing shared memory
    xshared = cuda.shared.array(shape=(2, 1024), dtype=np.float32)
    xshared[0, tid] = (
        math.exp(-r * dt * dt * n_steps) * max(S[idx] - I[idx], 0.0) / n_paths
    )
    xshared[1, tid] = xshared[0, tid] ** 2 * n_paths

    cuda.syncthreads()  # Hold until shared memory is fully initialized within the block
    i = blocksize // 2
    while i > 0:
        if tid < i:
            xshared[0, tid] += xshared[0, tid + i]
            xshared[1, tid] += xshared[1, tid + i]
        cuda.syncthreads()  # Hold until all threads perform addition
        i //= 2

    if tid == 0:
        cuda.atomic.add(X, 0, xshared[0, 0])
        cuda.atomic.add(X, 1, xshared[1, 0])


# Convenience function that calls the kernel 2
def asian_option_2(
    S: float,
    I: float,
    n_paths: int,
    n_steps: int,
    t: float,
    ttm: float,
    r: float,
    sigma: float,
    threadsperblock: int = 1024,
    seed: int = 42,
    rng_states=None,
) -> tuple[float, float]:
    """Monte Carlo estimation of an Asian option's price using cuda
    Args:
        - S: initial value S_t
        - I: initial value I_t
        - n_paths: number of sampled paths
        - n_steps: number of time steps per path
        - t: time of pricing
        - ttm: time to maturity
        - r: interest_rate
        - sigma: volatility
    Returns:
        - tuple[float, float]: mc estimator and corresponding variance"""

    blockspergrid = (n_paths + (threadsperblock - 1)) // threadsperblock
    dt = np.sqrt(ttm / n_steps)

    # Initializing initial conditions on gpu memory
    dtype = np.float32
    S_gpu = cuda.to_device(S * np.ones(n_paths, dtype=dtype))
    I_gpu = cuda.to_device(I * np.ones(n_paths, dtype=dtype))
    X_gpu = cuda.to_device(np.zeros(2, dtype=dtype))

    # Initializing random states
    if not rng_states:
        rng_states = create_xoroshiro128p_states(
            threadsperblock * blockspergrid, seed=seed
        )

    # Running the CUDA Kernel
    asian_option_kernel_2[blockspergrid, threadsperblock](
        rng_states, S_gpu, I_gpu, X_gpu, n_paths, n_steps, t, dt, r, sigma
    )

    X = X_gpu.copy_to_host()

    return X[0], X[1] - X[0] ** 2


#######################################################################################################

################## Paralellizing over initial conditions ####################################


# Sampling n_paths per thread / Every thread computes mc price estimate for certain (S, I)
@cuda.jit
def asian_option_kernel_3(rng_states, S, I, X, n_paths, n_steps, t, dt, r, sigma):
    """Cuda kernel that computes samples n_paths payoffs of the asian option
    Args:
        - rng_states: array of cuda random states
        - S: array of size n_paths containing the initial value of S_t
        - I: array of size n_paths containing the initial value of I_t
        - X: array of size (n_paths, 2) where we will save the mean and variance
        - n_paths: number of sampled paths
        - n_steps: number of time steps per path
        - t: time of pricing
        - dt: square root of time to maturity / n_steps
        - r: interest rate
        - sigma: volatility
    Returns:
        - Device.array: array containing the sampled payoffs"""

    tid = cuda.threadIdx.x
    blocksize = cuda.blockDim.x

    idx = tid + cuda.blockIdx.x * blocksize

    if idx >= S.shape[0]:
        return  # Exit kernel to avoid index out of bounds error

    total_time = t[idx] + n_steps * dt * dt  # T = t + time to maturity
    discount = math.exp(-r * dt * dt * n_steps)
    mean = 0.0
    var = 0.0
    for i in range(n_paths):
        spot = S[idx]
        path_integral = 0.0  # Variable where we store the path average of S
        for j in range(n_steps):
            # Generate a random normal increment
            rand = xoroshiro128p_normal_float32(rng_states, idx)
            # Increment the euler scheme
            spot = spot * (1.0 + r * dt * dt + sigma * dt * rand)
            # Increment the path integral
            path_integral += spot
        path_integral = (
            dt * dt * path_integral / total_time + t[idx] * I[idx] / total_time
        )
        payoff = discount * max(spot - path_integral, 0.0) / n_paths
        mean += payoff
        var += payoff**2 * n_paths

    X[idx, 0] = mean
    X[idx, 1] = var


def asian_option_3(
    S: np.ndarray,
    I: np.ndarray,
    n_paths: int,
    n_steps: int,
    t: np.ndarray,
    ttm: float,
    r: float,
    sigma: float,
    threadsperblock: int = 1024,
    seed: int = 42,
    rng_states=None,
) -> np.ndarray:
    """Monte Carlo estimation of an Asian option's price using cuda
    Args:
        - S: initial value S_t
        - I: initial value I_t
        - n_paths: number of sampled paths
        - n_steps: number of time steps per path
        - t: time of pricing
        - ttm: time to maturity
        - r: interest_rate
        - sigma: volatility
    Retruns:
        - np.ndarray: array of size (len(S), 2) containing monte carlo estimates along side their variance
    """

    blockspergrid = (len(S) + (threadsperblock - 1)) // threadsperblock
    dt = np.sqrt(ttm / n_steps)

    # Initializing initial conditions on gpu memory
    dtype = np.float32
    S_gpu = cuda.to_device(S)
    I_gpu = cuda.to_device(I)
    t_gpu = cuda.to_device(t)
    X_gpu = cuda.to_device(np.zeros((len(S), 2), dtype=dtype))

    # Initializing random states
    if not rng_states:
        rng_states = create_xoroshiro128p_states(
            threadsperblock * blockspergrid, seed=seed
        )

    # Running the CUDA Kernel
    asian_option_kernel_3[blockspergrid, threadsperblock](
        rng_states, S_gpu, I_gpu, X_gpu, n_paths, n_steps, t_gpu, dt, r, sigma
    )

    X = X_gpu.copy_to_host()
    X[:, 1] = X[:, 1] - X[:, 0] ** 2

    return X


# ************************************************************************************************


# Sampling n_paths per block / Every block computes mc price estimate for certain (S, I)
@cuda.jit
def asian_option_kernel_4(rng_states, S, I, X, n_paths, n_steps, t, dt, r, sigma):
    """Cuda kernel that computes samples n_paths payoffs of the asian option
    Args:
        - rng_states: array of cuda random states
        - S: array of size n_paths containing the initial value of S_t
        - I: array of size n_paths containing the initial value of I_t
        - X: array of size (n_paths, 2) where we will save the mean and variance
        - n_paths: number of sampled paths
        - n_steps: number of time steps per path
        - t: time of pricing
        - dt: square root of time to maturity / n_steps
        - r: interest rate
        - sigma: volatility
    Returns:
        - Device.array: array containing the sampled payoffs"""

    tid = cuda.threadIdx.x
    blocksize = cuda.blockDim.x
    blockid = cuda.blockIdx.x

    idx = tid + blockid * blocksize

    if blockid >= S.shape[0]:
        return  # Exit kernel to avoid index out of bounds error

    eff_n_paths = (
        n_paths // blocksize
    )  # effective number of paths to be sampled by each thread

    total_time = t[blockid] + n_steps * dt * dt  # T = t + time to maturity
    discount = math.exp(-r * dt * dt * n_steps)
    mean = 0.0
    var = 0.0
    for i in range(eff_n_paths):
        spot = S[blockid]
        path_integral = 0.0  # Variable where we store the path average of S
        for j in range(n_steps):
            # Generate a random normal increment
            rand = xoroshiro128p_normal_float32(rng_states, idx)
            # Increment the euler scheme
            spot = spot * (1.0 + r * dt * dt + sigma * dt * rand)
            # Increment the path integral
            path_integral += spot
        path_integral = (
            dt * dt * path_integral / total_time + t[blockid] * I[blockid] / total_time
        )
        payoff = discount * max(spot - path_integral, 0.0) / n_paths
        mean += payoff
        var += payoff**2 * n_paths

    # Initializing shared memory
    xshared = cuda.shared.array(shape=(2, 1024), dtype=np.float32)
    xshared[0, tid] = mean
    xshared[1, tid] = var

    cuda.syncthreads()  # Hold until shared memory is fully initialized within the block
    i = blocksize // 2
    while i > 0:
        if tid < i:
            xshared[0, tid] += xshared[0, tid + i]
            xshared[1, tid] += xshared[1, tid + i]
        cuda.syncthreads()  # Add
        i //= 2

    if tid == 0:
        cuda.atomic.add(X, (blockid, 0), xshared[0, 0])
        cuda.atomic.add(X, (blockid, 1), xshared[1, 0])


def asian_option_4(
    S: np.ndarray,
    I: np.ndarray,
    n_paths: int,
    n_steps: int,
    t: np.ndarray,
    ttm: float,
    r: float,
    sigma: float,
    threadsperblock: int = 1024,
    seed: int = 42,
    rng_states=None,
) -> np.ndarray:
    """Monte Carlo estimation of an Asian option's price using cuda
    Args:
        - S: initial value S_t
        - I: initial value I_t
        - n_paths: number of sampled paths
        - n_steps: number of time steps per path
        - t: time of pricing
        - ttm: time to maturity
        - r: interest_rate
        - sigma: volatility
    Retruns:
        - np.ndarray: array of size (len(S), 2) containing monte carlo estimates along side their variance
    """

    blockspergrid = len(S)
    dt = np.sqrt(ttm / n_steps)

    # Initializing initial conditions on gpu memory
    dtype = np.float32
    S_gpu = cuda.to_device(S)
    I_gpu = cuda.to_device(I)
    t_gpu = cuda.to_device(t)
    X_gpu = cuda.to_device(np.zeros((len(S), 2), dtype=dtype))

    # Initializing random states
    if not rng_states:
        rng_states = create_xoroshiro128p_states(
            threadsperblock * blockspergrid, seed=seed
        )

    # Running the CUDA Kernel
    asian_option_kernel_4[blockspergrid, threadsperblock](
        rng_states, S_gpu, I_gpu, X_gpu, n_paths, n_steps, t_gpu, dt, r, sigma
    )

    X = X_gpu.copy_to_host()
    X[:, 1] = X[:, 1] - X[:, 0] ** 2

    return X
