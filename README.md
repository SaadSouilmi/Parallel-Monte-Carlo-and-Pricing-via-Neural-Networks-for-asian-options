# Cuda-Project
Cuda course project.

The black scholes model for the spot price under risk neutral measure is:
$$dS_t = rS_tdt + \sigma S_tdW(t)$$
Which is equivalent to:
$$d\log(S_t) = \left(r - \frac{\sigma^2}{2}\right)dt + \sigma dW(t)$$
The corresponding Euler-Maruyama scheme for a discrete grid $\{t_k = \frac{kT}{n}| k\in\{0,...,n\}\}$ is:
$$S_{t_{k+1}} = S_{t_k}\left(1 + r\frac{T}{n} + \sigma\sqrt{\frac{T}{n}} Z_{k+1}\right)$$
Otherwise, if we consider the log spot: 
$$\log(S_{t_{k+1}})  = \log(S_{t_k}) + \left(r - \frac{\sigma^2}{2}\right)\frac{T}{n} + \sigma \sqrt\frac{T}{n} Z_{k+1}$$
Where $Z$ are i.i.d standard normal variables.

We want to compute the following price :
$$F(t, T, S_t, I_t, r, \sigma) = e^{-r(T-t)}\mathbb E\left[\left(S_T - I_T\right)^+\Big|\,S_t, \,I_t\right]$$
$$\text{Where :}\quad I_t = \frac{1}{t}\int_{0}^{t} S_u\,du$$

In order to compute the price, we sample trajectories on GPU using cuda.

We also want to learn the price map using neural networks, which translates to :
$$\theta^* \in argmin_{\theta\in\Theta} L(\theta) = argmin_{\theta\in\Theta}\mathbb E_{x\sim D}\left[(F(x) - T_\theta(x))^2\right]$$

Where $x = (t, T, S_t, I_t, r, \sigma)$ and $D$ is a prior distribution over the parameter space. We can rewrite the problem as:
$$\theta^* \in argmin_{\theta\in\Theta}\tilde{L}(\theta) = argmin_{\theta\in\Theta}\mathbb E_{x\sim D}\left[\mathbb E_{(S_T, I_T)}\left[ (S_T - I_T)^+ - T_\theta(x)^2 \Big | x\right]\right]$$

$$\text{Since  }F(x) = \mathbb E\left[(S_T - I_T)^+|x\right]$$

$$\tilde{L}(\theta) - L(\theta) = \mathbb E_{x\sim D}\left[\text{Var}((S_T - I_T)^+|x)\right]\Longrightarrow \partial_\theta \tilde{L}(\theta) = \partial_\theta L(\theta)$$

Ergo, we can either have a neural network learn a monte carlo approximation of the price or learn directly sample payoffs.
