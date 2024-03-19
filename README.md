# Cuda-Project
Cuda course project.

The black scholes model for the spot price under risk neutral measure is:
$$dS_t = rS_t\,dt + \sigma S_t\,dW(t)$$
Which is equivalent to:
$$d\log(S_t) = \left(r - \frac{\sigma^2}{2}\right)\,dt + \sigma\,dW(t)$$
The corresponding Euler-Maruyama scheme for a discrete grid $\{t_k = \frac{kT}{n},\; k\in\{0,...,n\}\}$ is:
$$d\log(S_{t_{k+1}})  = d\log(S_{t_k}) + \left(r - \frac{\sigma^2}{2}\right)\frac{T}{n} + \sigma \sqrt\frac{T}{n} Z_{k+1}$$
Where $Z$ are i.i.d standard normal variables.

We want to compute the following price :
$$F(t, T, S_t, I_t, r, \sigma) = e^{-r(T-t)}\mathbb E\left[\left(S_T - I_T\right)^+\Big|\,S_t, \,I_t\right]$$
$$\text{Where :}\quad I_t = \frac{1}{t}\int_{0}^{t} S_u\,du$$

In order to compute the price, we sample trajectories on GPU using cuda.

We also want to learn the price map using neural networks, which translates to :
$$\theta^* \in \argmin_{\theta\in\Theta}\mathbb E_{x\sim D}\left[(F(x) - T_\theta(x))^2\right]$$

Where $x = (t, T, S_t, I_t, r, \sigma)$ and $D$ is a prior distribution over the parameter space. Since we are attempting to learn an expectation, we can rewrite the problem as:
$$\theta^* \in \argmin_{\theta\in\Theta}\mathbb E_{x\sim D}\left[\mathbb E_{(S_T, I_T)}\left[ (S_T - I_T)^+ - T_\theta(x))^2 \Big | x\right]\right]$$

We can thus train our network on payoffs and test it agaisnt MC estimations of the price.