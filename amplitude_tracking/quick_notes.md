# Considerations on the Nekhoroshev fit on Hènon diffusion data

## The Hènon data

I consider the usual modulated Hénon map with the following parameters:

* $q_x = 0.31$, $q_y = 0.32$ (LHC top energy tunes);
* No octupolar kicks (for now);
* Different values of modulation intensity;
* Tracking up to $10^6$.

In order to fit numerically the diffusion coefficient in a specific point of the phase space, I consider the following:

* 10000 initial conditions with an initial gaussian distribution around the point of interest, with a very small initial standard deviation;
* I track the evolution of the standard deviation of the distribution, and I fit it with a function of the form $\sigma^2(t) = b + a t$. The slope $a$ is taken as a numerical estimate of the diffusion coefficient for the given point.

This procedure is repeated for different points of the phase space, and for different values of the modulation intensity. The results are shown in the following plots. When points are missing, it means that the particles were partially lost before a minimum number of $10^2$ turns was reached.

### Overview of the diffusion coefficient

| | |
|:---:|:---:|
|![](/img/henon_diffusion/eps_0_a.png)| ![](/img/henon_diffusion/eps_4_a.png) |
|![](/img/henon_diffusion/eps_8_a.png)| ![](/img/henon_diffusion/eps_16_a.png) |
|![](/img/henon_diffusion/eps_24_a.png)| ![](/img/henon_diffusion/eps_32_a.png) |
|![](/img/henon_diffusion/eps_42_a.png)| ![](/img/henon_diffusion/eps_64_a.png) |

## Averaging over the initial amplitude

The previous plots show that the diffusion coefficient is not constant over the phase space. In order to have a more general idea of the diffusion coefficient, I average over the initial amplitude of the particles. The results are shown in the following plots.

![](/img/henon_diffusion/diffusion_all.png)

Note that, inevitably, the mean diffusion coefficient will be affected at high amplitudes by the fact that particles are lost before a minimum number of turns is reached. This is highlighted in the following plot, where the diffusion coefficient for different modulations is shown along the number of valid angular samples.

| | |
|:---:|:---:|
|![](/img/henon_diffusion/diffusion_eps_0.png)| ![](/img/henon_diffusion/diffusion_eps_4.png) |
|![](/img/henon_diffusion/diffusion_eps_8.png)| ![](/img/henon_diffusion/diffusion_eps_16.png) |
|![](/img/henon_diffusion/diffusion_eps_24.png)| ![](/img/henon_diffusion/diffusion_eps_32.png) |
|![](/img/henon_diffusion/diffusion_eps_42.png)| ![](/img/henon_diffusion/diffusion_eps_64.png) |

This can lead to specific problems when fitting the diffusion coefficient, but can possibly be solved by performing a cut of the data that has less than a certain number of valid samples.

It can be observed how the gradual increase of the modulation intensity leads to a gradual increase of the diffusion coefficient, as well as a drastic change in the shape of the diffusion coefficient curve. For zero or close-to-zero modulation, the diffusion coefficient shows a well-defined core region with an extremely small value (to the point of being arguably comparable to an absence of diffusion), and a sharp increase in the diffusion coefficient at high amplitudes, corresponding effectively to the region where particles are starting to get lost. For higher modulation intensities, such sharp difference in regions is not present, and the diffusion coefficient steadily increases over various orders of magnitude as the amplitude increases.

## Fitting the diffusion coefficient

Let us consider the standard Nekhoroshev-like functional form

$$
D(I) = \exp\left[-2\left(\frac{I_\ast}{I}\right)^\frac{1}{2\kappa}\right]
$$

as we are not sure yet about the global timescale of the diffusion on the entire system we can consider the more general form $cD(I)$, where

$$
c = \int_{I_\mathrm{min}}^{I_\mathrm{max}} D(x) \mathrm{d}x
$$

with $I_\mathrm{min}$ and $I_\mathrm{max}$ being the minimum and maximum values of the action considered in the radial scans performed in the previous section. The data can then made comparable with the function by normalizing it as well over the same interval. The resulting normalized diffusion coefficient is shown in the following plot.

![](/img/henon_diffusion/diffusion_all_normalized.png)

### Fitting the data "directly"

The first attempt to fit the data was to consider the whole dataset as it is, forcing the same $\kappa$ for all the modulations and letting each modulation have its own $I_\ast$. $c$ is always evaluated on the same interval, as explained above. The results are shown in the following plot.

| |
|:---:|
|![](/img/henon_diffusion/fit_all_global_plot.png) |
| ![](/img/henon_diffusion/fit_all_shot.png) |

As it can be seen, the fit is not good at all. We observe both a bad reconstruction and a saturation of some parameters up to their maximum boundary. This is partially to be expected and due to the fact that some data needs to be discarded as it does not contain valid information related to a Nekhoroshev-like behaviour. In particular, the data at low amplitudes is not reliable in the case of low or zero modulation amplitudes, as it does not show significative diffusion in the first place, and the data at high amplitudes might have severe biases due to the fact that particles are lost before a minimum number of turns is reached. In order to solve this problem, we can consider a cut on the data, discarding all the points that have less than a certain number of valid samples, and/or discard the measured diffusion coefficient below a certain threshold.

The best improvement in the fitting performance was observed by performing the latter cut, discarding all the points with a diffusion coefficient below, for example, $10^{-4.5}$. The results are shown in the following plot.

| |
|:---:|
| ![](/img/henon_diffusion/diffusion_all_normalized_cuts_down_thresh_-4.0_fit.png)|
| ![](/img/henon_diffusion/shot_cut_down.png) |

While the result is better and looks promising, the correlation matrix between parameters always highlights a close-to-one correlation between the various $I_\ast$ and the $\kappa$ parameter. This inevitably leads to strong uncertainties on the "stability" of the fit and, most importantly, on the posibility to assess wether or not the parameter $\kappa$ can be treated as an emerging universal parameter of the system.

This can be greatly highlighted by considering different choices of lower threshold cuts on the diffusion coefficient. As it turns out, while the fit is always good, the values of the parameters are not stable at all, and do tend to displace themselves towards a well-defined curve as the threshold is altered. The following plots shows the values of the parameters as a function of the lower threshold cut on the diffusion coefficient.

| |
|:---:|
| ![](/img/henon_diffusion/fit_params_cuts_thresh_Iast.png)|
| ![](/img/henon_diffusion/fit_params_cuts_thresh_kappa.png)|
| ![](/img/henon_diffusion/fit_params_cuts.png)|

I have observed similar behaviour in similar cases where multiple datasets were available, and the only temporary solution I have found was to perform a separate scan on parameters and then land on a "best guess" for the parameters. This is not ideal, and I am still looking for a better solution.

Inevitably, this (for now) suggests that this specific functional form will always be strongly biased by the specific characteristics of the dataset, and that it is not possible to assess wether or not the parameter $\kappa$ can be treated as an emerging universal parameter of the system.