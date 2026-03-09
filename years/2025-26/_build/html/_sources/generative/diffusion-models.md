# Diffusion Models

## Why Diffusion Models Enter the Course After VAEs and GANs

VAEs and GANs are historically fundamental, but they do not exhaust the modern theory of generative modeling. In recent years, diffusion and score-based models have become central in imaging because they combine high sample quality with a relatively stable training procedure. They are also particularly interesting from a mathematical point of view, because they connect denoising, stochastic processes, and Bayesian inference.

For students, diffusion models often look mysterious at first. The best way to introduce them is not to start from the final algorithm, but from the basic guiding idea: if generating complex images directly is difficult, perhaps it is easier to learn how to reverse a gradual noising process.

## Forward Diffusion

Let $x_0 \sim p_{\mathrm{data}}$ be a sample from the image distribution. The forward diffusion process progressively corrupts this sample by adding Gaussian noise over many steps:

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}\big(\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\big),
$$

where the sequence $(\beta_t)$ is called the variance schedule.

The meaning of this construction is simple. For small $t$, the sample still resembles the original image. For large $t$, the signal is progressively destroyed and the distribution approaches a simple Gaussian law.

One of the most important facts is that this multi-step process admits the closed form

$$
x_t
=
\sqrt{\bar{\alpha}_t}\,x_0
+
\sqrt{1-\bar{\alpha}_t}\,\varepsilon,
\qquad
\varepsilon\sim\mathcal{N}(0,I),
$$

with

$$
\alpha_t=1-\beta_t,
\qquad
\bar{\alpha}_t=\prod_{s=1}^t \alpha_s.
$$

This formula is extremely useful because it means we can generate a noisy sample at any time step directly from the clean image, without simulating the full chain step by step.

## The Reverse Problem

Once the forward noising mechanism is fixed, the generative task becomes the reverse one: starting from Gaussian noise, recover a sample distributed like a clean image.

In principle, one would like to model the reverse transitions

$$
p_\theta(x_{t-1}\mid x_t).
$$

If these reverse conditionals were known exactly, one could sample $x_T \sim \mathcal{N}(0,I)$ and then successively denoise until reaching a realistic image.

The remarkable insight of diffusion models is that one does not need to learn arbitrary reverse transitions directly. It is enough to learn the structure of denoising at each noise scale.

## Noise Prediction Objective

In the DDPM formulation, a neural network $\varepsilon_\theta(x_t,t)$ is trained to predict the noise used to generate $x_t$. The standard loss is

$$
\mathcal{L}_{\mathrm{DDPM}}(\theta)
=
\mathbb{E}_{x_0,\varepsilon,t}
\Big[
\|\varepsilon-\varepsilon_\theta(x_t,t)\|_2^2
\Big].
$$

This objective looks almost deceptively simple. One samples a clean image, chooses a time step, corrupts the image with Gaussian noise, and asks the network to predict the noise component. Yet this simple denoising problem is enough to learn a powerful generative model.

Pedagogically, this is worth emphasizing. Diffusion training is effective because it decomposes a difficult global generation problem into many local denoising tasks of varying difficulty.

## Denoising Point of View

It is helpful to explain the intuition behind the loss. If the network can reliably infer which part of $x_t$ is signal and which part is noise, then it can estimate how to move back toward the clean data manifold. Repeating this operation across multiple scales gradually reconstructs an image from pure noise.

Thus, diffusion models can be introduced to students as a hierarchy of denoisers linked together by a probabilistic time evolution.

## Score-Based Interpretation

There is a second, deeper interpretation. Instead of predicting the noise directly, one may think of the model as learning the score

$$
\nabla_x \log p_t(x),
$$

namely the gradient of the log density of the noisy image distribution at time $t$.

In continuous-time score-based modeling, one trains a network

$$
s_\theta(x_t,t)\approx \nabla_{x_t}\log p_t(x_t).
$$

The score indicates the direction in which probability density increases. Therefore, if one knows the score field, one knows how to move noisy samples toward more likely clean images.

This viewpoint is conceptually powerful because it links diffusion models to classical statistical objects such as likelihood gradients.

## Tweedie's Formula

One of the most beautiful identities in this area is Tweedie's formula. Under Gaussian perturbations, the posterior mean of the clean image given the noisy one satisfies

$$
\mathbb{E}[x_0\mid x_t]
=
x_t+\sigma_t^2 \nabla_{x_t}\log p_t(x_t).
$$

This formula tells us that once we know the score, we also know the conditional mean denoiser. It is exactly this bridge between denoising and score estimation that makes diffusion models so useful for inverse problems.

From a teaching perspective, this is a crucial milestone. It shows that diffusion models are not just image generators. They encode differential information about the image prior.

## Reverse Sampling

Once the score or noise predictor has been learned, one can generate samples by approximately simulating the reverse dynamics. There are several possibilities.

### Stochastic sampling

One follows a reverse-time stochastic process whose drift depends on the learned score. This tends to preserve the probabilistic nature of the model.

### Deterministic sampling

Methods such as DDIM or probability-flow ODE samplers replace the stochastic reverse process with a deterministic trajectory. These methods often reduce the number of function evaluations needed to obtain a good sample.

This distinction is worth explaining because it prepares the transition to flow matching later in the course.

## Why Diffusion Models Work Well in Imaging

There are several reasons for their success.

First, the training loss is stable compared with adversarial methods. One does not need to solve a minimax game as in GANs.

Second, the model naturally learns features across many noise scales. This is particularly valuable in imaging, where structures may appear at very different intensities and resolutions.

Third, the denoiser architecture used in practice is often a UNet. This brings all the benefits of multiscale image processing into the generative model.

Fourth, diffusion models can be adapted naturally to conditional settings, which is exactly what inverse problems require.

## Main Cost of Diffusion Models

The main drawback is computational cost. Sampling usually requires many denoising steps, and each step involves a large neural network evaluation. Compared with a single forward pass in a GAN or VAE, this can be much slower.

This is not a minor inconvenience. In scientific imaging, inference speed can matter greatly. It is one of the reasons the field has invested so much effort in faster samplers and alternative transport-based generative models.

## Why Diffusion Models Are So Relevant for Inverse Problems

For inverse problems, the central quantity of interest is often the posterior distribution

$$
p(x\mid y^\delta)
\propto
p(y^\delta\mid x)p(x).
$$

Classical regularization methods usually provide only a point estimate. Diffusion models, by contrast, can approximate the prior part of this posterior through score information. Since the posterior score decomposes as

$$
\nabla_x \log p(x\mid y^\delta)
=
\nabla_x \log p(y^\delta\mid x)
+
\nabla_x \log p(x),
$$

the diffusion model supplies the prior score term, while the forward model supplies the likelihood score term.

This is the moment in the teaching roadmap where generative modeling reconnects directly with inverse problems.

## Summary

The main ideas that students should retain are:

- diffusion models learn to reverse a progressive Gaussian corruption process;
- the DDPM loss trains a hierarchy of denoisers across noise scales;
- the score-based viewpoint connects the model with gradients of log densities;
- Tweedie's formula links score estimation to posterior mean denoising;
- diffusion models are powerful in imaging because they provide a strong learned prior, but they are computationally expensive at sampling time.
