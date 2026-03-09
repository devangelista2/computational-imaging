# Flow Matching Models

## Why Flow Matching Comes After Diffusion

Once students understand diffusion models, there is a natural next question: is stochastic denoising the only way to transport a simple distribution into a complex image distribution? Flow matching answers this question in the negative.

The main conceptual shift is the following. Diffusion models generate by reversing a noising process. Flow matching models instead learn a deterministic time-dependent vector field that transports samples from a simple base distribution to the data distribution.

This is important pedagogically because it shows that modern generative modeling is not tied to a single probabilistic mechanism. One can learn density structure either through stochastic score dynamics or through deterministic transport.

## Probability Paths

Let $x_0 \sim p_0$ be sampled from a simple base law, usually Gaussian, and let $x_1 \sim p_{\mathrm{data}}$ be a target image. The first step is to define an interpolation path between them:

$$
x_t = \phi_t(x_0,x_1),
\qquad
t \in [0,1].
$$

A particularly simple choice is linear interpolation:

$$
x_t = (1-t)x_0 + t x_1.
$$

This path should be interpreted as a family of intermediate random variables that gradually moves from noise to data.

## Velocity Fields

If the interpolation path is differentiable in time, it has an associated velocity

$$
u_t(x_0,x_1)=\frac{d}{dt}\phi_t(x_0,x_1).
$$

For the linear path one gets

$$
u_t(x_0,x_1)=x_1-x_0.
$$

The learning problem is then: can we train a neural network to predict the correct velocity at each intermediate point of the path?

## The Flow Matching Objective

One introduces a neural vector field

$$
v_\theta(x,t)
$$

and trains it by minimizing

$$
\mathcal{L}_{\mathrm{FM}}(\theta)
=
\mathbb{E}_{x_0,x_1,t}
\left[
\|v_\theta(x_t,t)-u_t(x_0,x_1)\|_2^2
\right].
$$

The meaning of this loss is very direct. At every time $t$, the model is asked to reproduce the velocity that should move the sample along the chosen path from noise toward data.

Once training is complete, the learned vector field defines the ODE

$$
\frac{d}{dt}x_t = v_\theta(x_t,t),
\qquad
x_0 \sim p_0.
$$

Solving this ODE transports the base distribution into the data distribution.

## Comparison With Diffusion

The comparison with diffusion is very instructive.

Diffusion models learn how to denoise a sample corrupted by a stochastic process. Flow matching models learn how to move a sample deterministically through a velocity field.

Diffusion emphasizes score estimation and reverse-time stochastic dynamics.

Flow matching emphasizes transport and ordinary differential equations.

The two viewpoints are related, but they give different algorithmic and conceptual advantages.

## Why Flow Matching Is Attractive

One major attraction is speed. Because the learned dynamics can be integrated with relatively few ODE solver steps, sampling can be significantly faster than in many diffusion pipelines.

Another attraction is flexibility. The model does not need to be tied to a particular noising schedule in the same way as diffusion. Instead, one chooses a probability path and learns the associated transport field.

This means that flow matching is often presented as a promising route toward faster high-quality generative modeling.

## The Importance of the Chosen Path

At this stage, students should be warned that the path $\phi_t$ is not arbitrary in practice. Different choices of interpolation yield different target vector fields and therefore different learning difficulties.

A poor path may force the model to learn unnecessarily complicated transport dynamics. A good path can make the flow smoother and easier to approximate.

This is another instance of a recurring course principle: the design of the training objective already contains substantial modeling assumptions.

## Conditional Flow Matching for Inverse Problems

To adapt the model to inverse problems, one conditions the vector field on the measured datum:

$$
v_\theta(x,t,y^\delta).
$$

During training, one uses pairs $(x^\dagger,y^\delta)$ satisfying

$$
y^\delta = A x^\dagger + e.
$$

The conditional objective becomes

$$
\mathcal{L}_{\mathrm{CFM}}(\theta)
=
\mathbb{E}_{x_0,x^\dagger,y^\delta,t}
\left[
\|v_\theta(x_t,t,y^\delta)-u_t(x_0,x^\dagger)\|_2^2
\right].
$$

At inference time, for a fixed measurement $y^\delta$, one integrates the learned conditional ODE from a sample of the base distribution. The result is a reconstruction distributed according to the learned conditional image law.

## Why This Is Interesting in Imaging

This conditional viewpoint is powerful for at least three reasons.

First, it allows uncertainty-aware reconstruction. One may sample multiple plausible outputs for the same measurement and thereby explore posterior variability.

Second, it can be computationally attractive because deterministic transport may require fewer steps than iterative reverse diffusion.

Third, it integrates naturally with the conditioning information coming from the acquisition process, making it a promising framework for modern learned inverse solvers.

## Summary

The conceptual roadmap of this chapter is:

- flow matching learns deterministic transport rather than reverse denoising;
- the model is trained to predict the velocity field along a chosen probability path;
- ODE integration transforms a simple base law into the data distribution;
- conditional flow matching adapts this idea to inverse problems by conditioning on the measurements;
- the main appeal of the method is the possibility of faster posterior-like sampling compared with diffusion approaches.
