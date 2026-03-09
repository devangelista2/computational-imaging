# Variational Autoencoders and GANs

## Why the Course Now Moves to Generative Models

Up to this point the main object of study has been a discriminative reconstructor, namely a map that takes a measured datum and outputs a reconstructed image. This is a natural place to start, but it has a limitation. It typically returns a single answer, while inverse problems are often ambiguous. Several different images may be consistent with the same measurements, especially when the forward operator is ill-conditioned or undersampled.

This motivates a broader question: instead of learning only how to map data to images, can we learn the image distribution itself? If we can do this, then we gain access to a learned image prior. This prior can later be combined with the data-consistency information coming from the forward model.

The rest of the course therefore shifts from discriminative learning to generative modeling. Historically, the first major deep generative paradigms were variational autoencoders and generative adversarial networks. They are different in philosophy, in mathematical formulation, and in the kind of prior they induce.

## Latent Variables and the Low-Dimensional Hypothesis

Both VAEs and GANs are built around a common idea: high-dimensional images may be generated from lower-dimensional latent variables. If $x \in \mathbb{R}^n$ is an image and $z \in \mathbb{R}^k$ is a latent code with $k \ll n$, then one posits a generator or decoder map

$$
G_\theta : \mathbb{R}^k \to \mathbb{R}^n,
\qquad
x \approx G_\theta(z).
$$

This expresses the belief that realistic images occupy only a tiny, structured region inside the ambient high-dimensional space. Inverse problems benefit enormously from this viewpoint, because restricting the search space from $\mathbb{R}^n$ to the range of a generator can act as a powerful regularizer.

## Autoencoders as the Starting Point

Before introducing VAEs, it is useful to recall the ordinary autoencoder. An autoencoder consists of:

- an encoder $E_\phi$ that maps an image $x$ to a latent representation $z$;
- a decoder $G_\theta$ that maps the latent representation back to an approximate reconstruction.

The training problem is usually

$$
\min_{\phi,\theta}
\mathbb{E}\big[\|x-G_\theta(E_\phi(x))\|^2\big].
$$

This architecture learns a compressed representation of the data, but by itself it does not define a full probabilistic generative model. In particular, it does not tell us how latent codes should be sampled in order to generate new images.

This is the point where the VAE enters.

## Variational Autoencoders

A VAE defines a probabilistic latent-variable model

$$
p_\theta(x,z)=p_\theta(x\mid z)p(z),
$$

where the prior on the latent variable is often chosen as

$$
p(z)=\mathcal{N}(0,I).
$$

The induced model for the image is obtained by marginalization:

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz.
$$

This is the key point. The VAE is not merely compressing images. It is trying to assign a probability law to them through latent variables.

## Why the Exact Likelihood Is Difficult

The log-likelihood

$$
\log p_\theta(x)
=
\log \int p_\theta(x\mid z)p(z)\,dz
$$

is generally intractable because the integral over latent space cannot be computed exactly for a complex neural decoder. The VAE solves this by introducing an approximate posterior distribution

$$
q_\phi(z\mid x),
$$

often called the encoder distribution, and then deriving a tractable lower bound on the log-likelihood.

## The ELBO

The central identity of the VAE is

$$
\log p_\theta(x)
\geq
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-\operatorname{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big).
$$

The right-hand side is the evidence lower bound, or ELBO. It has a very instructive decomposition.

The first term is the reconstruction term. It asks the decoder to explain the observed image well when the latent variable is sampled from the encoder distribution.

The second term is the KL regularization term. It pushes the encoder posterior toward the prior $p(z)$. This is what makes latent sampling possible and gives the model genuine generative semantics.

## Interpreting the Gaussian Decoder

If the conditional model is chosen as

$$
p_\theta(x\mid z)=\mathcal{N}(\mu_\theta(z),\sigma^2I),
$$

then maximizing the reconstruction term is equivalent, up to constants, to minimizing

$$
\|x-\mu_\theta(z)\|_2^2.
$$

This is one of the reasons VAEs are often associated with smooth reconstructions. The Gaussian decoder and the averaged nature of the likelihood term favor mean-like outputs.

This is not a flaw of implementation. It is a direct consequence of the probabilistic assumptions built into the model.

## Reparameterization Trick

There is one additional ingredient that deserves explicit explanation in teaching: how does one differentiate through the random latent variable? The standard answer is the reparameterization trick. If the encoder outputs a mean $\mu_\phi(x)$ and a standard deviation $\sigma_\phi(x)$, then one writes

$$
z = \mu_\phi(x) + \sigma_\phi(x)\odot \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,I).
$$

This converts sampling into a deterministic function of the parameters and an auxiliary random variable. Backpropagation can then proceed normally.

This is a good example of how probability and optimization interact in deep learning.

## Strengths and Weaknesses of VAEs

VAEs have several attractive properties:

- a principled probabilistic interpretation;
- an explicit encoder and decoder;
- a latent space that is regularized and sampleable;
- a meaningful relation to approximate Bayesian inference.

Their weaknesses are equally important:

- generated samples and reconstructions can be too smooth;
- the ELBO may not align perfectly with perceptual visual quality;
- the chosen likelihood model can be too simplistic for complex textures.

For inverse problems, these strengths and weaknesses matter because a smooth latent prior may be mathematically convenient but may underrepresent fine imaging detail.

## Generative Adversarial Networks

GANs take a strikingly different path. Instead of optimizing a tractable lower bound on the data likelihood, a GAN trains two networks in competition:

- a generator $G_\theta(z)$ that maps latent codes to images;
- a discriminator $D_\psi(x)$ that tries to distinguish real samples from generated ones.

The classical minimax problem is

$$
\min_\theta \max_\psi
\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D_\psi(x)]
+
\mathbb{E}_{z\sim p(z)}[\log(1-D_\psi(G_\theta(z)))].
$$

The generator tries to fool the discriminator, while the discriminator tries not to be fooled.

## Intuition Behind Adversarial Training

The key idea is that instead of comparing each generated sample to a specific target image, one asks whether generated images as a distribution are indistinguishable from real images. This is a radical conceptual shift.

Because GANs are not driven by pointwise reconstruction losses in the usual way, they can generate much sharper and more realistic fine detail than VAEs. This is one of the main reasons GANs attracted enormous interest.

## Typical Difficulties of GANs

At the same time, GANs are notoriously delicate. Three issues are especially important in teaching.

### Mode collapse

The generator may map many latent vectors to very similar images, thereby covering only a limited portion of the data distribution.

### Training instability

The optimization problem is not a simple minimization but a saddle-point game. This can produce oscillations, imbalance between generator and discriminator, and sensitivity to hyperparameters.

### Lack of an explicit tractable likelihood

GANs often generate excellent samples, but they do not naturally provide a convenient density model. This limits the direct probabilistic interpretation of the learned prior.

## Why VAEs and GANs Are Complementary in a Course

These two models are pedagogically valuable precisely because they illustrate two contrasting philosophies of generative modeling.

The VAE is likelihood-oriented, variational, and probabilistically explicit.

The GAN is adversarial, game-theoretic, and focused on distributional realism rather than tractable density.

Seeing both helps students understand that "generative model" is not a single recipe. It is a family of approaches for representing complex data distributions.

## Relation to Inverse Problems

Now we come to the point that matters most for the course. Why are these models useful in imaging inverse problems?

The answer is that both VAEs and GANs define low-dimensional models of plausible images. If the reconstruction is constrained to lie in the range of a decoder or generator, then the search space is drastically reduced.

Suppose

$$
x = G_\theta(z).
$$

Then instead of solving for $x \in \mathbb{R}^n$, one may solve for $z \in \mathbb{R}^k$:

$$
\widehat{z}
=
\operatorname*{arg\,min}_{z}
\|AG_\theta(z)-y^\delta\|^2+\lambda\|z\|^2.
$$

The recovered image is then

$$
\widehat{x}=G_\theta(\widehat{z}).
$$

This is powerful because $k \ll n$. However, it also introduces a bias: the true image must be well approximated by the range of the generator.

## Summary

This chapter should leave the following clear roadmap:

- generative modeling aims to learn the image distribution, not only an input-output map;
- VAEs build a probabilistic latent-variable model through the ELBO;
- GANs learn realism through adversarial distribution matching;
- VAEs are principled and stable but often smooth;
- GANs are sharp and expressive but harder to train and analyze;
- both are useful in inverse problems because they provide learned low-dimensional priors over images.
