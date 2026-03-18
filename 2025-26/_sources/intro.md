# Computational Imaging

These notes collect the material for Module 2 of the course *Computational Imaging* for the academic year 2025-26. Module 1, taught by Professor Elena Loli Piccolomini, focused on the mathematical formulation of inverse problems, classical regularization techniques, and optimization-based reconstruction. Module 2 continues from that foundation and asks what changes when the reconstruction operator is no longer hand-designed, but learned from data through neural and generative models.

The guiding philosophy of the module is that modern imaging sits at the intersection of mathematics, computation, and modeling. A neural network is not just a piece of software. It is a parameterized nonlinear map. A generative model is not just a black box that creates images. It is an implicit or explicit probabilistic prior. A training pipeline is not just a collection of scripts. It is a concrete way of encoding assumptions on the distribution of images, the acquisition model, and the perturbations affecting the data.

For this reason, the notes will not present machine learning as a disconnected topic. The goal is to integrate it into the language of inverse problems as naturally as possible.

## Teaching Roadmap and Main Questions

The whole module can be organized around a few central questions, and these questions define the internal logic of the lectures.

The first question is how to reinterpret image reconstruction as a **supervised learning** problem. If we observe pairs of measurements and clean images, can we learn a map that approximates the inverse operator directly?

The second question is architectural. Once we decide to learn a reconstruction map, how should we choose the model class? Why are **convolutions** natural for images? Why are **UNets** so successful? When do transformers become useful?

The third question concerns realism. What happens if the training data are generated with an overly simplified forward model? Why is noise modeling not optional? What is the **inverse crime**, and why can it produce very strong but potentially misleading numerical results?

The fourth question is probabilistic. A single point estimate is often not enough in ill-posed imaging problems. Can generative models describe the distribution of plausible images, and can this learned prior be incorporated into the reconstruction process?

These are the questions that organize the teaching roadmap of the course.

The same roadmap also clarifies the intended learning goals. By the end of Module 2, the student should be able to:

- formulate **end-to-end reconstruction** as **empirical risk** minimization over parameterized inverse maps;
- explain why nonlinear models are needed beyond **linear estimators**;
- discuss the mathematical role of **convolutions**, **receptive fields**, skip connections, multiscale processing, and self-attention;
- compare **CNNs**, **UNets**, residual variants, and **Vision Transformers (ViT)** from the viewpoint of computational imaging;
- explain self-supervised training strategies when clean targets are unavailable;
- define the **inverse crime** and discuss the impact of **model mismatch** and realistic noise;
- derive and interpret the main ideas behind VAEs, **GANs**, diffusion models, and **flow matching** models;
- explain how generative models can be used as learned priors, latent parameterizations, denoisers, or posterior samplers in inverse problems.

This should not be interpreted rigidly. Some topics, especially diffusion models and the **inverse crime**, can naturally expand if discussion is lively. The point of the table is simply to show that the module has an internal progression rather than being a collection of isolated techniques.

## How to Use These Notes

The notes are designed to be read alongside executable Jupyter notebooks. Each major topic comes with a companion notebook where the ideas are implemented and tested. Before working through the material, make sure your Python environment is correctly configured. The [environment setup](intro/environment-setup) page describes how to do this using `uv`, including how to install Python itself on a machine where it is not yet present.

## Office hour
The office hour is Wednesday morning, from 9a.m. to 12p.m., via Microsoft Teams or in my office, at Ex Veneta, close to the Zanolini train station. 

Either cases, it is **mandatory** to set an appointment by e-mail, via the address ``davide.evangelista5@unibo.it``.