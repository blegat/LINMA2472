### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 3fde5f41-32a9-4585-9b6a-131173947346
using PlutoUI, DataFrames, PrettyTables, LinearAlgebra, Luxor, LaTeXStrings, MathTeXEngine

# ╔═╡ 19741723-3cc3-40df-83a3-fb2727504cd1
include("utils.jl")

# ╔═╡ 7d05bc54-a693-11ef-2d83-a5caeb91676e
section("Diffusion Models")

# ╔═╡ 44bb88d2-2fc7-4901-9fc3-890e54754932
frametitle("Tweedie's formula")

# ╔═╡ 423874b8-6d4c-423a-bf45-39d590600066
md"""
Let ``\sigma`` be a constant and consider a random variable ``X`` with probability density function ``f_X`` as well as a random Gaussian noise ``\mathcal{E} \sim \mathcal{N}(0, 1)`` that is independent from ``X``.
If ``Y = X + \sigma \mathcal{E}`` then
```math
\mathbb{E}[X|Y=y] = y + \sigma^2 \nabla_y \log f_Y(y) \qquad
\mathbb{E}[\mathcal{E}|Y=y] = -\sigma \nabla_y \log f_Y(y).
```
"""

# ╔═╡ abc07fcf-35f5-43eb-90aa-23ec0b581e56
qa(md"Proof",
md"""
Using the fact that ``\mathcal{E}`` is normally distributed,
we have
```math
\begin{align}
\nabla_y f_\mathcal{E}((y - x)/\sigma)
& = \frac{1}{\sqrt{2\pi}} \nabla_y \exp\left(-\frac{1}{2\sigma^2}\|x-y\|^2\right)\\
& = -\frac{\nabla_y \|x-y\|^2}{2\sigma^2} f_\mathcal{E}((y - x)/\sigma)\\
& = \frac{x-y}{\sigma^2} f_\mathcal{E}((y - x)/\sigma)\\
\end{align}
```
We have
```math
\begin{align*}
f_{X,Y}(x,y) & = f_X(x) f_\mathcal{E}((y - x)/\sigma)\\
\nabla_y f_{X,Y}(x,y) & = f_X(x) \nabla_y f_\mathcal{E}((y - x)/\sigma)\\
& = \frac{x-y}{\sigma^2} f_X(x) f_\mathcal{E}((y - x)/\sigma)\\
& = \frac{x-y}{\sigma^2} f_{X,Y}(x,y)\\
f_Y(y)
& = \int_x f_{X,Y}(x,y) \, \text{d} x\\
\nabla_y f_Y(y)
& = \int_x \nabla_y f_{X,Y}(x,y) \, \text{d} x\\
& = \int_x \frac{x-y}{\sigma^2} f_{X,Y}(x,y) \, \text{d} x\\
\sigma^2 \nabla_y f_Y(y)
& = \int_x x f_{X,Y}(x,y) \, \text{d} x - y \int_x f_{X,Y}(x,y) \, \text{d} x\\
& = \int_x x f_{X|Y}(x|y) \, \text{d} x f_{Y}(y) - y f_{Y}(y)\\
\sigma^2 \nabla_y f_Y(y) / f_{Y}(y)
& = \mathbb{E}[X | Y = y] - y\\
y + \sigma^2 \nabla_y \log f_Y(y) & = \mathbb{E}[X | Y = y].
\end{align*}
```
Now we have
```math
\begin{align}
\mathbb{E}[\mathcal{E}|Y=y]
& =
\frac{\mathbb{E}[Y|Y=y] - \mathbb{E}[X|Y=y]}{\sigma}\\
& =
\frac{
y - (y + \sigma^2 \nabla_y \log f_Y(y))
}{\sigma}\\
& =
-\sigma \nabla_y \log f_Y(y).
\end{align}
```
""")

# ╔═╡ df9e106b-16c3-4605-b5f6-0a4e82dd4b3b
HAlign(md"""
From Tweedie's formula, ``\epsilon`` is estimated to be
``-\sigma \nabla_y \log f_Y(y)``.

**Sampler** *Langevin dynamics*:
```math
\begin{multline}
y_{k+1} = y_k + \delta_k \nabla_y \log f_Y(y_k) + \sqrt{2\delta_k} w_k\\
\text{where } w_k \sim \mathcal{N}(0, 1)
\end{multline}
```

[Image source](https://yang-song.net/blog/2021/score/).
""",
img("langevin.gif", :width => 200),
)

# ╔═╡ eaf0bcaa-75d9-4d9c-9e19-ca5a8532ed35
frametitle("Score matching")

# ╔═╡ 687f5cb9-5a17-49cb-8d96-a35b66fbcba3
md"""
Diffusion models are also known as *energy-based models* and then *score-matching*.

For **fixed** ``\sigma``, the matching is ``\epsilon_\theta(y) \approx \mathbb{E}[\mathcal{E}|X + \sigma \mathcal{E} = y] = -\sigma \nabla_y \log f_{X+\sigma\mathcal{E}}(y)``.

**Training**: sample ``x`` / pick ``x`` in dataset, sample ``\varepsilon``, update ``\theta`` to minimize (e.g., using gradient descent), the loss:
```math
\mathbb{E}[\|\epsilon_\theta(X + \sigma \mathcal{E}) - \mathcal{E}\|^2]
```
[Image source](https://yang-song.net/blog/2021/score/).
"""

# ╔═╡ 3cdcd14a-9102-4ed4-b186-629ce183b73c
img("score")

# ╔═╡ a835f101-bfd0-428e-adba-64f5de7af14c
frametitle("Issue with small variance")

# ╔═╡ 0886ac78-6530-417b-80a5-5410c4719dd2
md"""
If ``\sigma`` is too small then the support of ``X + \sigma \mathcal{E}`` may not cover the whole state space → inaccurate ``\epsilon_\theta(y)`` in these parts. More formally, the loss is the Fisher divergence:
```math
\mathbb{E}[\|\epsilon_\theta(y) + \sigma \nabla_y \log f_Y(y)\|^2]
= \int_y f_Y(y) \|\epsilon_\theta(y) + \sigma \nabla_y \log f_Y(y)\|^2 \,\text{d}y
```
so it is inaccurate for ``y`` such that ``f_Y(y)`` is too small. [Image source](https://yang-song.net/blog/2021/score/).
"""

# ╔═╡ d826a144-2c79-4c1e-ae3f-c11a4b5353d4
img("inaccurate.jpg")

# ╔═╡ 7e7e2845-4bee-4dbc-b0f9-d964cd362248
frametitle("Issue with large variance")

# ╔═╡ e33b16ec-4526-4dbe-affa-3e68aa508ab8
md"""
If ``\sigma`` is too large then ``\epsilon_\theta(y) \approx  -\sigma \nabla_y \log f_{X+\sigma\mathcal{E}}(y)`` everywhere but the distribution ``Y \sim X + \sigma \mathcal{E}`` is too noisy, less specific to ``X`` (small signal to noise ratio). [Image source](https://yang-song.net/blog/2021/score/).
"""

# ╔═╡ 06e194ec-42dc-46aa-8935-86c4ed04b7c8
img("accurate.jpg")

# ╔═╡ a7435857-1526-4155-b934-4c4cda9f6b30
frametitle("Variance-dependent score")

# ╔═╡ c3af23f7-8919-4e27-a97e-6013038b8d8c
HAlign(md"""
The matching is ``\epsilon_\theta(y\textcolor{red}{, \sigma}) \approx -\sigma \nabla_y \log f_{X+\sigma\mathcal{E}}(y)``.

**Training**: sample ``x`` / pick ``x`` in dataset, sample ``\textcolor{red}{\sigma, }\varepsilon``, update ``\theta`` to minimize (e.g., using gradient descent), the loss:
```math
\mathbb{E}[\|\epsilon_\theta(X + \sigma \mathcal{E} \textcolor{red}{, \sigma}) - \mathcal{E}\|^2]
```

Animation generated with [`smalldiffusion`](https://github.com/yuanchenyang/smalldiffusion) using a deterministic (i.e. ``w_k=0``) sampler.
""",
img("spiral.gif"),
)

# ╔═╡ 784608dc-cf8f-483b-964f-b66575dc430e
frametitle("Sampling")

# ╔═╡ 87f3dafc-6452-4a3f-b635-9b7497e1481a
img("DDIM")

# ╔═╡ c3574e26-6e50-4f08-852a-df3ae91c197f
frametitle("Denoising with randomness")

# ╔═╡ 2cc866e8-fb23-4010-bec1-29d00202c7e1
qa(md"What is the order between ``\sigma_{t-1}, \sigma_t`` and ``\sigma_{t'}`` ?",
md"""
We have ``\sigma_{t-1} < \sigma_t`` by assumption.
Then, because ``\sigma_{t-1}`` is the geometric mean of ``\sigma_{t'}`` and ``\sigma_t``, it must be between so
``\sigma_{t'} < \sigma_{t-1} < \sigma_t``.
""")

# ╔═╡ e50cffb2-ac51-483a-92d0-a19531ebb547
qa(md"What should ``\eta`` be ?",
md"""
We want the noise level of ``x_{t-1}`` to be at ``\sigma_{t-1}``.
So we want
```math
\sigma_{t-1}^2 = \text{Var}(\sigma_{t'} \mathcal{E}_\theta(x_t, \sigma_t) + \eta W_t)
```
As ``W_t`` and ``\mathcal{E}_\theta(x_t, \sigma_t)`` are independent, their covariance is zero so
```math
\sigma_{t-1}^2 = \sigma_{t'}^2 \text{Var}(\mathcal{E}_\theta(x_t, \sigma_t)) + \eta^2 \text{Var}(W_t)
```
Their variance is one, so we have
```math
\sigma_{t-1}^2 = \sigma_{t'}^2 + \eta^2
```
This means that we should take ``\eta = \sqrt{\sigma_{t-1}^2 - \sigma_{t'}^2}``.
""")

# ╔═╡ a9e46466-cdbf-4539-ae36-a3113634b353
frametitle("Acceleration")

# ╔═╡ d3792580-6a04-4934-ad5a-cd054a5b4421
frametitle("Auto-Encoder")

# ╔═╡ de5ddee9-7a4c-4ae7-b6e2-ccdf1a406919
md"""
Find encoder ``E`` and decoder ``D`` that minimize the loss:
```math
\mathbb{E}[\|X - D(E(X))\|_2^2]
```
The *code* (aka *latent variable*) ``z = E(x)`` typically has a smaller size compared to ``x`` to force the model to only keep essential features.
"""

# ╔═╡ 40f21927-1d88-41f4-b55d-6b81ca2ec211
qa(md"What is the solution if ``E`` and ``D`` were linear (i.e. matrices) ?",
md"""
Given a matrix of data ``X``, consider the SVD ``X = U\Sigma V^\top``:
```math
\begin{align}
\|X - DEX\|_F^2
& =
\|U \Sigma V^\top - DEU \Sigma V^\top\|_F^2\\
& =
\|\Sigma - U^\top DEU \Sigma\|_F^2\\
\end{align}
```
The matrix ``DE`` can be any matrix of rank equal to the dimension of the latent space.
So the optimal solution is ``DE = U\text{Diag}(\mathbf{1}_r, \mathbf{0}_{n-r})U^\top``
where ``r`` is the dimension of the latent space.

This is similar to the PCA ``D\hat{E} = U\text{Diag}(\sigma_1, \ldots, \sigma_r, \mathbf{0}_{n-r})V^\top`` which is the minimizer of ``\|X - D\hat{E}\|_F^2``.
More precisely, given the optimal rank-``r`` solution ``D = U_{:,1:r}, \hat{E} = \text{Diag}(\sigma_1, \ldots, \sigma_r)U_{:,1:r}^\top``,
an optimal solution of the linear auto-encoder can be obtained with ``E = \hat{E}X^\dagger`` when ``X^\dagger = V\Sigma^{-1}U^\top``.

In view of this, an Auto-Encoder can be thought of as a nonlinear generalization of PCA.
"""
)

# ╔═╡ f50e757b-2f0d-458c-a577-3de499d9ba14
md"""
## Variational Auto-Encoder

We add artificial noise to improve learning:
```math
\mathbb{E}[\|X - D(E_\mu(X) + \mathcal{E} \odot E_\sigma(X))\|_2^2]
```
An insightful way to model this is as follows.
* Assume our dataset come from a distribution ``X`` that is obtained from a latent space ``Z`` using a decoder ``X = D(Z)``.
* If we knew the latent variable ``z``, we would search for a decoder ``D`` that maximizes ``\mathbb{E}[\log(f_{X|Z}(x|z))]``.
* Since we do not know ``z``, we make an estimate of ``y`` using an encoder ``Y = E(X)``.
* The validity of these encoder and decoder models can now be measured with a Maximum Likelihood Estimator that searches for the encoder and decoder models that maximize ``\mathbb{E}[\log(f_X(X))]``.
"""

# ╔═╡ 6d63c708-6f10-4a21-b57b-a9bd138b5c8c
frametitle("Evidence Lower BOund (ELBO)")

# ╔═╡ 8e39960f-ab0a-4103-bd6b-00e779333b01
qa(md"Proof",
md"""
> Reminder: ``D_\text{KL}(A \parallel B) = \sum_{a \in \text{Dom}(A)} f_A(a) \log \frac{f_A(a)}{f_B(a)} = \mathbb{E}[\log f_A(A) - \log f_B(A)]``

```math
\begin{align}
  D_\text{KL}((Y|X = x) \parallel (Z | X = x))
  & =
  \mathbb{E}[\log(f_{Y|X}(Y|x)) - \log(f_{Z|X}(Y|x)) | X = x]\\
  & =
  \mathbb{E}[\log(f_{Y|X}(Y|x)) - \log(f_{Z,X}(Y,x)) | X = x]\\
  & \qquad + \log(f_{X}(x))\\
  & =
  \mathbb{E}[\log(f_{Y|X}(Y|x)) - \log(f_{X|Z}(x|Y)) - \log(f_{Z}(Y)) | X = x]\\
  & \qquad + \log(f_{X}(x))\\
  & =
  D_\text{KL}((Y|X = x) \parallel Z) - \mathbb{E}[\log(f_{X|Z}(x|Y)) | X = x]\\
  & \qquad + \log(f_{X}(x))
\end{align}
```
""",
)

# ╔═╡ d821d85c-74ae-44d0-909b-02d3099d8201
md"""
``\mathcal{L}(x)`` is a **lower bound** to ``\log(f_X(x))`` as the Kullback-Leibler divergence ``D_\text{KL}((Y|X = x) \parallel (Z | X = x))`` is always nonnegative.

In the context of VAEs, ``Z`` represents the actual latent variable and ``Y`` represents the estimated random variables, so ``D_\text{KL}((Y|X = x) \parallel (Z | X = x))`` is a measure of the error made by our estimator.
"""

# ╔═╡ 0788ac02-dab8-4ddd-9130-2a632d8d3685
frametitle("Gaussian ELBO")

# ╔═╡ 76eaa13b-e76a-4e31-b702-78fd9694a059
frametitle("Monte-Carlo sampling")

# ╔═╡ 228d70f8-dd36-48a6-9c95-b2744c5b20b6
md"""
This can be approximated using Monte-Carlo given ``L`` samples ``\epsilon_1, \ldots \epsilon_L`` from the distribution ``\mathcal{N}(0, I)`` as
```math
\mathbb{E}[\log(f_{X|Z}(x|Y))] \approx \frac{1}{L} \sum_{i=1}^L \log(f_{X|Z}(x|E_\mu(x) + \epsilon_i \odot E_\sigma(x))).
```
In the simpler case where ``D_\sigma(z) = \mathbf{1}``, we recognize the classical L2 norm:
```math
\begin{align}
\mathbb{E}[\log(f_{X|Z}(x|Y))]
& \approx -\frac{\log(2\pi)}{2}+\frac{1}{L}\sum_{i=1}^L\|x - D_\mu(E_\mu(x) + \epsilon_i \odot E_\sigma(x))\|_2^2.
\end{align}
```
"""

# ╔═╡ 8f51a619-d0ad-42d2-bac9-acb9eab7cccb
frametitle("Variational AutoEncoders (VAEs)")

# ╔═╡ 23f3de75-0617-4232-bb71-bd9f3e355a1e
md"""
* We want to learn the distribution of our data represented by the random variable ``X``.
* The encoder maps a data point ``x`` to a Gaussian distribution ``Y \sim \mathcal{N}(E_\mu(x), E_{\sigma}(x))`` over the latent space
* The decoder maps a latent variable ``z \sim Z`` to a the Gaussian distribution ``\mathcal{N}(D_\mu(z), D_\sigma(z))``

The Maximum Likelihood Estimator (MLE) maximizes the following sum over our datapoints ``x`` with its ELBO:
```math
\sum_x \log(f_X(x)) \ge \sum_x -D_\text{KL}((Y|X = x) \parallel Z) + \mathbb{E}[\log(f_{X|Z}(x|Y))]
```
So the MLE minimizes the loss
```math
-\mathbb{E}[\log(f_{X|Z}(x|Y))].
```
with the KL-regularizer
```math
D_\text{KL}((Y|X = x) \parallel Z)
```
"""

# ╔═╡ ca8b4f75-f8c7-4384-80ae-fbbdba85bd46
frametitle("Denoising Auto-Encoder")

# ╔═╡ ca68ba1d-5f02-4f80-8cd1-d0c297f19f58
frametitle("Conditioned diffusion")

# ╔═╡ b94dd60a-2116-43ce-a5c3-5f1ff24479c0
img("stable_diffusion")

# ╔═╡ 530110cc-1dc9-427e-950b-64c0d04fc0da
frametitle("Classifier-Free Guidance")

# ╔═╡ 97d6e0da-0b53-4467-95f3-a524575daa49
img("guidance")

# ╔═╡ cddf4399-c659-4bbd-bae1-6d88f37a9b93
frametitle("Optical illusions")

# ╔═╡ 2a0f227f-28b4-419e-9f82-f8bd13fca8c0
section("Utils")

# ╔═╡ 4b272be9-6881-41e8-9168-3fe471527aa8
import DocumenterCitations, CSV, Logging

# ╔═╡ cd8567a7-f4fe-4592-ae4c-facdc35db563
biblio = load_biblio!()

# ╔═╡ d077a471-6b43-41df-8218-e3b4a7e38550
cite(args...) = bibcite(biblio, args...)

# ╔═╡ 6fd4e448-992f-4976-82e9-6313ba038d26
md"""
Image from $(cite("permenter2024Interpretinga", "Figure 1")). Deterministic DDIM : move from a variance estimate ``\sigma_t`` to ``\sigma_{t-1}``:
```math
\begin{align}
X_t & = X_0 + \sigma_t \mathcal{E}\\
\mathbb{E}[X_{t-1}|X_t = x_{t}]
& =
\mathbb{E}[X_0|X_t = x_t] + \sigma_{t-1} \mathbb{E}[\mathcal{E}|X_t = x_t]\\
& =
\mathbb{E}[X_t|X_t = x_t] - \sigma_t \mathbb{E}[\mathcal{E}|X_t = x_t] + \sigma_{t-1} \mathbb{E}[\mathcal{E}|X_t = x_t]\\
& =
x_t + (\sigma_{t-1} - \sigma_t) \mathbb{E}[\mathcal{E}|X_t = x_t]\\
\end{align}
```
"""

# ╔═╡ 45382f1f-89d7-4972-96df-1d9288a52102
md"""
Given ``0 \le \mu < 1``, pick ``\sigma_{t'}`` such that ``\sigma_{t-1} = \sigma_t^\mu\sigma_{t'}^{1 - \mu}`` we have the following sampler of $(cite(["permenter2024Interpretinga"])):
```math
\begin{align}
x_{t - 1} & = x_t + (\textcolor{purple}{\sigma_{t'}} - \sigma_t) \epsilon_\theta(x_t, \sigma_t) \textcolor{purple}{+ \eta w_t} \qquad w_t \sim \mathcal{N}(0, I)
\end{align}
```
For ``\mu = 0``, ``\sigma_{t'} = \sigma_{t-1}`` and we recover deterministic DDIM.
For ``\mu = 1/2``, we have DDPM of $(cite("ho2020Denoising"))
"""

# ╔═╡ 78dd39e3-d549-4012-92d1-3a1c8033ac29
HAlign(
md"""
As ``\epsilon_\theta(x_t, \sigma_t)`` is more accurate than
``\epsilon_\theta(x_{t+1}, \sigma_{t+1})``,
$(cite("permenter2024Interpretinga", "Section 5"))
suggests accelerating the convergence by correcting part of
the previous step. That is, we take:
```math
\bar{\epsilon}_t = \textcolor{purple}{\gamma} \epsilon_\theta(x_t, \sigma_t) \textcolor{purple}{+ (1 - \gamma) \epsilon_\theta(x_{t+1}, \sigma_{t+1})}
```
with ``\gamma > 1``.

Image is $(cite("permenter2024Interpretinga", "Figure 3")).
""",
img("ge_step")
)

# ╔═╡ 0e01c57f-65b1-480c-a258-9a76513ac8d6
md"""
For any random variables ``X``, ``Y`` and ``Z``, we have
```math
\log(f_X(x)) = D_\text{KL}((Y|X = x) \parallel (Z | X = x)) + \mathcal{L}(x)
```
where the *evidence lower bound* $(cite("kingma2013AutoEncoding"))
```math
\mathcal{L}(x) = -D_\text{KL}((Y|X = x) \parallel Z) + \mathbb{E}[\log(f_{X|Z}(x|Y)) | X = x]
```
"""

# ╔═╡ 76643525-2649-48b9-9d09-6d7dcef6e355
md"""
Consider two deterministic functions ``E_\mu, E_\sigma : \mathbb{R}^n \to \mathbb{R}^r`` and ``D_\mu, D_\sigma : \mathbb{R}^r \to \mathbb{R}^n``.
Suppose ``X = D_\mu(Z) + \mathcal{E}_1 \odot D_\sigma(Z)`` and ``Y = E_\mu(X) + \mathcal{E}_2 \odot E_\sigma(X)`` with ``\mathcal{E}_1, \mathcal{E}_2 \sim \mathcal{N}(0, I)``.
We have (see $(cite("kingma2013AutoEncoding", "Appendix B")) for a proof):
```math
2D_\text{KL}((Y|X = x) \parallel Z) = \|E_\mu(X)\|_2^2 + \|E_\sigma(X)\|_2^2 - r - \sum_{i=1}^r \log((E_\sigma(X))_i^2)
```
For the second part of the ELBO, we have
```math
\begin{align}
& \mathbb{E}[\log(f_{X|Z}(x|Y))]\\
& = \mathbb{E}[\log(f_{X|Z}(x|E_\mu(x) + \mathcal{E}_2 \odot E_\sigma(x)))]\\
& = \mathbb{E}[\log(f_{\mathcal{E}_1}(\text{Diag}(D_\sigma(E_\mu(x) + \mathcal{E}_2 \odot E_\sigma(x)))^{-1} (x - D_\mu(E_\mu(x) + \mathcal{E}_2 \odot E_\sigma(x)))))]\\
& = -\frac{\log(2\pi)}{2}+\mathbb{E}[\|\text{Diag}(D_\sigma(E_\mu(x) + \mathcal{E}_2 \odot E_\sigma(x)))^{-1} (x - D_\mu(E_\mu(x) + \mathcal{E}_2 \odot E_\sigma(x)))\|_2^2].
\end{align}
```
"""

# ╔═╡ 1a1335db-6bc5-4a0e-a138-3513f7a2140d
md"""
```math
\begin{align}
\text{Auto-Encoder} & & D(E(X))\\
\text{Variational Auto-Encoder} & & D(E(X) + \mathcal{E})\\
\text{Denoising Auto-Encoder} & & D(E(X + \sigma \mathcal{E}))\\
\end{align}
```
The goal of the diffusion model is, given a noisy image ``Y = X + \sigma \mathcal{E}`` and ``\sigma``, to find the noise ``\mathcal{E}``.
The denoising Auto-Encoder instead attempts to find the original image ``X``.
At the limit ``\sigma \to 0``, the denoising Auto-Encoder ``D(E(X + \sigma \mathcal{E}))`` is a classical Auto-Encoder, hence the name.

To train the denoising Auto-Encoder, we can use the Evidence Lower-Bound with ``Y = X + \sigma \mathcal{E}`` and ``Z`` such that ``X = D(Z)`` $(cite("ho2020Denoising")):
```math
-\log(f_X(x)) \le \mathbb{E}[-\log(f_{X|Z}(x|Y))] + D((Y|X = x) \parallel Z)
```
"""

# ╔═╡ e0ba3b61-a618-49ce-9466-e6a8f674bd53
md"""Source: $(cite("rombach2022HighResolution", "Figure 3"))"""

# ╔═╡ 410eac96-a384-48fa-b3f3-478be5bd236c
md"""
Improvement for conditioned diffusion for multi-modal distributions: use classifier $(cite("dhariwal2024Diffusion")). **Issue**: need to train a classifier...

This classifier-guided strategy was replaced in $(cite("ho2022ClassifierFree"))
by a simpler trick: Train both a conditioned (with condition ``\tau``) and an unconditioned diffusion model, and combine them with:
```math
\bar{\epsilon}_t = \textcolor{purple}{\lambda} \epsilon_\theta(x_t, \sigma_t, \tau) \textcolor{purple}{+ (1 - \lambda) \epsilon_\theta(x_t, \sigma_t)} \qquad \text{with } \lambda > 1
```
"""

# ╔═╡ 9cd90023-cf5d-4291-ae69-bc902c5c41be
HAlign(
md"""
It was shown in $(cite("burgert2023Diffusion")) how to train a diffusion model to generate illusions.
Suprisingly, $(cite("geng2024Visual")) showed that you don't need a specialized model and you can use a pre-trained diffusion model.
Given ``N`` transformations ``v_1, \ldots, v_N`` $(cite("geng2024Visual", "Equation (2)")):
```math
\bar{\epsilon}_t = \textcolor{purple}{\frac{1}{N} \sum_{i=1}^N v_i^{-1}(}\epsilon_\theta(\textcolor{purple}{v_i(}x_t\textcolor{purple}{)}, \sigma_t, \tau)\textcolor{purple}{)}
```
""",
img("flips")
)

# ╔═╡ c1fe952e-c074-49c1-91a4-0de76d235f25
refs(args...) = bibrefs(biblio, args...)

# ╔═╡ 2a728c9a-4268-40ff-905d-60b3b4c5b99b
refs(["ho2020Denoising", "permenter2024Interpretinga"])

# ╔═╡ fbb03a1a-99ef-4d98-8ba4-10d3cc026abc
refs("permenter2024Interpretinga")

# ╔═╡ c64fc228-d1e3-423f-8fb3-b8e7dff9d097
refs("kingma2013AutoEncoding")

# ╔═╡ 0708fa36-bf99-44e3-a69e-991ad521d34b
refs("kingma2013AutoEncoding")

# ╔═╡ 53b3d56e-f11d-44c2-96ca-3714f4b4f929
refs("ho2020Denoising")

# ╔═╡ e728cc9a-c17f-4143-9e0e-12c03a965bec
refs("rombach2022HighResolution")

# ╔═╡ 8cf52c38-a9a6-4614-a80b-b36e9d4e563f
refs(["dhariwal2024Diffusion", "ho2022ClassifierFree"])

# ╔═╡ 09829600-3488-492a-a507-53a4bbcb1cfc
refs(["burgert2023Diffusion", "geng2024Visual"])

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DocumenterCitations = "daee34ce-89f3-4625-b898-19384cb65244"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
Luxor = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
MathTeXEngine = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"

[compat]
CSV = "~1.0.0"
DataFrames = "~1.8.2"
DocumenterCitations = "~1.5.0"
LaTeXStrings = "~1.4.1"
Luxor = "~4.5.0"
MathTeXEngine = "~0.6.9"
PlutoUI = "~0.7.83"
PrettyTables = "~3.4.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.0"
manifest_format = "2.1"
project_hash = "4685174dc6d8943fe2cc87e422ca8f4a5c306fde"

[[deps.ANSIColoredPrinters]]
git-tree-sha1 = "574baf8110975760d391c710b6341da1afa48d8c"
registries = "General"
uuid = "a4c015fc-c6ff-483c-b24f-f7ea428134e9"
version = "0.0.1"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
registries = "General"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "94eab0b3ccdcac361188cc661daf69d4433c1818"
registries = "General"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.2.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "8c290a1b223deaeea9aea44b235d24546da8eb98"
registries = "General"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.4.0"

[[deps.BibInternal]]
deps = ["TestItems"]
git-tree-sha1 = "ec1f1c78c1a6c917125abdee98518685fe653e64"
registries = "General"
uuid = "2027ae74-3657-4b95-ae00-e2f7d55c3e64"
version = "0.4.0"

[[deps.BibParser]]
deps = ["BibInternal", "DataStructures", "TestItems"]
git-tree-sha1 = "5a6aeb593475157a5911cda3e0297f1552d45102"
registries = "General"
uuid = "13533e5b-e1c2-4e57-8cef-cac5e52f6474"
version = "0.3.0"

    [deps.BibParser.extensions]
    BibParserCFFExt = ["Dates", "JSONSchema", "YAML"]
    BibParserCSLExt = "JSON3"
    BibParserXMLExt = "EzXML"

    [deps.BibParser.weakdeps]
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    EzXML = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
    JSONSchema = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
    YAML = "ddb6d928-2868-570f-bddf-ab3f9cf99eb6"

[[deps.Bibliography]]
deps = ["BibInternal", "BibParser", "DataStructures", "Dates", "FileIO", "JSONSchema", "TestItems", "YAML"]
git-tree-sha1 = "2246611f8da01251b73665fe0d57ed9d0e3b880b"
registries = "General"
uuid = "f1be7e48-bf82-45af-a471-ae754a193061"
version = "0.4.0"

    [deps.Bibliography.extensions]
    BibliographyCSLExt = "JSON3"
    BibliographyXMLExt = "EzXML"

    [deps.Bibliography.weakdeps]
    EzXML = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.Bijections]]
git-tree-sha1 = "a2d308fcd4c2fb90e943cf9cd2fbfa9c32b69733"
registries = "General"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.2.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
registries = "General"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CSV]]
deps = ["CodecZlib", "DataStrings", "Dates", "Downloads", "Durations", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "Printf", "Tables", "Unicode"]
git-tree-sha1 = "b5c1d7ec1595d9cca3f8b7b8879d3a372e6972be"
registries = "General"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "1.0.0"

    [deps.CSV.extensions]
    CSVDataDecimalsExt = "DataDecimals"
    CSVFilePathsBaseExt = "FilePathsBase"
    CSVInlineStringsExt = "InlineStrings"

    [deps.CSV.weakdeps]
    DataDecimals = "3e2245cb-6932-498d-a7cf-39d39de8bde3"
    FilePathsBase = "48062228-2e41-5def-b9a4-89aafe57970f"
    InlineStrings = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
registries = "General"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "970758a3d591a2a5c2a907c53f2e2f8c1b1d3537"
registries = "General"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.9"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
registries = "General"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
registries = "General"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStrings]]
git-tree-sha1 = "b8032263d364177a0b192ea15f277df4e6ab07e2"
registries = "General"
uuid = "48204bd6-5611-42ea-b167-fc1d713f9be2"
version = "1.1.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Documenter]]
deps = ["ANSIColoredPrinters", "AbstractTrees", "Base64", "CodecZlib", "Dates", "DocStringExtensions", "Downloads", "Git", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "MarkdownAST", "Pkg", "PrecompileTools", "REPL", "RegistryInstances", "SHA", "TOML", "Test", "Unicode"]
git-tree-sha1 = "191e6bef0cf32cac3a3913cd4787d851715254c2"
registries = "General"
uuid = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
version = "1.19.0"

[[deps.DocumenterCitations]]
deps = ["AbstractTrees", "Bibliography", "Bijections", "Dates", "Documenter", "Logging", "Markdown", "MarkdownAST", "OrderedCollections", "SHA", "Unicode"]
git-tree-sha1 = "5699464c66535cb99ad6ab075513cb215217b9f2"
registries = "General"
uuid = "daee34ce-89f3-4625-b898-19384cb65244"
version = "1.5.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.Durations]]
deps = ["Dates"]
git-tree-sha1 = "a2f9c974a78a23661f0e41d0b80b687f844c018e"
registries = "General"
uuid = "41ac09f5-0a95-4810-aa18-c88bf48ee130"
version = "1.2.0"

    [deps.Durations.extensions]
    DurationsArrowExt = "Arrow"

    [deps.Durations.weakdeps]
    Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6123b906924c935459624131058fab6a587fc6c1"
registries = "General"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "3.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
registries = "General"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6621fef488e496356c9c9625d0562c12a6070819"
registries = "General"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.20.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
registries = "General"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
registries = "General"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
registries = "General"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Git]]
deps = ["Git_LFS_jll", "Git_jll", "JLLWrappers", "OpenSSH_jll"]
git-tree-sha1 = "824a1890086880696fc908fe12a17bcf61738bd8"
registries = "General"
uuid = "d7ba0133-e1db-5d97-8f8c-041e4b3a1eb2"
version = "1.5.0"

[[deps.Git_LFS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8c66e385d631bb934ff05e76d4a566c640c8df69"
registries = "General"
uuid = "020c3dae-16b3-5ae5-87b3-4cb189e250b2"
version = "3.7.1+0"

[[deps.Git_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "Libiconv_jll", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "7b16700f9e313c0d972d1222ede50a2076aa0770"
registries = "General"
uuid = "f8c6e375-362e-5223-8a59-34ff63f689eb"
version = "2.55.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
registries = "General"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "9d9531a9cb63a9edc33836414e82a07e81710de2"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14004.0+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "06b65886c7577a3784d616e29f1302c2e36e389d"
registries = "General"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.6"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
registries = "General"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
registries = "General"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "88352712893ec50bee3680605891eaf0e9ed6368"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.8.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JSONSchema]]
deps = ["Downloads", "JSON", "URIs"]
git-tree-sha1 = "d13f79c4242969874da7d00bda17d59bc7699aa7"
registries = "General"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "1.5.0"

    [deps.JSONSchema.extensions]
    JSONSchemaJSON3Ext = "JSON3"

    [deps.JSONSchema.weakdeps]
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "39bca05343661c347aae0bca57a5994a0bf4f08d"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.2.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.LazilyInitializedFields]]
git-tree-sha1 = "0f2da712350b020bc3957f269c9caad516383ee0"
registries = "General"
uuid = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.18.0+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.1+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.103+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Librsvg_jll]]
deps = ["Artifacts", "Cairo_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pango_jll", "XML2_jll", "gdk_pixbuf_jll"]
git-tree-sha1 = "e6ab5dda9916d7041356371c53cdc00b39841c31"
registries = "General"
uuid = "925c91fb-5dd6-59dd-8e8c-345e74382d89"
version = "2.54.7+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.Luxor]]
deps = ["Base64", "Cairo", "Colors", "DataStructures", "Dates", "FFMPEG", "FileIO", "PolygonAlgorithms", "PrecompileTools", "Random", "Rsvg"]
git-tree-sha1 = "fe8060b3d693f682e14f1019b058c64effb62b43"
registries = "General"
uuid = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
version = "4.5.0"

    [deps.Luxor.extensions]
    LuxorExtLatex = ["LaTeXStrings", "MathTeXEngine"]
    LuxorExtTypstry = ["Typstry"]

    [deps.Luxor.weakdeps]
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    MathTeXEngine = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MarkdownAST]]
deps = ["AbstractTrees", "Markdown"]
git-tree-sha1 = "93c718d892e73931841089cdc0e982d6dd9cc87b"
registries = "General"
uuid = "d0879d2d-cac2-40c8-9cee-1863dc0c7391"
version = "0.1.3"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "aa1078778be5a8e5259ff04fbc3d258b3e78d464"
registries = "General"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.9"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.8.13"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSH_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "OpenSSL_jll", "Zlib_jll"]
git-tree-sha1 = "2da18ab26a6eb38b374c16b157d5a4cc3ab74e02"
registries = "General"
uuid = "9bd350c2-7e96-507f-8002-3f2e150b4e1b"
version = "10.5.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05f45c2e0de6259db764adbfd2f1dc6d3f8de13c"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.46.0+0"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PolygonAlgorithms]]
git-tree-sha1 = "c1092ada65e6d59d6361d5086ddb0a5ea63ae204"
registries = "General"
uuid = "32a0d02f-32d9-4438-b5ed-3a2932b48f96"
version = "0.4.0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
registries = "General"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "5005266de4bfe50e53ff44a5cb5c540b6e47a254"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1b8aa19f229b1cea7fc93874a52e49db6a854450"
registries = "General"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.4.8"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegistryInstances]]
deps = ["LazilyInitializedFields", "Pkg", "TOML", "Tar"]
git-tree-sha1 = "ffd19052caf598b8653b99404058fce14828be51"
registries = "General"
uuid = "2792f1a3-b283-48e8-9a74-f99dce5104f3"
version = "0.1.0"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rsvg]]
deps = ["Cairo", "Glib_jll", "Librsvg_jll"]
git-tree-sha1 = "e53dad0507631c0b8d5d946d93458cbabd0f05d7"
registries = "General"
uuid = "c4c386cf-5103-5370-be45-f3a111cca3b8"
version = "1.1.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
registries = "General"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "e206cf4850fd7ac4255ffd2b98922f563e18ac53"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.20"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "b765e46ba27ecf6b44faf70df40c57aa3a547dcb"
registries = "General"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "773065c6e0e903924a9d838259be74338422aef2"
registries = "General"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.5.0"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "9537ef82c42cdd8c5d443cbc359110cbb36bae10"
registries = "General"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.21"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "2d0fc55c61321ba245c47be599570d11bac50303"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.5"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TestItems]]
git-tree-sha1 = "a6dd904babc04670c784f81b67957a3545271610"
registries = "General"
uuid = "1c621080-faea-4a02-84b6-bbd5e436b8fe"
version = "1.1.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
registries = "General"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
registries = "General"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.YAML]]
deps = ["Base64", "Dates", "Printf", "StringEncodings"]
git-tree-sha1 = "a1c0c7585346251353cddede21f180b96388c403"
registries = "General"
uuid = "ddb6d928-2868-570f-bddf-ab3f9cf99eb6"
version = "0.4.16"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.gdk_pixbuf_jll]]
deps = ["Artifacts", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Xorg_libX11_jll", "libpng_jll"]
git-tree-sha1 = "895f21b699121d1a57ecac57e65a852caf569254"
registries = "General"
uuid = "da03df04-f53b-5353-a52f-6a8b0620ced0"
version = "2.42.13+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ef17c47d22224aaecc76e597ab21a072e025cf7b"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.14.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.67.1+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.2+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─7d05bc54-a693-11ef-2d83-a5caeb91676e
# ╟─44bb88d2-2fc7-4901-9fc3-890e54754932
# ╟─423874b8-6d4c-423a-bf45-39d590600066
# ╟─abc07fcf-35f5-43eb-90aa-23ec0b581e56
# ╟─df9e106b-16c3-4605-b5f6-0a4e82dd4b3b
# ╟─eaf0bcaa-75d9-4d9c-9e19-ca5a8532ed35
# ╟─687f5cb9-5a17-49cb-8d96-a35b66fbcba3
# ╟─3cdcd14a-9102-4ed4-b186-629ce183b73c
# ╟─a835f101-bfd0-428e-adba-64f5de7af14c
# ╟─0886ac78-6530-417b-80a5-5410c4719dd2
# ╟─d826a144-2c79-4c1e-ae3f-c11a4b5353d4
# ╟─7e7e2845-4bee-4dbc-b0f9-d964cd362248
# ╟─e33b16ec-4526-4dbe-affa-3e68aa508ab8
# ╟─06e194ec-42dc-46aa-8935-86c4ed04b7c8
# ╟─a7435857-1526-4155-b934-4c4cda9f6b30
# ╟─c3af23f7-8919-4e27-a97e-6013038b8d8c
# ╟─784608dc-cf8f-483b-964f-b66575dc430e
# ╟─87f3dafc-6452-4a3f-b635-9b7497e1481a
# ╟─6fd4e448-992f-4976-82e9-6313ba038d26
# ╟─c3574e26-6e50-4f08-852a-df3ae91c197f
# ╟─45382f1f-89d7-4972-96df-1d9288a52102
# ╟─2cc866e8-fb23-4010-bec1-29d00202c7e1
# ╟─e50cffb2-ac51-483a-92d0-a19531ebb547
# ╟─2a728c9a-4268-40ff-905d-60b3b4c5b99b
# ╟─a9e46466-cdbf-4539-ae36-a3113634b353
# ╟─78dd39e3-d549-4012-92d1-3a1c8033ac29
# ╟─fbb03a1a-99ef-4d98-8ba4-10d3cc026abc
# ╟─d3792580-6a04-4934-ad5a-cd054a5b4421
# ╟─de5ddee9-7a4c-4ae7-b6e2-ccdf1a406919
# ╟─40f21927-1d88-41f4-b55d-6b81ca2ec211
# ╟─f50e757b-2f0d-458c-a577-3de499d9ba14
# ╟─6d63c708-6f10-4a21-b57b-a9bd138b5c8c
# ╟─0e01c57f-65b1-480c-a258-9a76513ac8d6
# ╟─8e39960f-ab0a-4103-bd6b-00e779333b01
# ╟─d821d85c-74ae-44d0-909b-02d3099d8201
# ╟─c64fc228-d1e3-423f-8fb3-b8e7dff9d097
# ╟─0788ac02-dab8-4ddd-9130-2a632d8d3685
# ╟─76643525-2649-48b9-9d09-6d7dcef6e355
# ╟─0708fa36-bf99-44e3-a69e-991ad521d34b
# ╟─76eaa13b-e76a-4e31-b702-78fd9694a059
# ╟─228d70f8-dd36-48a6-9c95-b2744c5b20b6
# ╟─8f51a619-d0ad-42d2-bac9-acb9eab7cccb
# ╟─23f3de75-0617-4232-bb71-bd9f3e355a1e
# ╟─ca8b4f75-f8c7-4384-80ae-fbbdba85bd46
# ╟─1a1335db-6bc5-4a0e-a138-3513f7a2140d
# ╟─53b3d56e-f11d-44c2-96ca-3714f4b4f929
# ╟─ca68ba1d-5f02-4f80-8cd1-d0c297f19f58
# ╟─b94dd60a-2116-43ce-a5c3-5f1ff24479c0
# ╟─e0ba3b61-a618-49ce-9466-e6a8f674bd53
# ╟─e728cc9a-c17f-4143-9e0e-12c03a965bec
# ╟─530110cc-1dc9-427e-950b-64c0d04fc0da
# ╟─410eac96-a384-48fa-b3f3-478be5bd236c
# ╟─97d6e0da-0b53-4467-95f3-a524575daa49
# ╟─8cf52c38-a9a6-4614-a80b-b36e9d4e563f
# ╟─cddf4399-c659-4bbd-bae1-6d88f37a9b93
# ╟─9cd90023-cf5d-4291-ae69-bc902c5c41be
# ╟─09829600-3488-492a-a507-53a4bbcb1cfc
# ╟─2a0f227f-28b4-419e-9f82-f8bd13fca8c0
# ╠═3fde5f41-32a9-4585-9b6a-131173947346
# ╠═4b272be9-6881-41e8-9168-3fe471527aa8
# ╠═19741723-3cc3-40df-83a3-fb2727504cd1
# ╠═cd8567a7-f4fe-4592-ae4c-facdc35db563
# ╠═d077a471-6b43-41df-8218-e3b4a7e38550
# ╠═c1fe952e-c074-49c1-91a4-0de76d235f25
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
