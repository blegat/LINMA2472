### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ f1ba3d3c-d0a5-4290-ab73-9ce34bd5e5f6
using PlutoUI, PlutoUI.ExperimentalLayout, HypertextLiteral, PlutoTeachingTools

# ╔═╡ 64250438-9ba6-44f7-b4a0-fcd5d72f05e5
using LinearAlgebra, RowEchelon

# ╔═╡ 40baa108-eb68-433f-9917-ac334334f198
@htl("""
<p align=center style=\"font-size: 40px;\">Implicit Differentiation</p><p align=right><i>Benoît Legat</i></p>
$(PlutoTeachingTools.ChooseDisplayMode())
$(PlutoUI.TableOfContents(depth=1))
""")

# ╔═╡ d6797e7b-fa78-4366-b88f-9925f72a5d0f
md"""
# Differentiating numerical procedures

Consider a program that numerically converge to solutions:
```julia
function f(x)
	while abs(error) > tol
    	y += # Make some step in the right direction
    	error = # update error
	end
	return y
end
```
"""

# ╔═╡ d907c5d7-f517-467a-a840-f94868ad7680
md"""
## Square root example

To illustrate consider [this Jax example](https://docs.jax.dev/en/latest/advanced-autodiff.html#example-implicit-function-differentiation-of-iterative-implementations) rewriting ``x^2 = a`` into ``2x^2 = x^2 + a`` or ``x = (x + a/x)/2``
"""

# ╔═╡ 5c39f536-346c-48bd-a3c7-acaefb849b9f
function fixed_point(f, x)
    fx = f(x)
	while abs(x - fx) > 1e-6
		x = fx
		fx = f(x)
	end
	return x
end

# ╔═╡ a73e6557-9fed-4a1e-9e8e-c67e500eba5b
my_sqrt(a) = fixed_point(x -> (x + a / x) / 2, a)

# ╔═╡ abee47e4-1b75-4466-908a-e14113b181ab
my_sqrt(2)

# ╔═╡ 23049939-1b3a-4328-b159-99c3f1470e39
md"## AD through fixed point"

# ╔═╡ c9603b16-113b-4e05-8517-d814b0879ab8
md"""
# Implicit function theorem

> *In a sense, the implicit function theorem can be thought as the mother theorem, as it can be used to prove envelope theorems, the adjoint state method and the inverse function theorem.* Section 11.6 of [The Elements of Differentiable Programming book](https://diffprog.github.io/)
"""

# ╔═╡ b23a4c1d-cf35-42e6-b286-7fbcdddfd424
md"""
## Inverse function theorem

Assume
* ``f : \mathcal{W} \to \mathcal{W}`` is ``C^2``
* ``\partial f(w_0)`` is **invertible**

Then
* ``f`` is bijective from a neighborhood of ``w_0`` to a neighborhood of ``f(w_0)``
* For ``\omega`` in a neighborhood of ``f(w_0)``, ``f^{-1}`` is ``C^2`` and ``\partial f^{-1}(\omega) = (\partial f(f^{-1}(w)))^{-1}``
"""

# ╔═╡ 0b091706-da13-41f8-9852-4fa685a0fa95
md"""## Example"""

# ╔═╡ 114b2603-81da-4a5c-baa1-4d95f456aeb3
function ift(case, F_def, ift_inv, ift_eq)
	return md"""
## Implicit function theorem ($case case)

Assume
* $F_def
* ``(w_0, \lambda_0)`` such that ``F(w_0, \lambda_0) = 0`` and $(ift_inv.content[].content)
* ``F(w, \lambda)`` is ``C^2`` in a neighborhood ``\mathcal{U}`` of ``(w_0, \lambda_0)``

Then there exists a neighborhood ``\mathcal{V} \subseteq \mathcal{U}`` there exists ``w^\star(\lambda)`` such that
$ift_eq
"""
end;

# ╔═╡ 2b30ae9e-e537-4ad1-a25e-570709c7d517
ift(
	"univariate",
	md"``F: \mathbb{R} \times \mathbb{R} \to \mathbb{R}``",
	md"``\partial_1 F(w_0, \lambda_0) \neq 0``",
	md"""
```math
\begin{align}
w^\star(\lambda_0) & = w_0\\
F(w^\star(\lambda), \lambda) & = 0,
\qquad\qquad\qquad \forall(w^\star(\lambda),\lambda)\in \mathcal{V}\\
\partial w^\star(\lambda) & = -\frac{\partial_2 F(w^\star(\lambda), \lambda)}{\partial_1 F(w^\star(\lambda), \lambda)}
\end{align}
```
""",
)

# ╔═╡ c1efb93f-efd8-487a-94d2-aa7ecedb61b8
ift(
	"multivariate",
	md"``F: \mathcal{W} \times \Lambda \to \mathcal{W}``",
	md"``\partial_1 F(w_0, \lambda_0)`` is invertible",
	md"""
```math
\begin{align}
w^\star(\lambda_0) & = w_0\\
F(w^\star(\lambda), \lambda) & = 0,
\qquad\qquad\qquad \forall(w^\star(\lambda),\lambda)\in \mathcal{V}\\
\partial w^\star(\lambda) & = -\partial_1 F(w^\star(\lambda), \lambda)^{-1}\partial_2 F(w^\star(\lambda), \lambda)
\end{align}
```
""",
)

# ╔═╡ bb4e1c20-1d24-494a-82ae-433e8ba56110
md"""
# Implicit VJP and JVP

How can we integrate implicit function in a chain of functions ? Say, ``s_{k-1} = \lambda`` and ``f_k(\lambda) = w^\star(\lambda)`` such that ``F(w^\star(\lambda), \lambda) = 0``.
"""

# ╔═╡ 0fc437e3-37cc-4905-8bf9-3e8b9155d6e8
md"""
## Pushforward operator (JVP)

Let ``A = -\partial_1 F(w^\star(\lambda), \lambda)`` and ``B = \partial_2 F(w^\star(\lambda), \lambda)``.
Assuming the condition of the IFT holds, we have
```math
A \partial w^\star(\lambda)/\partial s_0 =
B \partial \lambda/\partial s_0
```
Given a forward tangent ``t_{k-1} = \partial \lambda / \partial s_0``,
the forward tangent ``t_{k} = \partial w^\star(\lambda) / \partial s_0`` can be obtained
by solving the linear system:
```math
\begin{align}
A t_k & =
B t_{k-1}\\
t_k & =
A^{-1} B t_{k-1}
\end{align}
```
"""

# ╔═╡ e9fd46e4-78ce-4cba-b20f-1c997c80e842
md"""
## Interlude: Repeated linear system solve

We need to solve the system ``At_k = Bt_{k-1}`` once by forward tangent, so once by entry of ``s_0`` in case of forward-mode AD.
We may also pre-compute ``A^{-1}B`` by solving one linear system by column of ``B``.
In both case, we need to solve several linear system with the **same** ``A`` matrix.
"""

# ╔═╡ fa019742-c473-4f1d-86ce-a795edbd1f92
A = [1 2; 3 4]

# ╔═╡ d0c4b33b-63e2-49dd-aff2-d9bd1bbb4661
b = [5, 6]

# ╔═╡ 1de5d608-dc91-4f4a-995b-3dcfad38b81c
md"Classical Gaussian elimination finds the solution for one vector only, even though the same row operations would be applied for different vectors."

# ╔═╡ c3b59d62-144e-4541-9007-19187f2fea1c
rref([A b])

# ╔═╡ 64c4e42c-e444-44f1-8318-befed7d11bb5
md"""
## Solution: precompute the LU decomposition

LU decomposition remember the row operations to make ``U`` triangular in the ``L`` matrix.
"""

# ╔═╡ 17f3594f-b568-416f-9a19-38bfd4d99dd0
F = lu(A)

# ╔═╡ 6f8714a1-1372-42f4-9b51-da93ef2ed4c9
md"""As ``L`` and ``U`` are triangular, solving the linear systems ``LUx = b`` can now be done by solving two simple linear systems (and permutation in case of pivoting)
1. ``Lc = b``
2. ``Ux = c``
"""

# ╔═╡ b19451b4-2a9e-4c25-888e-d72a13eb7589
md"## Interlude: Adjoint of inverse"

# ╔═╡ f369cf71-6eea-4aaa-b75d-79c2629ce88e
md"""
Consider a linear map ``A : \mathcal{X} \to \mathcal{Y}`` between linear spaces of the same dimension.
Assume ``A`` is invertible, what is the adjoint of ``A^{-1}`` ?
"""

# ╔═╡ ea41e038-21d8-4321-bc17-25eab0a8cbb2
md"""
## Pullback operator (VJP)

The pullback operation is the adjoint of the pushforward operator:
```math
\langle r, A^{-1} B t\rangle
=
\langle (A^{-1})^* r, B t\rangle
=
\langle (A^*)^{-1} r, B t\rangle
=
\langle B^*(A^*)^{-1} r ,  t\rangle
```
So the pullback operator maps ``r`` to ``B^*(A^*)^{-1} r``.

This means that it first solves the linear system ``A^*v = r`` (possibly in a matrix-free way) and then returns ``Bv``.
"""

# ╔═╡ 10ca4307-929a-4e5f-942b-5d6c0e28e2be
md"""
# Sensitivity of a linear program

Consider the primal-dual pair of programs
```math
\begin{align}
\min & \,\, c^\top x & \max & \,\, b^\top y\\
Ax & = b & A^\top y & \le c\\
x & \ge 0
\end{align}
```

The Lagrangian function is
```math
\mathcal{L}(x, y) = c^\top x - y^\top (Ax - b) = y^\top b - (A^\top y - c)^\top x
```

The KKT condition give
```math
\begin{align}
\frac{\partial \mathcal{L}}{\partial y} & = Ax - b = 0 & \Leftrightarrow \qquad Ax & = b\\
(A^\top y - c) & \perp x \ge 0 & \Leftrightarrow \qquad \text{Diag}(x)(A^\top y - c) & = 0
\end{align}
```
"""

# ╔═╡ 6f4e1705-22f2-4be6-9f93-cbaabf809e84
md"""
## IFT for linear programs

The system of equation to consider for the IFT is therefore
```math
F((x, y), (A, b, c)) = (Ax - b, \text{Diag}(x)(A^\top y - c))
```
We have
```math
\begin{align}
\partial_1 F & = \begin{bmatrix}
	A & 0\\
	\text{Diag}(A^\top y - c) & \text{Diag}(x) A^\top
\end{bmatrix}
\end{align}
```

See [OptNet: Differentiable optimization as a layer in neural networks](https://dl.acm.org/doi/abs/10.5555/3305381.3305396) for a generalization to quadratic objective.
"""

# ╔═╡ b16f6225-1949-4b6d-a4b0-c5c230eb4c7f
md"## Utils"

# ╔═╡ 66a8b541-b06f-403a-a45d-23461a2c539b
import ForwardDiff, FiniteDifferences, Mooncake

# ╔═╡ cb886515-1635-43d7-82fd-a2604d1e3ad6
FiniteDifferences.central_fdm(5, 1)(my_sqrt, 2.0)

# ╔═╡ 0867ec30-4fd7-4338-9256-4a84215ad5aa
ForwardDiff.derivative(my_sqrt, 2.0)

# ╔═╡ 878bec4d-9d8a-4e92-9817-ddcd95bd8d35
import DifferentiationInterface as DI

# ╔═╡ 08c4ac83-4361-41d5-b036-789c2475ff8e
DI.derivative(my_sqrt, DI.AutoMooncake(), 2.0)

# ╔═╡ cbfc0129-9361-4edb-a467-1456a1f3aeae
begin
struct Path
    path::String
end

function imgpath(path::Path)
    file = path.path
    if !('.' in file)
        file = file * ".png"
    end
    return joinpath(joinpath(@__DIR__, "images", file))
end

function img(path::Path, args...; kws...)
    return PlutoUI.LocalResource(imgpath(path), args...)
end

struct URL
    url::String
end

function save_image(url::URL, html_attributes...; name = split(url.url, '/')[end], kws...)
    path = joinpath("cache", name)
    return PlutoTeachingTools.RobustLocalResource(url.url, path, html_attributes...), path
end

function img(url::URL, args...; kws...)
    r, _ = save_image(url, args...; kws...)
    return @htl("<a href=$(url.url)>$r</a>")
end

function img(file::String, args...; kws...)
    if startswith(file, "http")
        img(URL(file), args...; kws...)
    else
        img(Path(file), args...; kws...)
    end
end
end

# ╔═╡ b6f76d2d-3389-4dec-9aef-b7feeb7e0793
img("Blondel_Rouvet_Figure_8_3")

# ╔═╡ 9d883700-0959-4ccd-be50-9ee316a6077e
md"""## Acknowledgements and further readings

* Example from $(img("https://upload.wikimedia.org/wikipedia/commons/8/86/Google_JAX_logo.svg", :height => 16)) : [Implicit function differentiation of iterative implementations](https://docs.jax.dev/en/latest/advanced-autodiff.html#example-implicit-function-differentiation-of-iterative-implementations)
* Chapter 11 of [The Elements of Differentiable Programming book](https://diffprog.github.io/)
* See [DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl) for implicit differentiation of $(img("https://jump.dev/assets/jump-logo-with-text.svg", :height => 20)) optimization problems.
"""

# ╔═╡ 81deb227-a822-4857-a584-a51cc8ff51f4
begin
function qa(question, answer)
    return @htl("<details><summary>$question</summary>$answer</details>")
end
function _inline_html(m::Markdown.Paragraph)
    return sprint(Markdown.htmlinline, m.content)
end
function qa(question::Markdown.MD, answer)
    # `html(question)` will create `<p>` if `question.content[]` is `Markdown.Paragraph`
    # This will print the question on a new line and we don't want that:
    h = HTML(_inline_html(question.content[]))
    return qa(h, answer)
end
end

# ╔═╡ e1b83a0b-2e8f-4db6-ac2f-7a7c2ccce965
qa(md"How can we differentiate with respect to `x` ? Forward, reverse, something else ?",
md"""
* We could throw dual numbers at it and it would work, but `f` would be twice slower with dual numbers and we would need to do it as many times as the dimension of `x`.
* We could also create the expression graph to run reverse diff but the graph may be quite large as well.
* Some programs may be external libraries written in Fortran or C so we can't easily run AD through it.
* Instead, we could run the iteration through the end and once we get the pair `x, y`, we can see if there isn't a better way to find the derivative without rerunning the numerical iterations. This will be implicit differentiation.
""")

# ╔═╡ 383b3a50-ef75-4ccd-a6ea-1ddb87d6e9d2
qa(md"Can't we do something simpler using the solution of the fixed point and the fixed point equation ?", md"IFT...")

# ╔═╡ f1f1a354-cecc-4fb2-8d1e-096f51e4cf88
qa(md"**Proof sketch**", md"""
The proof of bijectivity and that ``f`` is ``C^2`` in the neighborhood of ``f(w_0)`` are in Section 11.5.3 of [The Elements of Differentiable Programming book](https://diffprog.github.io/).
The formula for ``\partial f^{-1}(\omega)`` is a direct consequence of the chain rule:
```math
\begin{align}
   f \circ f^{-1}(\omega) & = \omega\\
   \partial f(f^{-1}(\omega)) \partial f^{-1}(\omega) & = I.
\end{align}
```
""")

# ╔═╡ 0fc99cae-bfe4-4517-a578-453cab15d977
circle_qa = qa(md"Does that contradict the Inverse Function Theorem (IFT) ?", md"""
Let ``F(y, x) = x^2 + y^2 - 1``.
Given an initial point ``(x_0, y_0)`` with ``x_0^2 + y_0^2 = 1``,
* if ``y_0 > 0`` then IFT holds and ``y^\star(x) = y^+(x)``,
* if ``y_0 < 0`` then IFT holds and ``y^\star(x) = y^-(x)``,
* if ``y_0 = 0`` then ``\partial_1 F(y_0, x_0) = 2y_0 = 0`` so IFT does not hold.
""");

# ╔═╡ d701392f-0b25-4c27-989a-a61c23c41d9c
hbox([
	md"""
Implicit relation between ``x`` and ``y``:
```math
x^2 + y^2 = 1
```
Two possible explicit functions:
```math
\begin{align}
y^+(x) & = \sqrt{1 - x^2}\\
y^-(x) & = -\sqrt{1 - x^2}
\end{align}
```
$circle_qa
""",
	Div(html" ", style = Dict("flex-grow" => "1")),
	img("Blondel_Rouvet_Figure_11_3", :height => 300),
])

# ╔═╡ f22bfdc8-c25d-4844-98e3-08d05533ba56
qa(md"**Proof**", md"""
Define ``f(\lambda, w) = (\lambda, F(w, \lambda))``. The Jacobian is
```math
\partial f(\lambda, w)
=
\begin{bmatrix}
   I & 0\\
   \partial_2 F(w, \lambda) & \partial_1 F(w, \lambda)
\end{bmatrix}
```
By assumption, ``\partial_1 F(w, \lambda)`` is invertible so ``\det(\partial f(\lambda, w)) = \det(I)\det(\partial_1 F(w, \lambda)) \neq 0``.
By the inverse function theorem, the inverse of ``f`` around ``f(\lambda_0, w_0) = (\lambda_0, 0)`` is ``C^2``.
The ``w^\star(\lambda)`` be the second component of ``f^{-1}``: ``(\lambda, w^\star(\lambda)) = f^{-1}(\lambda, 0)``.
The Jacobian ``\partial w^\star(\lambda)`` is the derivative of the second component of ``f^{-1}`` with respect to its first input so it is the block 21 in the Jacobian
```math
\partial f^{-1}(\lambda, 0) = \begin{bmatrix}
   \sim & \sim\\
   \partial w^\star(\lambda) & \sim
\end{bmatrix}
```
By the inverse function theorem
```math
\partial f^{-1}(\lambda, 0)
= (\partial f(\lambda, w))^{-1}
= \begin{bmatrix}
   I & 0\\
   \partial_2 F(w, \lambda) & \partial_1 F(w, \lambda)
\end{bmatrix}^{-1}
```
We can then use the following result:
   
> [**Schur complement**](https://en.wikipedia.org/wiki/Schur_complement): If ``A`` is an invertible matrix then for any matrices ``B, C, D`` of compatible dimension:
> ```math
> \begin{bmatrix}
> A & B\\
> C & D
> \end{bmatrix}^{-1} = \begin{bmatrix}
> \sim & \sim\\
> -(M/A)^{-1}CA^{-1} & \sim
> \end{bmatrix}
> ```
> where the *Schur complement* of the block ``A`` is ``M/A = D - CA^{-1}B``.

In our case the Schur complement of the block ``I`` is ``\partial f(\lambda, w)/I = \partial_2 F(w, \lambda)``, this means that
```math
\partial w^\star(\lambda) = -\partial_1 F(w, \lambda)^{-1}\partial_2 F(w, \lambda).
```
""")

# ╔═╡ 528326dd-1344-4baa-9edd-5b09f63669dd
qa(md"Can we also solve the linear system with only access to the matrix-vector product of the linear map ``\partial_1 F(w^\star(\lambda), \lambda)`` ? (aka matrix-free inversion)", md"""
If the matrix is positive semidefinite, we can use the [Conjugate Gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method). Otherwise, [GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method) or [BiCGSTAB](https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method) are options.
""")

# ╔═╡ 30afb6e9-1ce0-4004-807b-dca57b789573
qa(md"Is the adjoint ``A^*`` always invertible ?", md"""
As ``\mathcal{X}`` and ``\mathcal{Y}`` have the same dimension, this is equivalent to
the ``\mathrm{Ker}(A^*) = \{0\}``. Consider an arbitrary ``y \in \text{Ker}(A^*)``.
Since ``A^*(y) = 0``, for all ``x \in \mathcal{X}``, we have
```math
\begin{align}
   \langle x, A^*(y) \rangle & = 0\\
   \langle A(x), y \rangle & = 0
\end{align}
```
As this is true for all ``x``, this is true in particular for ``x = A^{-1}(y)`` hence we have ``\langle y, y \rangle`` hence ``y = 0``.
""")

# ╔═╡ 877c563a-2ce3-4f97-9746-7d8329c0178a
qa(md"Is the adjoint of the inverse equal to the inverse of the adjoint ?",
  md"""
For arbitrary ``r, t``, we have
```math
\begin{align}
\langle (A^{-1})^*r, t\rangle
&= \langle r, A^{-1} t\rangle\\
&= \langle A^*(A^*)^{-1} r, A^{-1} t \rangle\\
&= \langle (A^*)^{-1} r, AA^{-1} t\rangle\\
&= \langle (A^*)^{-1} r, t\rangle.
\end{align}
```
Since ``r, t`` are arbitrary, we have ``(A^{-1})^* = (A^*)^{-1}``.
""")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
RowEchelon = "af85af4c-bcd5-5d23-b03a-a909639aa875"

[compat]
DifferentiationInterface = "~0.7.12"
FiniteDifferences = "~0.12.33"
ForwardDiff = "~1.3.0"
HypertextLiteral = "~0.9.5"
Mooncake = "~0.4.182"
PlutoTeachingTools = "~0.4.6"
PlutoUI = "~0.7.75"
RowEchelon = "~0.2.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "6770a17a88126d8ad47eab013341e29ee9fa3688"

[[deps.ADTypes]]
git-tree-sha1 = "8b2b045b22740e4be20654175cc38291d48539db"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.20.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3b704353e517a957323bd3ac70fa7b669b5f48d4"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.6"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
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

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "80bd15222b3e8d0bc70d921d2201aa0084810ce5"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.12"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "fc34127e78323c49984e1a146d577d0f890dd2b4"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.26"

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"
    DispatchDoctorEnzymeCoreExt = "EnzymeCore"

    [deps.DispatchDoctor.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FiniteDifferences]]
deps = ["ChainRulesCore", "LinearAlgebra", "Printf", "Random", "Richardson", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "0ff4ed4351e1884beff16fc4d54490c6d56b2199"
uuid = "26cc04aa-876d-5657-8c51-4c34ba976000"
version = "0.12.33"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cd33c7538e68650bd0ddbb3f5bd50a4a0fa95b50"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.0"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Inflate", "LinearAlgebra", "Random", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "d80e8b7220b884117938b2d219487eea06f9e42e"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.2"

    [deps.Graphs.extensions]
    GraphsSharedArraysExt = "SharedArrays"

    [deps.Graphs.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    SharedArrays = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "5b6bb73f555bc753a6153deec3717b8904f5551c"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.3.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MistyClosures]]
git-tree-sha1 = "d1a692e293c2a0dc8fda79c04cad60582f3d4de3"
uuid = "dbe65cb8-6be2-42dd-bbc5-4196aaced4f4"
version = "2.1.0"

[[deps.Mooncake]]
deps = ["ADTypes", "ChainRules", "ChainRulesCore", "DispatchDoctor", "ExprTools", "Graphs", "LinearAlgebra", "MistyClosures", "Random", "Test"]
git-tree-sha1 = "0127d6b56adef8977212b5225628119fa5a65ffe"
uuid = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
version = "0.4.182"

    [deps.Mooncake.extensions]
    MooncakeAllocCheckExt = "AllocCheck"
    MooncakeCUDAExt = "CUDA"
    MooncakeDynamicExpressionsExt = "DynamicExpressions"
    MooncakeFluxExt = "Flux"
    MooncakeFunctionWrappersExt = "FunctionWrappers"
    MooncakeJETExt = "JET"
    MooncakeLuxLibExt = ["LuxLib", "MLDataDevices", "Static"]
    MooncakeLuxLibSLEEFPiratesExtension = ["LuxLib", "SLEEFPirates"]
    MooncakeNNlibExt = ["NNlib", "GPUArraysCore"]
    MooncakeSpecialFunctionsExt = "SpecialFunctions"

    [deps.Mooncake.weakdeps]
    AllocCheck = "9b6a8646-10ed-4001-bbdc-1d2f46dfbb1a"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    DynamicExpressions = "a40a106e-89c9-4ca8-8020-a735e8728b6b"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    FunctionWrappers = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    JET = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
    LuxLib = "82251201-b29d-42c6-8e01-566dec8acb11"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    SLEEFPirates = "476501e8-09a2-5ece-8869-fb82de89a1fa"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    Static = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.5.20"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "dacc8be63916b078b592806acd13bb5e5137d7e9"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "db8a06ef983af758d285665a0398703eb5bc1d66"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.75"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Richardson]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "48f038bfd83344065434089c2a79417f38715c41"
uuid = "708f8203-808e-40c0-ba2d-98a6953ed40d"
version = "1.4.2"

[[deps.RowEchelon]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f479526c4f6efcbf01e7a8f4223d62cfe801c974"
uuid = "af85af4c-bcd5-5d23-b03a-a909639aa875"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "a2c37d815bf00575332b7bd0389f771cb7987214"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.2"

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
git-tree-sha1 = "79529b493a44927dd5b13dde1c7ce957c2d049e4"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.0"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─40baa108-eb68-433f-9917-ac334334f198
# ╟─d6797e7b-fa78-4366-b88f-9925f72a5d0f
# ╟─e1b83a0b-2e8f-4db6-ac2f-7a7c2ccce965
# ╟─d907c5d7-f517-467a-a840-f94868ad7680
# ╠═5c39f536-346c-48bd-a3c7-acaefb849b9f
# ╠═a73e6557-9fed-4a1e-9e8e-c67e500eba5b
# ╠═abee47e4-1b75-4466-908a-e14113b181ab
# ╟─23049939-1b3a-4328-b159-99c3f1470e39
# ╠═cb886515-1635-43d7-82fd-a2604d1e3ad6
# ╠═0867ec30-4fd7-4338-9256-4a84215ad5aa
# ╠═08c4ac83-4361-41d5-b036-789c2475ff8e
# ╟─383b3a50-ef75-4ccd-a6ea-1ddb87d6e9d2
# ╟─c9603b16-113b-4e05-8517-d814b0879ab8
# ╟─b23a4c1d-cf35-42e6-b286-7fbcdddfd424
# ╟─f1f1a354-cecc-4fb2-8d1e-096f51e4cf88
# ╟─2b30ae9e-e537-4ad1-a25e-570709c7d517
# ╟─0b091706-da13-41f8-9852-4fa685a0fa95
# ╟─d701392f-0b25-4c27-989a-a61c23c41d9c
# ╟─0fc99cae-bfe4-4517-a578-453cab15d977
# ╟─c1efb93f-efd8-487a-94d2-aa7ecedb61b8
# ╟─f22bfdc8-c25d-4844-98e3-08d05533ba56
# ╟─114b2603-81da-4a5c-baa1-4d95f456aeb3
# ╟─bb4e1c20-1d24-494a-82ae-433e8ba56110
# ╟─b6f76d2d-3389-4dec-9aef-b7feeb7e0793
# ╟─0fc437e3-37cc-4905-8bf9-3e8b9155d6e8
# ╟─e9fd46e4-78ce-4cba-b20f-1c997c80e842
# ╠═fa019742-c473-4f1d-86ce-a795edbd1f92
# ╠═d0c4b33b-63e2-49dd-aff2-d9bd1bbb4661
# ╟─1de5d608-dc91-4f4a-995b-3dcfad38b81c
# ╠═c3b59d62-144e-4541-9007-19187f2fea1c
# ╟─64c4e42c-e444-44f1-8318-befed7d11bb5
# ╟─17f3594f-b568-416f-9a19-38bfd4d99dd0
# ╟─6f8714a1-1372-42f4-9b51-da93ef2ed4c9
# ╟─528326dd-1344-4baa-9edd-5b09f63669dd
# ╟─b19451b4-2a9e-4c25-888e-d72a13eb7589
# ╟─f369cf71-6eea-4aaa-b75d-79c2629ce88e
# ╟─30afb6e9-1ce0-4004-807b-dca57b789573
# ╟─877c563a-2ce3-4f97-9746-7d8329c0178a
# ╟─ea41e038-21d8-4321-bc17-25eab0a8cbb2
# ╟─10ca4307-929a-4e5f-942b-5d6c0e28e2be
# ╟─6f4e1705-22f2-4be6-9f93-cbaabf809e84
# ╟─9d883700-0959-4ccd-be50-9ee316a6077e
# ╟─b16f6225-1949-4b6d-a4b0-c5c230eb4c7f
# ╠═f1ba3d3c-d0a5-4290-ab73-9ce34bd5e5f6
# ╠═66a8b541-b06f-403a-a45d-23461a2c539b
# ╠═878bec4d-9d8a-4e92-9817-ddcd95bd8d35
# ╠═64250438-9ba6-44f7-b4a0-fcd5d72f05e5
# ╟─cbfc0129-9361-4edb-a467-1456a1f3aeae
# ╠═81deb227-a822-4857-a584-a51cc8ff51f4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
