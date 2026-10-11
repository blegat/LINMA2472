### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ f1ba3d3c-d0a5-4290-ab73-9ce34bd5e5f6
using PlutoUI, PlutoUI.ExperimentalLayout, HypertextLiteral, PlutoTeachingTools, Statistics, CUDA, OneHotArrays

# ╔═╡ 40baa108-eb68-433f-9917-ac334334f198
@htl("""
<p align=center style=\"font-size: 40px;\">Automatic Differentiation</p><p align=right><i>Benoît Legat</i></p>
$(PlutoTeachingTools.ChooseDisplayMode())
$(PlutoUI.TableOfContents(depth=1))
""")

# ╔═╡ 77a7de14-87d2-11ef-21ef-937b8239db5b
md"""
# Differentiation approaches

We can compute partial derivatives in different ways:

1. **Symbolically**, by fixing one of the variables and differentiating with respect to the others, either manually or using a computer.

2. **Numerically**, using the formula  
   ``f'(x) \approx (f(x + h) - f(x)) / h``.

3. **Algorithmically**, either forward or reverse : this is what we will explore here.
"""

# ╔═╡ e46fb3ff-b26f-4efb-aaa1-760e80017797
md"# Chain rule"

# ╔═╡ af404768-0663-4bc3-81dd-6931b3a486be
md"""
Consider ``f(x) = f_3(f_2(f_1(x)))``. If we don't have the expression of $f_1$ but we can only evaluate $f_i(x)$ or $f'(x)$ for a given $x$ ?
The chain rule gives
```math
f'(x) = f_3'(f_2(f_1(x))) \cdot f_2'(f_1(x)) \cdot f_1'(x).
```
Let's define $s_0 = x$ and $s_{k} = f_k(s_{k-1})$, we now have:
```math
f'(x) = f_3'(s_2) \cdot f_2'(s_1) \cdot f_1'(s_0).
```
Two choices here:
```math
\begin{align*}
& \text{Forward} & & \text{Reverse}\\
t_0 & = 1 & r_3 & = 1\\
t_{k} & = f_k'(s_{k-1}) \cdot t_{k-1} & r_k & = r_{k+1} \cdot f_{k+1}'(s_{k})\\
\end{align*}
```
"""

# ╔═╡ 2ae277d5-8be4-4ea9-a3fc-8ad601577c3a
md"# Forward Differentiation"

# ╔═╡ 586c2c6b-4a53-46d5-924b-c843e4c09859
aside(md"Figure 8.1", v_offset = -150)

# ╔═╡ 1c44706b-fd7d-4826-84a3-73e2db5adadd
md"## Implementation"

# ╔═╡ 45051873-3f0d-49b0-a9a6-bfde240594aa
struct Dual{T}
	value::T # s_k
	derivative::T # t_k
end

# ╔═╡ 3803804a-f44d-4f56-bd59-fb1401d8fb9e
Base.:-(x::Dual{T}) where {T} = Dual(-x.value, -x.derivative)

# ╔═╡ 63bdc7f5-89a5-4061-9e42-6588e4cd96c6
Base.:*(x::Dual{T}, y::Dual{T}) where {T} = Dual(x.value * y.value, x.value * y.derivative + x.derivative * y.value)

# ╔═╡ 4174d41c-877b-4878-b76e-442991907af0
-Dual(1, 2) * Dual(3, 4)

# ╔═╡ 8b8682e4-8adb-4fb1-a013-054b1d9750d7
f_1(x, y) = x * y

# ╔═╡ e211118d-eba8-467e-ae4a-665f1df02934
f_2(s1) = -s1

# ╔═╡ bc93fb96-1097-4b22-a077-558f5662efec
(f_2 ∘ f_1)(Dual(1, 2), Dual(3, 4))

# ╔═╡ 85fc455c-36cf-4a4c-aa64-84a827884693
md"# Reverse Differentiation"

# ╔═╡ 277bd2ce-fa7f-4288-be8a-0ddd8f23635c
md"""
## Two different takes on the multivariate chain rule

The chain rule gives us  
```math
\frac{\partial f_3}{\partial x} (f_1(x), f_2(x)) = \partial_1 f_3(s_1, s_2) \cdot \frac{\partial s_1}{\partial x} + \partial_2 f_3(s_1, s_2) \cdot \frac{\partial s_2}{\partial x}
```
To compute this expression, we need the values of ``s_1(x)`` and ``s_2(x)`` as well as the derivatives ``\partial s_1 / \partial x`` and ``\partial s_2 / \partial x``.

Common to forward and reverse: Given ``s_1, s_2``, computes **local** derivatives ``\partial_1 f_3(s_1, s_2)`` and ``\partial_2 f_3(s_1, s_2)``, shortened ``\partial_1 f_3, \partial_2 f_3`` for conciseness.
"""

# ╔═╡ fa5dba01-a3f7-452c-877e-352d578ecf51
hbox([
	md"""
#### Forward

```math
\begin{align}
t_3 & = \partial_1 f_3 \cdot t_1 + \partial_2 f_3 \cdot t_2\\
& =
\begin{bmatrix}
	\partial_1 f_3 & \partial_2 f_3
\end{bmatrix} \cdot
\begin{bmatrix}
	t_1\\
	t_2
\end{bmatrix}\\
& =
\partial f_3 \cdot
\begin{bmatrix}
	t_1\\
	t_2
\end{bmatrix}
\end{align}
```""",
	Div(html" ", style = Dict("flex-grow" => "1")),
	md"""
#### Reverse

```math
\begin{align}
\begin{bmatrix}
	r_1 &
	r_2
\end{bmatrix}
& \mathrel{\raise{0.19ex}{\scriptstyle+}} = r_3 \cdot \partial f_3\\
& \mathrel{\raise{0.19ex}{\scriptstyle+}} = r_3 \cdot
\begin{bmatrix}
	\partial_1 f_3 & \partial_2 f_3
\end{bmatrix}\\
& \mathrel{\raise{0.19ex}{\scriptstyle+}} =
\begin{bmatrix}
	r_3 \cdot\partial_1 f_3 & r_3 \cdot\partial_2 f_3
\end{bmatrix}
\end{align}
```
""",
	Div(html" ", style = Dict("flex-grow" => "1")),
	md"""
#### Reverse*

```math
\begin{align}
\begin{bmatrix}
	r_1\\
	r_2
\end{bmatrix} & \mathrel{\raise{0.19ex}{\scriptstyle+}} = \partial f_3^* \cdot r_3\\
& \mathrel{\raise{0.19ex}{\scriptstyle+}} =
\begin{bmatrix}
	\partial_1 f_3\\ \partial_2 f_3
\end{bmatrix} \cdot r_3\\
& \mathrel{\raise{0.19ex}{\scriptstyle+}} =
\begin{bmatrix}
	\partial_1 f_3 \cdot r_3 \\ \partial_2 f_3 \cdot r_3
\end{bmatrix}
\end{align}
```
"""
])

# ╔═╡ 69c08fab-c317-462c-817c-3f841a8a0941
md"""When using automatic differentiation, don't forget that we must always evaluate the derivatives. For the following example we choose to evaluate it in ``x=3``"""

# ╔═╡ 885bc5c9-aefc-4d8a-a4da-6062c64eaa41
md"## Forward tangents"

# ╔═╡ 5aff8e66-787d-4dc5-a9b1-0fdec25ce0f0
md"## Reverse tangents"

# ╔═╡ 90850509-463d-44c7-88ae-4406aebd4be1
md"## Computation graph"

# ╔═╡ c4e91a07-2b8d-4f1e-9a63-7d05be21c8f4
md"""
We build the graph of
```math
f(x_1, x_2) = x_2 e^{x_1} \sqrt{x_1 + x_2 e^{x_2}}
```
at ``x_1 = 1``, ``x_2 = 2``. The slider first reveals the forward evaluation
of each node ``v_i``, then accumulates the adjoints
``\bar{v}_i = \partial f / \partial v_i`` backwards, one node at a time.
"""

# ╔═╡ 1d56075c-e28d-46c9-9a0a-210079172388
md"## Reverse mode in action"

# ╔═╡ 19578219-c6dd-4322-a2f4-44088ef640af
md"""
```math
f(x_1, x_2) = x_2 e^{x_1} \sqrt{x_1 + x_2 e^{x_2}}
```
"""

# ╔═╡ 7f75e3f3-c4e2-402d-be7b-336a4f65042a
md"""# Comparison

* Forward mode of ``f(x)`` with dual numbers `Dual.(x, v)` computes Jacobian-Vector Product (JVP) ``J_f(x) \cdot v``
* Reverse mode of ``f(x)`` computes Vector-Jacobian Product (VJP) ``v^\top J_f(x)`` or in other words ``J_v(x)^\top v``
"""

# ╔═╡ f4d1ee7c-4a01-4b2d-aa9b-ec41ceb0ad0f
md"## Memory usage of forward mode"

# ╔═╡ 73ba544c-616a-4db1-b91d-0b20a7b8924b
md"## Memory usage of reverse mode"

# ╔═╡ dc4feb58-d2cf-4a97-aaed-7f4593fc9732
md"""
# Discontinuity
"""

# ╔═╡ 2b631fcd-2703-42df-8a75-2fdff64b3311
md"## Forward mode"

# ╔═╡ 3556d366-0bc7-4239-b4f6-3f9bd28780e0
Base.isless(x::Dual, y::Real) = isless(x.value, y)

# ╔═╡ 69ae57b4-4e4c-44a2-aca7-d0fff89b9566
Base.isless(x::Real, y::Dual) = isless(x, y.value)

# ╔═╡ 9988fc4a-cedc-499b-a334-048cc13de000
abs(x) = ifelse(x < 0, -x, x)

# ╔═╡ ceaeb177-7a6a-4062-9659-56bebce0e77b
abs_bis(x) = ifelse(x > 0, x, -x)

# ╔═╡ e50f8f52-a73f-4186-af5e-b4ca2c021142
abs(Dual(0, 1))

# ╔═╡ 9862c791-31e8-4d59-8610-a929d72ea9c3
abs_bis(Dual(0, 1))

# ╔═╡ 6bd5ea51-54c4-46ba-8eb3-6427225e5249
md"## Issues with scalar AD"

# ╔═╡ 4d4e11a2-b6a1-490f-9c7b-ad617dbb395b
begin
	struct DummyMetadata end
	DummyMetadata(value) = DummyMetadata()
end;

# ╔═╡ 533bf4bc-494b-47c2-91b6-7b30221765f5
md"## Nodes with vector value"

# ╔═╡ 3f1068e0-97e2-444a-961d-2af55a4d797b
md"## Disadvantages of scalar-valued nodes"

# ╔═╡ 221a9c57-0072-4fb6-b61f-d7e3ab9c4be2
md"""
* If AD in an interpreted language like Python, the bottleneck of scalar-valued AD becomes the Python interpreter. Solved by using compiled language like Julia/Rust/C/C++/...
* If each node is heap allocated (e.g. with `malloc` in C), not cache friendly. Solved by a centralized storage for all the values of all nodes, e.g. `MathOptInterface.Nonlinear.ReverseAD`'s values vector (TODO hyperlink)
* Not using optimized BLAS routines or GPU kernels for matrix-matrix multiplications. Not solveable for scalar-valued AD, need vector-valued AD!
"""

# ╔═╡ e121f72b-fe6d-491a-ab03-ef92154c61ca
md"""
# Neural network
"""

# ╔═╡ 54457653-f029-4368-8bc0-15d69f22bfcd
md"""
1. Start with an input vector ``x``.
2. Apply a matrix product ``y = W_1 x`` (or ``y = W_1 x + b_1`` with bias).
3. Apply an activation function, say `tanh`, independently to each entry: ``z_i = \tanh(y_i)``.
   Using the broadcasting syntax of Julia, we can write ``z = \tanh.(y)``. Usually denoted ``\sigma(y)``.
4. Repeat with ``W_2`` for next layer etc...

```math
\hat{y} = W_\ell \tanh.(W_{\ell-1} \tanh.(\cdots W_2 \tanh.(W_1 x)))
```
"""

# ╔═╡ 8194cce9-115f-44b4-9f28-caa88b764185
md"""
## Batching

Applying a neural network to 2 inputs can be done simultaneously so that
**matrix-vector** products become **matrix-matrix** product, which better exploit the parallelization capabilities of CPUs and GPUs.
Let ``X = \begin{bmatrix}x^{(1)} & x^{(2)}\end{bmatrix}``:
```math
\begin{align}
\begin{bmatrix}
\sigma(W_1 x^{(1)}) & \sigma(W_1 x^{(2)})
\end{bmatrix}
&=
\sigma(\begin{bmatrix}
W_1 x^{(1)} & W_1 x^{(2)}
\end{bmatrix})\\
& =
\sigma(W_1 \begin{bmatrix}
x^{(1)} & x^{(2)}
\end{bmatrix})\\
& =
\sigma(W_1X)
\end{align}
```
With bias, let ``\mathbf{1}^\top = \begin{bmatrix} 1 & 1 \end{bmatrix}``:
```math
\begin{bmatrix}
W_1 x^{(1)} + b_1 & W_1 x^{(2)} + b_1
\end{bmatrix}
=
\begin{bmatrix}
W_1 x^{(1)} & W_1 x^{(2)}
\end{bmatrix} + \begin{bmatrix} b_1 & b_1 \end{bmatrix}
= W_1 X + b_1 \mathbf{1}^\top
```
"""

# ╔═╡ 292b182f-b8a4-4ac6-89ad-ef71d40eca1f
md"## Two equivalent approaches"

# ╔═╡ e9c14738-2c1f-43d8-94f9-fb5440d74092
md"``b_k`` is a **column** vector, ``S_i, X, W_i, Y`` are matrices."

# ╔═╡ b92d17a9-8481-458a-bc0a-efb7333cbc6e
TwoColumn(md"""
### Right-to-left

```math
\begin{align*}
S_{0} & = X\\
S_{2k-1} & = W_k S_{2k-2} + b_{k} \mathbf{1}^\top\\
S_{2k} & = \sigma(S_{2k-1})\\
S_{2H+1} & = W_{k+1} S_{2H}\\
S_{2H+2} & = \ell(S_{2H+1}; Y)\\
\end{align*}
```
""",
	 md"""
### Left-to-right

```math
\begin{align*}
S_{0} & = X\\
S_{2k-1} & = S_{2k-2} W_k + \mathbf{1} b_{k}^\top\\
S_{2k} & = \sigma(S_{2k-1})\\
S_{2H+1} & = S_{2H} W_{k+1}\\
S_{2H+2} & = \ell(S_{2H+1}; Y)\\
\end{align*}
```
""")

# ╔═╡ 9527686f-24e1-40bb-9a5d-22575aafec9b
md"## Evaluation"

# ╔═╡ 0980ef52-526d-421d-8c63-04272f16cbdd
md"## Spaces and scalar products"

# ╔═╡ 191f33ab-b535-4cb1-96b3-6b2f10fd633a
md"""
Let ``(\mathcal{S}_k)_k`` be the spaces for which ``s_k, t_k, r_k \in \mathcal{S}_k`` and ``f_k : \mathcal{S}_{k-1} \to \mathcal{S}_{k}``.

In the vectorized case, the Jacobian ``\partial f_k`` may be expressed more naturally as a linear map ``\partial f_k : \mathcal{S}_{k-1} \to \mathcal{S}_k`` and ``t_k = \partial f_k(t_{k-1})``. To obtain the matrix-form of the Jacobian, we need to fix a basis for ``\mathcal{S}_{k-1}`` and ``\mathcal{S}_k`` and vectorize ``t_k`` and ``t_{k-1}`` according to that basis which is often cumbersome.

Instead, we choose a scalar products ``\langle \cdot, \cdot \rangle_k`` for ``\mathcal{S}_k`` and the **adjoint** of the Jacobians according to these bases in the backward pass.
For instance, if ``\mathcal{S}_K = \mathcal{R}``, ``t_K = \langle t_K, 1 \rangle_K`` for the canonical scalar product of ``\mathcal{R}`` so we pick ``r_K = 1``. We then search for a way to obtain ``r_0(r_K)`` so that
``\langle t_0, r_0(r_K) \rangle_0 = \langle \partial f_K(\cdots\partial f_1(t_0), r_K\rangle_K``.
Then, for instance if ``\mathcal{S}_0 = \mathbb{R}^n``, with the canonical scalar product ``\langle x, y \rangle_0 = x^\top y``, we have
``\partial_i f(s_0) = \langle e_i, r_0(r_K) \rangle_0 = (r_0(r_K))_i``.
"""

# ╔═╡ c1ae773e-0f2d-4c1c-bae2-25f45bb28fbb
md"## Spaces and scalar products"

# ╔═╡ 068ef840-04cb-4de4-9512-721f57e5750e
md"## Adjoint"

# ╔═╡ 7331c447-c0a6-4a47-af00-36efc449d9d9
md"""
The *adjoint* of a linear map ``A : \mathcal{S}_1 \to \mathcal{S}_2`` for scalar products ``\langle \cdot, \cdot \rangle_1`` for ``\mathcal{S}_1`` and ``\langle \cdot, \cdot \rangle_2`` for ``\mathcal{S}_2`` is the linear map ``A^*`` such that
```math
\forall x \in \mathcal{S}_1, y \in \mathcal{S}_2, \qquad \langle A(x), y \rangle_2 = \langle x, A^*(y) \rangle_1.
```
"""

# ╔═╡ bebe2bb8-cf74-4eaa-85d1-712b990b5985
md"""
For our Jacobians:
```math
\begin{align}
\langle t_k, r_k \rangle_k
& =
\langle \partial f_k(t_{k-1}), r_k \rangle_k\\
& =
\langle t_{k-1}, \partial f_k^*(r_k) \rangle_{k-1}
\end{align}
```
so by defining the backward pass with ``r_{k-1} = \partial f_k^*(r_k)``, we get
```math
\langle t_k, r_k \rangle = \langle t_{k-1}, r_{k-1} \rangle
```
By induction, we get
```math
\langle t_0, r_0 \rangle = \langle t_K, r_K \rangle
```
"""

# ╔═╡ 29287c62-e892-448f-a9d5-12785ae4a02f
md"""## Matrix multiplication (Vectorized way)

[Kronecker product properties:](https://en.wikipedia.org/wiki/Kronecker_product)
```math
\begin{align}
\text{vec}(AXB) & = (B^\top \otimes A) \text{vec}(X) &
(A \otimes B)^\top & = A^\top \otimes B^\top
\end{align}
```
Deriving VJP and JVP for matrix product:
```math
\begin{align}
F(X) & = AX\\
G(\text{vec}(X)) \triangleq \text{vec}(F(X)) & = (I \otimes A) \text{vec}(X)\\
J_G & = (I \otimes A)\\
J_G^\top \text{vec}(R) & = (I \otimes A^\top) \text{vec}(R)\\
\partial F^*[R] = \text{mat}(J_G^\top \text{vec}(R)) & = A^\top R\\
\end{align}
```
"""

# ╔═╡ 5f6529a1-4ace-4dd0-a7e2-f51070eab695
md"""## Matrix multiplication (Scalar product way)

For the scalar product
```math
\langle X, Y \rangle
=
\sum_{i,j} X_{ij} Y_{ij}
=
\sum_{i,j} (X \odot Y)_{ij}
=
\langle \text{vec}(X), \text{vec}(Y) \rangle
=
\text{tr}(X Y^\top), \quad A^* = A^\top
```
Now, given a forward tangent ``T`` and a reverse tangent ``R``
```math
\begin{align}
\langle AT, R \rangle & = \langle T, A^\top R \rangle
\end{align}
```
so the backward pass computes ``A^\top R``.
"""

# ╔═╡ 402f0d1d-0a3f-404a-8019-afc92d154c85
md"## Inner products"

# ╔═╡ e9d6af05-63f7-4d31-85e2-b0233df23608
md"""
``\langle\rangle_i``, the last one should be just product of scalars to use ``r_K = 1``, the first one should be canonical to be one-hot.
The inner ones can be anything. Multiplying it by 2 makes one adjoint be multiplied by 2, the other by 1/2, it just cancels out.
"""

# ╔═╡ 802edb3a-4809-4c50-920b-25f7bdc255dd
md"""
## Broadcasting (Vectorized way)

Consider applying a scalar function ``f`` (e.g. ``\tanh`` to each entry of a matrix ``X``.)
```math
\begin{align}
(F(X))_{ij} & = f(X_{ij}) = f.(X)\\
G(\text{vec}(X)) \triangleq \text{vec}(F(X)) & = \text{vec}(f.(X))\\
J_G & = \text{Diag}(\text{vec}(f'.(X)))\\
J_G \text{vec}(T) & = \text{Diag}(\text{vec}(f'.(X))) \text{vec}(T)\\
\partial F[T] = \text{mat}(J_G \text{vec}(T)) & = f'.(X) \odot T\\
J_G^\top \text{vec}(R) & = \text{Diag}(\text{vec}(f'.(X))) \text{vec}(R)\\
\partial F^*[R] = \text{mat}(J_G^\top \text{vec}(R)) & = f'.(X) \odot R\\
\end{align}
```
"""

# ╔═╡ 98db9022-f8ff-4af3-9c81-89cf09771928
md"""
## Broadcasting (Scalar product way)

```math
\begin{align}
\langle f'.(X) \odot T, R \rangle = \langle T, f'.(X) \odot R \rangle.
\end{align}
```
"""

# ╔═╡ 8c202da6-1e13-43b8-a22b-94badcef2934
md"## Putting everything together"

# ╔═╡ 1994bf51-adf1-4b07-ab4c-f47552d90826
md"""## Product of Jacobians

Suppose that we need to differentiate a composition of functions:
``(f_n \circ f_{n-1} \circ \cdots \circ f_2 \circ f_1)(w)``.
For each function, we can compute a jacobian given the value of its input.
So, during a forward pass, we can compute all jacobians. We now just need to take the product of these jacobians:
```math
J_n J_{n-1} \cdots J_2 J_1
```
While the product of matrices is associative, its computational complexity depends on the order of the multiplications!
Let ``d_i \times d_{i - 1}`` be the dimension of ``J_i``.
"""

# ╔═╡ 906e5199-f2d2-4816-a195-6d2b1dee9403
md"# Wine example 🍷"

# ╔═╡ f5d3714d-3900-4dbe-9079-978a44584d1d
function normalise(x)
  μ = Statistics.mean(x, dims=2)
  σ = Statistics.std(x, dims=2, mean=μ)
  return (x .- μ) ./ σ
end

# ╔═╡ 6ddc06c0-3f5d-4cc9-8060-dd6997e0f662
md"## Neural network"

# ╔═╡ 0e13e63d-fd08-4cc1-aa37-851c537afbef
md"## Forward mode"

# ╔═╡ 2adc9595-8829-4d35-be90-a7718c2e7ce7
function forward_pass(W, X, y)
	W1, W2 = W
    y_1 = tanh.(W1 * X)
    local_der_tanh = 1 .- y_1.^2
    local_der_mse = 2 * (W2 * y_1 - y) / size(y, 2)
    return local_der_tanh, local_der_mse
end

# ╔═╡ 53b21ec0-28e9-46cd-a92e-8afc189c3a11
function forward_diff(W, X, y, j, k)
	W1, W2 = W
    T_1 = onehot(j, axes(W1, 1)) * onehot(k, axes(W1, 2))'
    J_1, J_2 = forward_pass(W, X, y)
    only((W2 * (J_1 .* (T_1 * X))) * J_2') # only: 1x1 matrix -> scalar
end

# ╔═╡ 778c40ff-4c9e-42fb-92a6-1e376837f6ef
function forward_diff(W, X, y)
	[forward_diff(W, X, y, i, j) for i in axes(W[1], 1), j in axes(W[1], 2)]
end

# ╔═╡ 43d2559f-8902-4c54-8fdf-cb268b6f868c
md"## Reverse mode"

# ╔═╡ 35f8cf4f-3fcb-4e27-9462-244406d7800e
function reverse_diff(W, X, y)
    J_1, J_2 = forward_pass(W, X, y)
    (J_1 .* (W[2]' * J_2)) * X'
end

# ╔═╡ 17c91ea8-acb7-4bbd-b0b0-0f8193f45303
md"## 🚀 GPU acceleration ⚡"

# ╔═╡ 9bbbda1f-74a6-458b-a084-9d034d6c291f
md"""
# Second-order

Consider a function ``f: \mathbb{R}^n \to \mathbb{R}``, we want to compute the Hessian ``\nabla^2 f(x)``, defined by
```math
(\nabla^2 f(x))_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
```
**Application**: Given the optimization problem:
```math
\begin{align}
\min_x f(x)\\
g_i(x) & = 0 \quad \forall i \in \{1, \ldots, m\}
\end{align}
```
The Hessian of the Lagrangian ``\mathcal{L}(x, \lambda) = f(x) - \lambda_1 g_1(x) - \cdots - \lambda_m g_m(x)`` is obtained as
```math
\nabla_x^2 \mathcal{L}(x, \lambda) = \nabla^2 f(x) - \sum_{i=1}^m \lambda_i \nabla^2 g_i(x)
```
"""

# ╔═╡ 03b6aa6d-7517-4906-9430-302516d0653b
md"""
## Second-order AD
"""

# ╔═╡ 7d79ff81-59e0-41f0-b2fe-70b41f44591f
md"""
## Notation
* Let ``f_k : \mathbb{R}^{d_{k-1}} \to \mathbb{R}^{d_k}``. ``\partial f_k \triangleq \partial f_k(s_{k-1}) \in \mathbb{R}^{d_k \times d_{k-1}}``, ``\partial^2 f_k \triangleq \partial^2 f_k(s_{k-1}) \in \mathbb{R}^{d_k \times d_{k-1} \times d_{k-1}}`` is a 3D array/tensor.
* Given ``v \in \mathbb{R}^{d_{k-1}}``, by the product ``(\partial^2 f_k \cdot v) \in \mathbb{R}^{d_k \times d_{k-1}}`` , we denote the contraction of the 3rd (or 2nd since the tensor is symmetric over its last 2 dimensions) dimension:
```math
(\partial^2 f_k \cdot v)_{ij} = \sum_{l = 1}^{d_{k-1}} (\partial^2 f_k)_{ijl} \cdot v_l
```
* Given ``u \in \mathbb{R}^{d_k}``, by the product ``(u \cdot \partial^2 f_k) \in \mathbb{R}^{d_{k-1} \times d_{k-1}}`` , we denote the contraction of the 1st dimension.
```math
(u \cdot \partial^2 f_k)_{ij} = \sum_{l = 1}^{d_k} u_l \cdot (\partial^2 f_k)_{lij}
```
* Both ``\partial^2 f_k \cdot v`` and ``u \cdot \partial^2 f_k`` are matrices so then we're back to matrix notations.
"""

# ╔═╡ da5895e7-af99-46ff-9f53-36529d1ca456
md"""
## Chain rule

```math
\begin{align}
\frac{\partial^2 (f_2 \circ f_1)}{\partial x_i \partial x_j}
& =
\frac{\partial}{\partial x_j} \left(\frac{\partial (f_2 \circ f_1)}{\partial x_i} \right)\\
& =
\frac{\partial}{\partial x_j} \left(\partial f_2 \cdot \frac{\partial f_1}{\partial x_i} \right)\\
& =
\left(\partial^2 f_2 \cdot \frac{\partial f_1}{\partial x_j} \right) \cdot \frac{\partial f_1}{\partial x_i} + 
\partial f_2 \cdot \frac{\partial^2 f_1}{\partial x_i \partial x_j}
\end{align}
```

In terms of the matrices ``J_k = \partial f_k`` and ``H_{kj} = \frac{\partial}{\partial x_j} J_k = \partial^2 f_k \cdot \frac{\partial s_{k-1}}{\partial x_j}``, it becomes
```math
\begin{align}
\frac{\partial^2 (f_2 \circ f_1)}{\partial x_i \partial x_j}
& =
H_{2j} \cdot \frac{\partial f_1}{\partial x_i} + 
J_2 \cdot \frac{\partial^2 f_1}{\partial x_i \partial x_j}
\end{align}
```
"""

# ╔═╡ 9415a6ed-c05e-4487-b0be-f342ec7424cd
md"""
## Forward on forward

Given ``\text{Dual}(s_1, t_1)`` with ``s_1 = \text{Dual}(f_1(x), \frac{\partial f_1}{\partial x_j})`` and ``t_1 = \text{Dual}(\frac{\partial f_1}{\partial x_i}, \frac{\partial^2 f_1}{\partial x_i \partial x_j})``
1. Compute ``s_2 = f_2(s_1) = (f_2(f_1(x)), J_2 \cdot \frac{\partial f_1}{\partial x_j}) = ((f_2 \circ f_1)(x), \partial (f_2 \circ f_1) / \partial x_j)``
2. Compute ``J_{f_2}(s_1)`` which gives ``\text{Dual}(J_2, H_{2j})``
3. Compute
```math
\begin{align}
J_{f_2}(s_1) \cdot t_1
& =
\text{Dual}(J_2, H_{2j}) \cdot
\text{Dual}(\frac{\partial f_1}{\partial x_i}, \frac{\partial^2 f_1}{\partial x_i \partial x_j})\\
& =
\text{Dual}(J_2 \cdot \frac{\partial f_1}{\partial x_i}, J_2 \cdot \frac{\partial^2 f_1}{\partial x_i \partial x_j} +
H_{2j} \cdot \frac{\partial f_1}{\partial x_i})\\
& =
\text{Dual}(\frac{\partial (f_2 \circ f_1)}{\partial x_i}, \frac{\partial^2 (f_2 \circ f_1)}{\partial x_i \partial x_j})
\end{align}
```
"""

# ╔═╡ 3a2132e7-7d69-42d7-89d3-b2d7679ad74f
md"""
## Forward on reverse

**Forward pass**: Given ``s_1 = \text{Dual}(f_1(x), \frac{\partial f_1}{\partial x_j})``
1. Compute ``s_2 = f_2(s_1)`` → same as forward on forward
2. Compute ``J_{f_2}(s_1)`` → same as forward on forward

**Reverse pass**: Given ``r_2 = \text{Dual}((r_2)_1, (r_2)_2)``, compute
```math
\begin{align}
r_2 \cdot J_2
& =
\text{Dual}((r_2)_1, (r_2)_2) \cdot \text{Dual}(J_2, H_{2j}) \cdot
\\
& = \text{Dual}(
(r_2)_1 \cdot J_2,
(r_2)_2 \cdot J_2 +
(r_2)_1 \cdot H_{2j})
\end{align}
```
"""

# ╔═╡ 5a79c09e-2a33-43d1-a5dc-caba3db467dd
md"""
## Reverse on forward

**Forward pass**: Given ``s_1 = \text{Dual}(f_1(x), \frac{\partial f_1}{\partial x_{\textcolor{red}i}})``
1. Forward mode computes ``s_2 = f_2(s_1) = (f_2(f_1(x)), J_2 \cdot \frac{\partial f_1}{\partial x_{\textcolor{red}i}}) = ((f_2 \circ f_1)(x), \partial (f_2 \circ f_1) / \partial x_{\textcolor{red}i})``
2. The reverse mode computes the local Jacobian of this operation : ``\partial s_2 / \partial s_1``. The local Jacobian of ``(s_1)_1 \mapsto f_2((s_1)_1)`` is ``J_2``. The local Jacobian of ``s_1 \mapsto \partial f_2((s_1)_1) (s_1)_2`` is ``(\partial^2 f_2((s_1)_1) \cdot (s_1)_2, \partial f_2((s_1)_1)) = (\partial^2 f_2(f_1(x)) \cdot \frac{\partial f_1}{\partial x_{\textcolor{red}i}}, \partial f_2(f_1(x)) = (H_{2\color{red}i}, J_2)``

**Reverse pass**:
```math
\begin{align}
  (r_1)_1 & = (r_2)_1 \cdot J_2 + (r_2)_2 \cdot H_{2{\color{red}i}}\\
  (r_1)_2 & = (r_2)_2 \cdot J_2
\end{align}
```
"""

# ╔═╡ fa6dd3f7-7b57-483b-ba0f-90c9bb7bb6a6
md"""
## Reverse on reverse

**Forward pass (2nd)**:
1. Forward pass computes ``s_2 = f_2(s_1)`` → Jacobian ``\partial s_2 / \partial s_1 = J_2``
2. Local Jacobian ``J_2 = \partial f_2(s_1)`` → The Jacobian is the 3D array ``\partial J_2 / \partial s_1 = \partial^2 f_2``
3. Backward pass computes ``r_1 = r_2 \cdot \partial f_2(s_1)`` → Jacobian of ``(s_1, r_2) \mapsto r_2 \cdot \partial f_2(s_1)`` is ``(r_2 \cdot \partial^2 f_2(s_1), \partial f_2(s_1)) = (r_2 \cdot \partial^2 f_2, J_2)``. Note that here ``r_2 \in \mathbb{R}^{d_k}`` is multiplying the first dimension of the tensor ``\partial^2 f_2(s_1) \in \mathbb{R}^{d_k \times d_{k-1} \times d_{k-1}}`` so the result is a symmetric matrix of dimension ``\mathbb{R}^{d_{k-1} \times d_{k-1}}``

**Reverse pass (2nd)**:
The result is ``r_0``, let ``\dot{r}_k`` be the second-order reverse tangent for ``r_k`` and ``\dot{s}_k`` be the second-order reverse tangent of ``s_k``.
We have
```math
\begin{align}
  \dot{r}_2 & = J_2 \cdot \dot{r}_1\\
  \dot{s}_1 & = (r_2 \cdot \partial^2 f_2(s_1)) \cdot \dot{r}_1 + \dot{s}_2 \cdot J_2
\end{align}
```
"""

# ╔═╡ c1da4130-5936-499f-bb9b-574e01136eca
md"### Acknowledgements and further readings

* `Dual` is inspired from [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)
* `Node` is inspired from [micrograd](https://github.com/karpathy/micrograd)
* [Here](https://gdalle.github.io/AutodiffTutorial/) is a good intro to AD
* Figures are from the [The Elements of Differentiable Programming book](https://diffprog.github.io/)
"

# ╔═╡ 5b85a063-cf0f-4afa-89f4-420d7350ecc3
html"<p align=center style=\"font-size: 20px; margin-bottom: 5cm; margin-top: 5cm;\">The End</p>"

# ╔═╡ b16f6225-1949-4b6d-a4b0-c5c230eb4c7f
md"## Utils"

# ╔═╡ b7d4a91e-3c52-4f86-9a0d-5e8c1f2b6d47
import PlotlyLight

# ╔═╡ 607000ef-fb7f-4204-b543-3cb6bb75ed71
let
	x = range(-1, stop = 1, length = 11)
	PlotlyLight.Plot(
		PlotlyLight.Config(
			x = collect(x),
			y = abs.(x),
			mode = "lines",
			name = "|x|",
		),
		PlotlyLight.Config(
			annotations = [
				PlotlyLight.Config(
					x = λ/2 - (1 - λ)/2,
					y = -1/2,
					ax = 0,
					ay = 0,
					xref = "x",
					yref = "y",
					axref = "x",
					ayref = "y",
					showarrow = true,
					arrowhead = 2,
					arrowcolor = "orange",
					text = "",
				)
				for λ in range(0, stop = 1, length = 11)
			],
		),
	)
end

# ╔═╡ a06be2d9-73c1-4f85-b2e7-18d3c5a90f47
import ComputationGraphExplorer as CGE

# ╔═╡ df1f29ce-026d-4935-83aa-0d597cb166dd
ScalarNode = CGE.Node{Int,DummyMetadata};

# ╔═╡ 310d6a10-b887-4dd0-ba93-f901726ddbbc
let
    width, height = 640, 360
    box_width = 45
    dy_up = 3box_width/8
    # Luxor only draws the shapes, the labels are overlaid in HTML so that
    # they are typeset by the MathJax of Pluto
    labels = []
    function _arrow(start, finish; rev, color)
        if rev
            start, finish = finish, start
        end
        Luxor.sethue(color)
        Luxor.arrow(start, finish; linewidth = 1.5, arrowheadlength = 10)
    end
    function _text(text, x, y; color = "black", fontsize = 14)
        push!(labels, @htl("""<span class="tex" style="position: absolute; left: $(x + width / 2)px; top: $(y + height / 2)px; transform: translate(-50%, -50%); white-space: nowrap; color: $color; font-size: $(fontsize)px;">\$$text\$</span>"""))
    end
    function draw_node(text, x, y; fill_color = "white", stroke_color = "#f79009", linewidth = 2, width = box_width, height = box_width)
        center = Luxor.Point(x, y)
        Luxor.sethue(fill_color)
        Luxor.box(center, width, height, 12, :fill)
        Luxor.sethue(stroke_color)
        Luxor.setline(linewidth)
        Luxor.box(center, width, height, 12, :stroke)
        _text(text, x, y)
    end
    function draw_line(f, v, y; rev, color = "black", stroke_color)
        dx_dots = box_width/4
        _text("$(v)_0", -3Δx, y; color)
        _arrow(Luxor.Point(-3Δx + box_width/2, y), Luxor.Point(-2Δx - box_width/2, y); rev, color)
        draw_node("$(f)_1", -2Δx, y; stroke_color)
        _arrow(Luxor.Point(-2Δx + box_width/2, y), Luxor.Point(-Δx - dx_dots, y); rev, color)
        _text("$(v)_1", -2Δx + box_width/2 + dy_up, y - dy_up; color)
        _text("\\cdots", -Δx, y)
        _arrow(Luxor.Point(-Δx + dx_dots, y), Luxor.Point(-box_width/2, y); rev, color)
        _text("$(v)_{k-1}", -box_width/2 - dy_up, y - dy_up; color)
        draw_node("$(f)_k", 0, y; stroke_color)
        _text("$(v)_k", box_width/2 + dy_up, y - dy_up; color)
        _arrow(Luxor.Point(box_width/2, y), Luxor.Point(Δx - dx_dots, y); rev, color)
        _text("\\cdots", Δx, y)
        _arrow(Luxor.Point(Δx + dx_dots, y), Luxor.Point(2Δx - box_width/2, y); rev, color)
        _text("$(v)_{K-1}", 2Δx - box_width/2 - dy_up, y - dy_up; color)
        draw_node("$(f)_K", 2Δx, y; stroke_color)
        _text("$(v)_K", 3Δx, y; color)
        _arrow(Luxor.Point(2Δx + box_width/2, y), Luxor.Point(3Δx - box_width/2, y); rev, color)
    end
    Luxor = CGE.Luxor
    Δx = 90
    Δy = 90
    Luxor.Drawing(width, height, :svg)
    Luxor.origin()
    Luxor.background("white")
    draw_line("f", "s", -Δy, rev = false, color = "#7ea6e0", stroke_color = "#546b8c")
    draw_line("\\partial f", "t", 0, rev = false; color = "#67ab9f", stroke_color = "#40635c")
    draw_line("\\partial f^*", "r", Δy, rev = true; color = "#ffb570", stroke_color = "#b58b63")
    ks = ["0", "1", "k-1", "k", "K-1", "K"]
    xs = [-3Δx, -2Δx + box_width/2 + dy_up, -box_width/2 - dy_up, box_width/2 + dy_up, 2Δx - box_width/2 - dy_up, 3Δx]
    for i in eachindex(ks)
        k = ks[i]
        x = xs[i]
        _text("\\mathcal{S}_{$k}", x, -3Δy/2)
        fontsize = 12
        _text("\\langle t_{$k}, r_{$k} \\rangle_{$k}", x, 3Δy/2; fontsize)
        if i != 1
            _text("=", (xs[i-1] + xs[i]) / 2, 3Δy/2; fontsize)
        end
    end
    Luxor.finish()
    svg = Luxor.svgstring()
    @htl("""<div style="position: relative; width: $(width)px; height: $(height)px;">
    <div style="position: absolute; inset: 0;">$(HTML(svg))</div>
    $labels
    </div>""")
end

# ╔═╡ 7f459370-eb6f-4d82-bf99-1528e136ca5c
import LinearAlgebra, DelimitedFiles

# ╔═╡ 895f4b81-51fe-4a1c-b958-1294743e3a9b
VectNode = let
	T = Int
	Value = Union{
		T,
		Vector{T},
		LinearAlgebra.Adjoint{T,Vector{T}},
	}
	CGE.Node{Value,DummyMetadata}
end;

# ╔═╡ 0a9d0913-a1bb-4f47-8b85-db8714d0f6ab
data = DelimitedFiles.readdlm(
    joinpath(@__DIR__, "data", "wine.data"),
    ',', Float64,
);

# ╔═╡ db7aaa4c-5568-4619-a9c4-00c7916c321a
features = permutedims(data[:, 2:end])

# ╔═╡ 0bcadb3a-4880-4e6c-bccb-b09df8ad8fa3
X = Float32.(normalise(features))

# ╔═╡ 6cfc019e-7b97-4bf1-b1f5-de9468b9ab4e
targets = permutedims(data[:, 1:1])

# ╔═╡ 2fcf25d2-fd51-4c13-b57c-86236aceead2
y = Float32.(targets .- 2)

# ╔═╡ e8b1c40d-27a6-4f39-b95e-6d3a0f81c72b
begin
	"""
	    StepSlider(range::AbstractRange; default = first(range), show_value = true)

	A `Slider` flanked by `◀` and `▶` buttons, so that a range can be walked
	one step at a time instead of by dragging.
	"""
	struct StepSlider
		range::AbstractRange
		default::Real
		show_value::Bool
	end

	StepSlider(range::AbstractRange; default = first(range), show_value = true) =
		StepSlider(range, default, show_value)

	function Base.show(io::IO, m::MIME"text/html", slider::StepSlider)
		show(io, m, @htl("""
		<div style="display: flex; align-items: center; gap: 0.6em;">
			<input type="button" value="&#9664;" title="Previous step">
			<input
				type="range"
				min=$(first(slider.range))
				step=$(step(slider.range))
				max=$(last(slider.range))
				value=$(slider.default)
				style="flex-grow: 1;">
			<input type="button" value="&#9654;" title="Next step">
			$(slider.show_value ?
				@htl("<output style='font-variant-numeric: tabular-nums;'>$(slider.default)</output>") :
				nothing)
			<script>
				const div = currentScript.parentElement
				const [back, range, forward] = div.querySelectorAll("input")
				const output = div.querySelector("output")

				const render = () => {
					if (output != null) output.value = range.value
				}

				// `@bind` reads and writes `div.value`; forward both to the range.
				Object.defineProperty(div, "value", {
					get: () => Number(range.value),
					set: (v) => { range.value = v; render() },
				})

				const publish = () => {
					render()
					div.dispatchEvent(new CustomEvent("input"))
				}

				back.addEventListener("click", () => { range.stepDown(); publish() })
				forward.addEventListener("click", () => { range.stepUp(); publish() })
				range.addEventListener("input", (e) => { e.stopPropagation(); publish() })
			</script>
		</div>
		"""))
	end

	# Pluto's `@bind` prefers `Base.get` over `Bonds.initial_value`, so this is
	# all that is needed for `graph_step` to be defined before the browser
	# renders the widget.
	Base.get(slider::StepSlider) = slider.default

	StepSlider
end;

# ╔═╡ 2e8dd18a-b784-4f7d-ac7d-e1f88ba2ca56
md"`n` = $(@bind scalar_n Slider(1:4, default = 2, show_value = true))"

# ╔═╡ ee3299e2-3367-4c68-a050-c26650a97c8c
let
	x = ScalarNode.(rand(-9:9, scalar_n))
	y = ScalarNode.(rand(-9:9, scalar_n))
	names = IdDict()
	for i in eachindex(x)
		names[x[i]] = "x[$i]"
		names[y[i]] = "y[$i]"
	end
	graph = CGE.Graph(x' * y; names)
	HTML(CGE.render_svg(graph, CGE.capture_frame(graph, "x' * y"), responsive = true))
end

# ╔═╡ 9a07a949-4ade-4325-8ae6-b05bdbbfae00
md"`n` = $(@bind vector_n Slider(1:4, default = 2, show_value = true))"

# ╔═╡ 9661a3b9-ae42-466e-967d-69230b9769b3
let
	x = VectNode(rand(-9:9, vector_n))
	y = VectNode(rand(-9:9, vector_n))
	names = IdDict(x => "x", y => "y")
	graph = CGE.Graph(x' * y; names)
	HTML(CGE.render_svg(graph, CGE.capture_frame(graph, "x' * y"), responsive = true, typeset_arrays = false))
end

# ╔═╡ c616136c-f29f-41c2-bef0-516c05831748
let
    box_width = 50
    dy_up = 3box_width/8
    function _arrow(start, finish; rev)
        if rev
            start, finish = finish, start
        end
        Luxor.arrow(start, finish; linewidth = 1.5, arrowheadlength = 10)
    end
    Typ = CGE.Typstry
    preamble = Typ.TypstString(Typ.TypstText("#set page(width: auto, height: auto, margin: 0pt, fill: none); #set text(size: 13.5pt);"))
    function _text(text, x, y)
        Luxor.sethue("black")
        io = IOBuffer()
        show(IOContext(io, :preamble => preamble), "image/svg+xml", Typ.TypstString(Typ.TypstText("\$$text\$")))
        # Typst exports its page as an opaque white path even with `fill: none`
        svg = replace(String(take!(io)), r"\n\s*<path class=\"typst-shape\" fill=\"#ffffff\"[^>]*/>" => ""; count = 1)
        Luxor.placeimage(Luxor.readsvg(svg), Luxor.Point(x, y); centered = true)
    end
    function draw_node(text, x, y; fill_color = "#fef0c7", stroke_color = "#f79009", linewidth = 2, width = box_width, height = box_width)
        center = Luxor.Point(x, y)
        Luxor.sethue(fill_color)
        Luxor.box(center, width, height, 12, :fill)
        Luxor.sethue(stroke_color)
        Luxor.setline(linewidth)
        Luxor.box(center, width, height, 12, :stroke)
        _text(text, x, y)
    end
    function draw_line(f, v, y; rev)
        dx_dots = box_width/4
        _text("$(v)_0", -3Δx, y)
        _arrow(Luxor.Point(-3Δx + box_width/2, y), Luxor.Point(-2Δx - box_width/2, y); rev)
        draw_node("$(f)_1", -2Δx, y)
        _arrow(Luxor.Point(-2Δx + box_width/2, y), Luxor.Point(-Δx - dx_dots, y); rev)
        _text("$(v)_1", -2Δx + box_width/2 + dy_up, y - dy_up)
        _text("dots.c", -Δx, y)
        _arrow(Luxor.Point(-Δx + dx_dots, y), Luxor.Point(-box_width/2, y); rev)
        _text("$(v)_(k-1)", -box_width/2 - dy_up, y - dy_up)
        draw_node("$(f)_k", 0, y)
        _text("$(v)_k", box_width/2 + dy_up, y - dy_up)
        _arrow(Luxor.Point(box_width/2, y), Luxor.Point(Δx - dx_dots, y); rev)
        _text("dots.c", Δx, y)
        _arrow(Luxor.Point(Δx + dx_dots, y), Luxor.Point(2Δx - box_width/2, y); rev)
        _text("$(v)_(K-1)", 2Δx - box_width/2 - dy_up, y - dy_up)
        draw_node("$(f)_K", 2Δx, y)
        _text("$(v)_K", 3Δx, y)
        _arrow(Luxor.Point(2Δx + box_width/2, y), Luxor.Point(3Δx - box_width/2, y); rev)
    end
    Luxor = CGE.Luxor
    Δx = 90
    Δy = 90
	@time CGE.Luxor.@drawsvg begin
	    Luxor.background("white")
        draw_line("f", "s", -Δy, rev = false)
        draw_line("partial f", "t", 0, rev = false)
        draw_line("partial f^*", "r", Δy, rev = true)
        ks = ["0", "1", "k-1", "k", "K-1", "K"]
        xs = [-3Δx, -2Δx + box_width/2 + dy_up, -box_width/2 - dy_up, box_width/2 + dy_up, 2Δx - box_width/2 - dy_up, 3Δx]
        for i in eachindex(ks)
            k = ks[i]
            x = xs[i]
            _text("S_($k)", x, -3Δy/2)
            _text("⟨t_($k), r_($k)⟩_($k)", x, 3Δy/2)
            if i != 1
                _text("=", (xs[i-1] + xs[i]) / 2, 3Δy/2)
            end
        end
	end 640 360
end

# ╔═╡ 2ca19ff6-ec22-4327-aea2-80bdca55ccef
h_slider = @bind h Slider(10:1000, default = 16, show_value = true);

# ╔═╡ 722ad63a-c2ac-4ed6-b268-41d0f8b745f1
md"`h` = $(h_slider)"

# ╔═╡ b5c3e2ef-3d47-4f44-b968-d04734be2f16
W = [rand(Float32, h, size(X, 1)), rand(Float32, size(y, 1), h)]

# ╔═╡ a580ef44-234a-4ed1-b007-920651415427
sum((W[2] * tanh.(W[1] * X) - y).^2) / size(y, 2)

# ╔═╡ 87c6a5bc-82bf-44a5-b4d6-6d50285348c0
@time reverse_diff(W, X, y)

# ╔═╡ 85303791-bdc4-468a-bc40-48ef2a186282
if CUDA.functional()
	X_gpu = CUDA.CuArray(X)
	y_gpu = CUDA.CuArray(y)
	W_gpu = CUDA.CuArray.(W)
	@time reverse_diff(W_gpu, X_gpu, y_gpu)
end

# ╔═╡ 9b4a78d8-e6da-41dd-b922-b35c895eee1a
if h < 200 # Forward Diff start being too slow for `h > 200`
	@time forward_diff(W, X, y)
end

# ╔═╡ d26cba1c-27d8-4ee7-99e5-6c0e0fc6b6e5
begin
	using LaTeXStrings
	import MathTeXEngine
	# Luxor asks Cairo for MathTeXEngine's fonts by name so we register them
	# to avoid having to install them on the system
	let fontconfig = Base.require(Base.PkgId(Base.UUID("a3f928ae-7b40-5064-980b-68af3947d34b"), "Fontconfig_jll"))
		add_font = Base.Libc.Libdl.dlsym(fontconfig.libfontconfig_handle, :FcConfigAppFontAddFile)
		dir = joinpath(pkgdir(MathTeXEngine), "assets", "fonts", "NewComputerModern")
		for file in readdir(dir; join = true)
			if endswith(file, ".otf")
				ccall(add_font, Cint, (Ptr{Cvoid}, Cstring), C_NULL, file)
			end
		end
	end
	# Luxor draws the italic glyphs of letters with the upright font so we
	# replace them with their Unicode mathematical italic counterpart.
	# The arguments of font commands such as `\mathrm{...}` are left untouched
	# and `\mathcal{X}` is replaced by the Unicode script letter since
	# MathTeXEngine would use the bold script one.
	_math_italic(c) = c == 'h' ? 'ℎ' : islowercase(c) ? Char(0x1D44E + (c - 'a')) : Char(0x1D434 + (c - 'A'))
	_math_script(c) = get(Dict('B' => 'ℬ', 'E' => 'ℰ', 'F' => 'ℱ', 'H' => 'ℋ', 'I' => 'ℐ', 'L' => 'ℒ', 'M' => 'ℳ', 'R' => 'ℛ'), c, Char(0x1D49C + (c - 'A')))
	function _latex(s)
		s = replace(s, r"\\mathcal\{([A-Z])\}" => m -> string(_math_script(m[end-1])))
		s = replace(s, r"\\(math[a-z]+|text[a-z]*|operatorname)\{[^}]*\}|\\[a-zA-Z]+|[a-zA-Z]" => m -> length(m) == 1 ? string(_math_italic(m[1])) : m)
		return latexstring(s)
	end
end

# ╔═╡ c62eb6b6-aca2-4c27-ab7e-e3374515d9aa
let
    box_width = 45
    dy_up = 3box_width/8
    function _arrow(start, finish; rev, color)
        if rev
            start, finish = finish, start
        end
        Luxor.sethue(color)
        Luxor.arrow(start, finish; linewidth = 1.5, arrowheadlength = 10)
    end
    function _text(text, x, y; color = "black", fontsize = 14)
        Luxor.sethue(color)
        Luxor.fontsize(fontsize)
        Luxor.text(
            _latex(text),
            Luxor.Point(x, y);
            halign = :center,
            valign = :middle,
        )
    end
    function draw_node(text, x, y; fill_color = "white", stroke_color = "#f79009", linewidth = 2, width = box_width, height = box_width)
        center = Luxor.Point(x, y)
        Luxor.sethue(fill_color)
        Luxor.box(center, width, height, 12, :fill)
        Luxor.sethue(stroke_color)
        Luxor.setline(linewidth)
        Luxor.box(center, width, height, 12, :stroke)
        _text(text, x, y)
    end
    function draw_line(f, v, y; rev, color = "black", stroke_color)
        dx_dots = box_width/4
        _text("$(v)_0", -3Δx, y; color)
        _arrow(Luxor.Point(-3Δx + box_width/2, y), Luxor.Point(-2Δx - box_width/2, y); rev, color)
        draw_node("$(f)_1", -2Δx, y; stroke_color)
        _arrow(Luxor.Point(-2Δx + box_width/2, y), Luxor.Point(-Δx - dx_dots, y); rev, color)
        _text("$(v)_1", -2Δx + box_width/2 + dy_up, y - dy_up; color)
        _text("\\cdots", -Δx, y)
        _arrow(Luxor.Point(-Δx + dx_dots, y), Luxor.Point(-box_width/2, y); rev, color)
        _text("$(v)_{k-1}", -box_width/2 - dy_up, y - dy_up; color)
        draw_node("$(f)_k", 0, y; stroke_color)
        _text("$(v)_k", box_width/2 + dy_up, y - dy_up; color)
        _arrow(Luxor.Point(box_width/2, y), Luxor.Point(Δx - dx_dots, y); rev, color)
        _text("\\cdots", Δx, y)
        _arrow(Luxor.Point(Δx + dx_dots, y), Luxor.Point(2Δx - box_width/2, y); rev, color)
        _text("$(v)_{K-1}", 2Δx - box_width/2 - dy_up, y - dy_up; color)
        draw_node("$(f)_K", 2Δx, y; stroke_color)
        _text("$(v)_K", 3Δx, y; color)
        _arrow(Luxor.Point(2Δx + box_width/2, y), Luxor.Point(3Δx - box_width/2, y); rev, color)
    end
    Luxor = CGE.Luxor
    Δx = 90
    Δy = 90
	@time CGE.Luxor.@drawsvg begin
	    Luxor.background("white")
        draw_line("f", "s", -Δy, rev = false, color = "#7ea6e0", stroke_color = "#546b8c")
        draw_line("\\partial f", "t", 0, rev = false; color = "#67ab9f", stroke_color = "#40635c")
        draw_line("\\partial f^*", "r", Δy, rev = true; color = "#ffb570", stroke_color = "#b58b63")
        ks = ["0", "1", "k-1", "k", "K-1", "K"]
        xs = [-3Δx, -2Δx + box_width/2 + dy_up, -box_width/2 - dy_up, box_width/2 + dy_up, 2Δx - box_width/2 - dy_up, 3Δx]
        for i in eachindex(ks)
            k = ks[i]
            x = xs[i]
            _text("\\mathcal{S}_{$k}", x, -3Δy/2)
            fontsize = 12
            _text("\\langle t_{$k}, r_{$k} \\rangle_{$k}", x, 3Δy/2; fontsize)
            if i != 1
                _text("=", (xs[i-1] + xs[i]) / 2, 3Δy/2; fontsize)
            end
        end
	end 640 360
end

# ╔═╡ b3c8d70e-9a41-4d26-85fb-6e02f19ca4d3
begin
	# The graph carries the primal value; the adjoint is user metadata.
	mutable struct Adjoint
		derivative::Float64
	end
	CGE.metadata(::Type{Adjoint}, ::Float64) = Adjoint(0.0)
	CGE.metadata_rows(data::Adjoint) = ["adjoint" => data.derivative]
	CGE.seed_metadata!(data::Adjoint, is_output::Bool) =
		data.derivative = is_output ? 1.0 : 0.0
	const GraphNode = CGE.Node{Float64,Adjoint}
end;

# ╔═╡ f9a37c14-5b62-4e8d-96a0-2c41db73e65f
begin
	# One pullback per operation: given the adjoint of the output node,
	# accumulate the adjoint of each input node.
	function CGE.pullback!(::typeof(+), node::GraphNode, args::GraphNode...)
		for arg in args
			arg.metadata.derivative += node.metadata.derivative
		end
	end
	function CGE.pullback!(::typeof(*), node::GraphNode, x::GraphNode, y::GraphNode)
		x.metadata.derivative += node.metadata.derivative * y.value
		y.metadata.derivative += node.metadata.derivative * x.value
	end
	function CGE.pullback!(::typeof(exp), node::GraphNode, x::GraphNode)
		x.metadata.derivative += node.metadata.derivative * node.value
	end
	function CGE.pullback!(::typeof(sqrt), node::GraphNode, x::GraphNode)
		x.metadata.derivative += node.metadata.derivative / (2 * node.value)
	end
end;

# ╔═╡ c71e4f82-0d35-49ba-97c6-84a1bd50e739
graph_example = let
	x₁ = GraphNode(1.0)
	x₂ = GraphNode(2.0)
	v₁ = exp(x₁)
	v₂ = exp(x₂)
	v₃ = x₂ * v₂
	v₄ = x₁ + v₃
	v₅ = sqrt(v₄)
	v₆ = x₂ * v₁
	f = v₆ * v₅
	names = IdDict(
		x₁ => "x₁", x₂ => "x₂",
		v₁ => "v₁", v₂ => "v₂", v₃ => "v₃",
		v₄ => "v₄", v₅ => "v₅", v₆ => "v₆",
		f => "f",
	)
	CGE.Graph(f; names)
end;

# ╔═╡ d492a165-3e70-4c18-b0d9-57fc2e8a1b96
graph_frames = let
	frames = CGE.forward_frames(graph_example)
	order = CGE.topological_order(graph_example.output)
	for node in order
		node.metadata.derivative = 0.0
	end
	graph_example.output.metadata.derivative = 1.0
	push!(frames, CGE.capture_frame(
		graph_example, "Reverse pass: seed f̄ = 1"; active = graph_example.output,
	))
	for node in reverse(order)
		isempty(node.args) && continue
		CGE.pullback!(node)
		push!(frames, CGE.capture_frame(
			graph_example,
			"Reverse pass: propagate from $(graph_example.names[node])";
			active = node,
		))
	end
	frames
end;

# ╔═╡ d18fb5c2-6e47-4a90-b3d1-90c7af4e2b16
@bind graph_step StepSlider(eachindex(graph_frames))

# ╔═╡ e52c7a3b-8d19-4c60-a7f2-31b6ec9d5a08
HTML(CGE.render_svg(graph_example, graph_frames[graph_step], responsive = true))

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

# ╔═╡ fa20b8db-9ac7-490d-b8d8-8d57469d24e4
img("Blondel_Rouvet_Figure_8_1", :height => 250)

# ╔═╡ cc2a09a1-c949-4b09-816b-b49ba7ca8983
img("Blondel_Rouvet_Figure_8_3", :height => 250)

# ╔═╡ d2b8fa5c-c604-4093-a2dd-5c95f2eaa676
img("Blondel_Rouvet_Figure_8_7", :height => 250)

# ╔═╡ d1dbdd3f-9782-4fba-8c4e-819f152e6c30
img("Blondel_Rouvet_Figure_8_8", :height => 200)

# ╔═╡ 673c3acc-0009-416a-91bb-f57c1fe8eefc
img("Blondel_Rouvet_Figure_4_3", :height => 250)

# ╔═╡ e8c60922-5bbf-45b5-8311-18c8f8525623
img("Blondel_Rouvet_Figure_8_2", :height => 250)

# ╔═╡ 2f8baccc-19d1-44d6-b71f-0243fd8696ba
img("Blondel_Rouvet_Figure_8_4", :height => 300)

# ╔═╡ cd6d807d-6238-44ce-9267-1614679f527a
img("Blondel_Rouvet_Figure_8_5", :height => 150)

# ╔═╡ 40ed6c94-d2f9-4225-80b3-9060f04f8971
img("Blondel_Rouvet_Figure_8_6", :height => 350)

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

# ╔═╡ 8deca676-8a0b-41eb-b7a0-4d65e1158b0b
qa(md"Apply the automatic differentiation to ``s_3=f_3(s_1, s_2) = s_1 + s_2``, with ``s_1=f_1(x) = x`` and ``s_2=f_2(x) = x^2``",
hbox([
	md"""
#### Forward

* ``\partial x / \partial x = 1``
* ``\partial s_1 / \partial x = 1 \vert_{x=3} \cdot 1 = 1``
* ``\partial s_2 / \partial x = 2x \vert_{x=3} \cdot 1 = 6``
* ``\partial s_3 / \partial x = 1 \vert_{x=3} \cdot 1 + 1 \vert_{x=3} \cdot 6 = 7``
""",
	Div(html" ", style = Dict("flex-grow" => "1")),
md"""
#### Reverse

* Initialize ``\partial s_3 / \partial s_1 = \partial s_3 / \partial s_2 = \partial s_3 / \partial x = 0``
* First part: ``\partial s_3/\partial s_1 \mathrel{\raise{0.19ex}{\scriptstyle+}} = 1``
  - ``\partial s_3 / \partial x \mathrel{\raise{0.19ex}{\scriptstyle+}} = 1 \cdot 1 \vert_{x=3}``
* Second part: ``\partial s_3/\partial s_2 \mathrel{\raise{0.19ex}{\scriptstyle+}} = 1``
  - ``\partial s_3 / \partial x \mathrel{\raise{0.19ex}{\scriptstyle+}} = 1 \cdot 2x \vert_{x=3}``
* The result is ``\partial s_3 / \partial x = 7``.
"""]))

# ╔═╡ 83ef86e0-bcfb-42ee-a574-16758606423a
qa(md"Why is ``\partial\text{dup}^*`` a sum ?", md"The Jacobian is ``\partial\text{dup} = \begin{bmatrix}
1\\1\\1\end{bmatrix}``. In reverse mode, we multiply by the adjoint (why ? See next lecture!) of the Jacobian (here the transpose) ``\partial\text{dup}^* = \begin{bmatrix}
1 & 1 & 1\end{bmatrix}``. Left-multiplying a vector with a row vector of ones results in its sum.")

# ╔═╡ 28df733e-7db9-4e78-9121-52d8e6ca7591
qa(md"Can this directed graph have cycles ?", md"No, it is a Directed Acyclic Graph. (DAG)")

# ╔═╡ 626abc7c-87ef-4838-9f0a-294cf0a4be6a
qa(md"What happens if ``f_4`` is handled before ``f_5`` in the backward pass ?",
md"""
``\partial f / \partial s_4`` is the sum of its contribution from ``f_5`` and ``f_7``. We must wait for ``f_5`` to be handled before we turn to ``f_4``.
""")

# ╔═╡ 6c60f9ca-ba04-41e2-9625-c9e10f1a853b
qa(md"How to prevent this from happening ?", md"We should first compute a [*topological ordering*](https://en.wikipedia.org/wiki/Topological_sorting) and then follow this order.")

# ╔═╡ bab3a3cb-0ad2-4ea5-a15c-6593fc22e496
qa(md"How can we compute the full Jacobian ?", md"By computing a JVP (resp. VJP) with a one-hot vector, we get a column (resp. row) of the jacobian.")

# ╔═╡ c73f79c6-a28f-4c7a-89e5-8d70a245a210
qa(md"When is each mode faster than the other one to compute the full Jacobian ?", md"If ``f: \mathbb{R}^n \to \mathbb{R}^m``, then computing the full Jacobian requires ``n`` JVP or ``m`` VJP. So
* if ``n \gg m``, then reverse mode is faster;
* if ``m \gg n``, then forward mode is faster;
* if ``m \approx n``, then it's a close call.")

# ╔═╡ ac52550e-3287-427f-b957-ac61bc850f4d
qa(md"When is the speed of numerical differentation comparable to autodiff ?",
md"""
With numerical differentiation, we compute a JVP with ``\partial f / \partial x_i \approx (f(x_1, \ldots, x_{i - 1}, x_i + \epsilon, x_{i + 1}, x_n) - f(x_1, \ldots, x_n)) / \epsilon``.
For this JVP, we need to evaluate ``f`` twice. On the other hand, forward mode evaluates ``f`` once but with dual numbers as inputs so this evaluation is probably around twice as expensive as evaluating ``f`` with `Float64` numbers. So the cost of a JVP should be roughly the same for numerical differentiation and forward differentiation.

Numerical differentiation may however need to increase its number of evaluations in order to improve its accuracy while forward is accurate (up to floating point rounding errors).
""")

# ╔═╡ 11eadcc1-68bf-4ca7-9603-ef46468b4779
qa(md"Why don't we show the local Jacobian ``\partial J_k`` in the memory usage ?", md"For a scalar AD, it is best to compute the local Jacobian ``\partial J_k`` and ``s_k`` from ``s_{k-1}`` during the forward pass at the same time because part of the computation for one can be reused for the other one. For instance, if ``f_k(x) = \sin(x)``, it is more efficient to compute ``\sin(x)`` and ``\cos(x)`` at the same time with the Julia function `sincos`. However, as we will see, for a vectorized AD, we usually don't compute a local Jacobian and just use ``s_{k-1}`` to do the backward propagation of ``r_k`` towards ``r_{k-1}``. This is why this picture, which is taken from the book which focuses on vectorized AD doesn't show a memory usage for the local Jacobian.")

# ╔═╡ 74063eb5-be06-466a-a2f1-e266c35295ea
qa(md"Is the function ``|x|`` is differentiable at ``x = 0`` ?.", md"No, if we approach from the left (that is, ``x < 0``, the function is ``-x``), then the derivative is ``-1``.
If we approach from the right (that is, ``x > 0``, the function is ``x``), then the derivative is ``1``.
There is no valid gradient!")

# ╔═╡ 88534196-9f4a-430c-a534-805177ba718d
qa(md"What about returning a convex combination of the derivative from the left and right ?", md"Any number between ``-1`` and ``1`` is a valid **subgradient**!
Whereas the gradient is the normal to the **unique** tangent, the subgradient is an element of the **tangent cone**, depicted below. For convex functions, the notion of subgradient appropriately generalizes the notion of gradient for nonsmooth functions.

Note that the notion of subgradient is not defined for nonconvex functions. So instead we use [Clarke differential](https://en.wikipedia.org/wiki/Clarke_generalized_derivative) but this is out of the scope of this course.")

# ╔═╡ 347a1500-43aa-4078-9ea3-480bc8a4c291
qa(md"We are free to choose any scalar product ``\langle \cdot, \cdot \rangle_k``. Does this choice affect the Jacobian values as well as gradient of ``f`` ?",
   md"""
If we change the scalar product ``\langle \cdot, \cdot \rangle_k``, it changes both the adjoint ``\partial f^*_k`` and ``\partial f^*_{k+1}``. But it does not affect the gradient of ``f`` as the changes made to both these Jacobian cancel out. The composition ``(\partial f^*_k) \circ (\partial f^*_{k+1})`` is unaffected by the choice of the scalar product ``\langle \cdot, \cdot \rangle_k``.

For example, suppose ``\mathcal{S}_0 = \mathcal{S}_1 = \mathcal{S}_2 = \mathbb{R}`` and ``f_1(x) = x^2`` and ``f_2(x) = \exp(x)``. For ``s_0 = 1``, we have ``s_1 = 2``, ``s_2  = e``, ``\partial f_1(1) = 2``, ``\partial f_2(1) = e``.
So forward math computes ``(f_2 \circ f_1)'(1) = \partial f_1(1) \cdot \partial f_2(1) = 2e``.

For ``\langle x, y \rangle_0 = \rangle x, y \rangle_1 = \rangle x, y \rangle_2 = xy``, we have ``\partial f_1^*(1) = 2``, ``\partial f_2^*(1) = e``. So the reverse mode computes ````(f_2 \circ f_1)'(1) = \partial f_1^*(1) \cdot \partial f_2^*(1) = 2e``.

For ``\rangle x, y \rangle_1 = 2xy``, we have ``\partial f_1^*(1) = 1``, ``\partial f_2^*(1) = 2e``. So the reverse mode computes ````(f_2 \circ f_1)'(1) = \partial f_1^*(1) \cdot \partial f_2^*(1) = 2e``. As we can see, by scaling the scalar product by 2, one jacobian gets scaled by 2 and one by 1/2. When we multiply them together, they cancel out.
""")

# ╔═╡ c733ca7e-b57e-4218-9bd4-238ab5749143
qa(md"How should we store the Jacobian in the forward pass to save it for the backward pass ?",
md"The matrix ``I \otimes A`` is block diagonal with the same block repeated so the structure is crucial to exploit. Storing ``A`` is enough.")

# ╔═╡ 33bdaa23-707e-4227-b936-c5d7aaf2c48e
qa(md"How to prove that ``A^* = A^\top`` ?", md"""
We have ``\langle X, Y \rangle = \text{tr}(XY^\top)`` ([why ?](https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Trace_of_a_product)) so, using the [cyclic property of the trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Cyclic_property)
```math
\langle AX, Y \rangle = \text{tr}(AXY^\top) = \text{tr}(XY^\top A) = \text{tr}(X(A^\top Y)^\top) = \langle X, A^\top Y \rangle.
```
""")

# ╔═╡ 7921c4c6-56b8-4c6f-a9be-cd1d9984680b
qa(md"Let ``A(X) = B \odot X``, what is the adjoint ``A^*`` ?", md"""
```math
\begin{align}
   \langle B \odot X, Y \rangle
   & =
   \sum_{ij} ((B \odot X) \odot Y)_{ij}\\
   & =
   \sum_{ij} (X \odot (B \odot Y))_{ij}\\
   & =
   \langle X, B \odot Y \rangle.
\end{align}
```
So ``A`` is self-adjoint, i.e., ``A^* = A``.
""")

# ╔═╡ edad5700-9b04-47c6-90f8-b6bac0897340
qa(md"What should be saved for the backward pass ?", md"The Jacobian is diagonal, we just need to compute and store the values on the diagonal: ``f.(X)`` or we can also just store ``X`` and compute ``f.(X)`` during the backward pass.")

# ╔═╡ 9ca3b930-3d9d-460c-9f20-e3c135477b05
qa(md"What is the complexity of forward mode", md"""
If the product is computed from right to left:
```math
\begin{align}
  J_{1,2} & = J_2 J_1 && \Omega(d_2d_1d_0)\\
  J_{1,3} & = J_3 J_{1,2} && \Omega(d_3d_2d_0)\\
  J_{1,4} & = J_4 J_{1,3} && \Omega(d_4d_3d_0)\\
  \vdots & \quad \vdots\\
  J_{1,n} & = J_n J_{1,(n-1)} && \Omega(d_nd_{n-1}d_0)\\
\end{align}
```
we have a complexity of
``\Omega(\sum_{i=2}^n d_id_{i-1}d_0)``.
""")

# ╔═╡ e27db9a0-09ff-4ee5-a807-d98933b6bcf1
qa(md"What is the complexity of reverse mode", md"""
The adjoint trick gives
```math
\langle J_n J_{n-1} \cdots J_2 J_1 \partial w, \partial y \rangle
=
\langle \partial w, J_1^\top J_2^\top \cdots J_{n-1}^\top J_n^\top \partial y \rangle.
```
So reverse differentation corresponds to multiplying the adjoint from right to left or equivalently the original matrices from left to right.
This means computing the product in the following order:
```math
\begin{align}
  J_{(n-1),n} & = J_n J_{n-1} && \Omega(d_nd_{n-1}d_{n-2})\\
  J_{(n-2),n} & = J_{(n-1),n} J_{n-2} && \Omega(d_nd_{n-2}d_{n-3})\\
  J_{(n-3),n} & = J_{(n-2),n} J_{n-3} && \Omega(d_nd_{n-3}d_{n-4})\\
  \vdots & \quad \vdots\\
  J_{1,n} & = J_{2,n} J_1 && \Omega(d_nd_1d_0)\\
\end{align}
```
We have a complexity of
```math
\Omega(\sum_{i=1}^{n-1} d_nd_id_{i-1}).
```
""")

# ╔═╡ 95eb9960-89f0-4edc-8943-77a75bce2b80
qa(md"What about the complexity of meeting in the middle between ``k`` and ``k+1``?",
  md"""
We can also write
```math
\langle J_n J_{n-1} \cdots J_2 J_1 \partial w, \partial y \rangle
=
\langle J_k J_{k-1} \cdots J_2 J_1\partial w, J_{k+1}^\top J_{k+2}^\top \cdots J_{n-1}^\top J_n^\top \partial y \rangle.
```
This corresponds to multiplying starting from some ``d_k`` where ``1 < k < n``.
We would then first compute the left side:
```math
\begin{align}
  J_{k+1,k+2} & = J_{k+2} J_{k+1} && \Omega(d_{k+2}d_{k+1}d_{k})\\
  J_{k+1,k+3} & = J_{k+3} J_{k+1,k+2} && \Omega(d_{k+3}d_{k+2}d_{k})\\
  \vdots & \quad \vdots\\
  J_{k+1,n} & = J_{n} J_{k+1,n-1} && \Omega(d_nd_{n-1}d_k)
\end{align}
```
then the right side:
```math
\begin{align}
  J_{k-1,k} & = J_k J_{k-1} && \Omega(d_kd_{k-1}d_{k-2})\\
  J_{k-2,k} & = J_{k-1,k} J_{k-2} && \Omega(d_kd_{k-2}d_{k-3})\\
  \vdots & \quad \vdots\\
  J_{1,k} & = J_{2,k} J_1 && \Omega(d_kd_1d_0)\\
\end{align}
```
and then combine both sides:
```math
J_{1,n} = J_{k+1,n} J_{1,k} \qquad \Omega(d_nd_kd_0)
```
we have a complexity of
```math
\Omega(d_nd_kd_0 + \sum_{i=1}^{k-1} d_kd_id_{i-1} + \sum_{i=k+2}^{n} d_id_{i-1}d_k).
```
""")

# ╔═╡ 9afd31c9-e938-417a-8c3f-e0d1ba88f95b
qa(md"Which mode should be used depending on the ``d_i`` ?", md"""
We see that we should find the minimum ``d_k`` and start from there. If the minimum is attained at ``k = n``, this corresponds mutliplying from left to right, this is reverse differentiation. If the minimum is attained at ``k = 0``, we should multiply from right to left, this is forward mode. Otherwise, we should start from the middle, this would mean mixing both forward and reverse mode.
""")

# ╔═╡ f010e781-f41e-4861-af1c-32cf5a76ce4d
qa(md"What about neural networks ?", md"""
In that case, ``d_0`` is equal to the number of entries in ``W_1`` added with the number of entries in ``W_2`` while ``d_n`` is ``1`` since the loss is scalar. We should therefore clearly multiply from left to right hence do reverse diff.
""")

# ╔═╡ 4317bba0-723f-4cdc-9d52-67033540a8d2
qa(md"How to deduce the backward pass for reverse mode from the forward mode ?", md"""
`W2 * (J_1 .* (T_1 * X))) * J_2'` → The broadcasted `*` is an Hadamard product, denoted $\odot$. So forward mode is:
```math
W_2 (J_1 \odot (T_1 X)) J_2^\top
```
As this is scalar, it is trivially equal to its scalar product with ``1``. That is, we start with a reverse tangent ``R = 1``. We can then move everything to the right-hand side with the adjoint (here, it is the transpose):
```math
\begin{align}
\langle W_2 (J_1 \odot (\partial W_1 X)) J_2^\top, 1 \rangle
& =
\langle J_1 \odot (\partial W_1 X), W_2^\top J_2 \rangle\\
& =
\langle \partial W_1 X, J_1 \odot (W_2^\top J_2) \rangle\\
& =
\langle \partial W_1, (J_1 \odot (W_2^\top J_2))X^\top \rangle\\
\end{align}
```
Now, for the derivative with respect to the entry $W_{i,j}$, we use $\partial W_1 = e_ie_j^\top$ and
```math
\langle e_ie_j^\top, (J_1 \odot (W_2^\top J_2))X^\top \rangle
=
((J_1 \odot (W_2^\top J_2))X^\top)_{ij}
```
So the gradient with respect to $W_1$ is exactly the matrix $(J_1 \odot (W_2^\top J_2))X^\top$ !
""")

# ╔═╡ 6ded46e6-1c89-4875-addb-8c709e949bb1
qa(md"Why is the GPU version slower than the CPU version ?",
md"""
The fixed cost of launching the GPU kernels is larger than the gain obtained by the GPU acceleration. It is only worth it for larger `h`.
`h` = $(h_slider).
""")

# ╔═╡ 78d8c8c9-568d-472a-9f06-a50b1cf2384b
qa(md"How can the Hessian of ``f`` be computed given an AD for Jacobian and gradient.", md"""
Define the function ``g(x) = \nabla f(x)``, the Hessian of ``f`` is then the Jacobian of ``g``:
``\nabla^2 f(x) = J_g(x)``. See the solutions of `LabAD`!
""")

# ╔═╡ f23ca90f-b567-4257-ae41-ec15c57c1f3f
qa(md"Does the AD need to be the same for the gradient and the Jacobian ?",
md"""
No. Given an implementation of reverse mode and forward mode, this gives 4 possibilities depending on which mode is used for the gradient and Jacobian.
""")

# ╔═╡ 4d76422e-711c-4f3e-87ab-0ce851bac064
qa(md"What is the closed form expression for ``t_k`` in terms of the matrices ``J_k`` and ``H_{kj}`` ?",
md"""
We can prove by induction that the second part of the dual number ``t_k`` is:
```math
\frac{\partial^2 f_k}{\partial x_i \partial x_j} = J_k \cdots J_2 H_{1j} e_i + J_k \cdots J_3 H_{2j} J_1 e_i + H_{kj} J_{k-1} \cdots J_1 e_i
```
""")

# ╔═╡ 24d52ef8-927e-46c8-b455-74ccdb33d3ca
qa(md"Which value of ``r_k`` is solution for this recurrence equation ?",
md"""
We find ``r_k = \text{Dual}(\frac{\partial f}{\partial s_k}, \frac{\partial^2 f}{\partial s_k \partial x_j})`` as solution:
```math
\begin{align}
(r_1)_2 = (r_2)_2\cdot J_2+(r_2)_1\cdot H_{2j}
& = \frac{\partial^2 f}{\partial s_2 \partial x_j} \cdot J_2 +
\frac{\partial f}{\partial s_2} \cdot H_{2j}\\
& = \frac{\partial^2 f}{\partial s_2 \partial x_j}\frac{\partial s_2}{\partial s_1} +
\frac{\partial f}{\partial s_2}\frac{\partial J_2}{\partial x_j}\\
& = \frac{\partial^2 f}{\partial s_2 \partial x_j}\frac{\partial s_2}{\partial s_1} +
\frac{\partial f}{\partial s_2}\frac{\partial^2 s_2}{\partial s_1\partial x_j}\\
& = \frac{\partial}{\partial x_j}\left(\frac{\partial f}{\cancel{\partial s_2}}\cdot\frac{\cancel{\partial s_2}}{\partial s_1}\right)\\
& = \frac{\partial^2 f}{\partial s_1 \partial x_j}
\end{align}
```
""")

# ╔═╡ 4de9baec-f444-47b8-b9e6-7f3d9e9609b1
qa(md"What is the closed form expression for ``r_k`` in terms of the matrices ``J_k`` and ``H_{kj}`` ?",
md"""
We can prove by induction that the second part of the dual number ``r_k`` is:
```math
\frac{\partial^2 f}{\partial s_k \partial x_j}^\top = (J_K \cdots J_{k+1} H_{kj} + J_K \cdots J_{k+2} H_{(k+1)j} J_k + H_{Kj} J_{K-1} \cdots J_k)^\top
```
""")

# ╔═╡ 16f611e8-8e41-43f3-830c-9976cb720b9f
qa(md"Which value of ``r_k`` is solution for this recurrence equation ?",
md"""
We find ``r_k = \text{Dual}(\frac{\partial f}{\partial s_k}, \frac{\partial^2 f}{\partial s_k \partial x_{\color{red}j}})`` as solution.
""")

# ╔═╡ 200e2b1b-065a-4d60-b5fd-86700e4c811a
qa(md"Which value of ``\dot{s}_k, \dot{r}_k`` is solution for this recurrence equation ?",
md"""
Starting with ``\dot{s}_0 = e_i``, we have
```math
\begin{align}
r_k & = J_K \cdots J_{k+1}\\
\dot{r}_k & = J_k \cdots J_1 e_i\\
(r_k \cdot \partial^2 f_k) \cdot \dot{r}_{k-1}
& =
r_k \cdot (\partial^2 f_k \cdot \dot{r}_{k-1})\\
& =
r_k \cdot H_{ki}\\
\dot{s}_k & = \sum_{k=1}^K r_k H_{ki} J_{k-1} \cdots J_1
\end{align}
```
So we find ``\dot{s}_k = \frac{\partial^2 f}{\partial s_k \partial x_i}, \dot{r}_k = \frac{\partial f}{\partial s_k}`` as solution.
""")

# ╔═╡ e4a9c57b-c811-428b-a6b8-191b78d5f361
qa(md"What is the difference with reverse on forward and forward on reverse ?",
md"""
Reverse on reverse computes the product in the order
```math
((J_K \cdots J_{k-1}) \cdot \partial^2 f_k) \cdot (J_k \cdots J_1 e_i)
```
while reverse on forward and forward on reverse compute it in the order
```math
(J_K \cdots J_{k-1}) \cdot (\partial^2 f_k \cdot (J_k \cdots J_1 e_i))
```
""")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
ComputationGraphExplorer = "c9fc7d07-15e8-4fd8-b152-23c2424d2de2"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MathTeXEngine = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
PlotlyLight = "ca7969ec-10b3-423e-8d99-40f33abb42bf"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CUDA = "~6.4.2"
ComputationGraphExplorer = "~0.2.2"
HypertextLiteral = "~1.0.0"
LaTeXStrings = "~1.4.1"
MathTeXEngine = "~0.6.9"
OneHotArrays = "~0.2.11"
PlotlyLight = "~0.13.1"
PlutoTeachingTools = "~0.4.7"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.1"
manifest_format = "2.1"
project_hash = "c121fa8ca76ca548215dd6e79d80b28fb25669a6"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
registries = "General"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

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

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6da5f4e2d02f93b0ae3896685a32971a3c61cdea"
registries = "General"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.2"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "501c2e38278e3a4b7c104ee7b160b90309fb4630"
registries = "General"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.5.0"

    [deps.Atomix.extensions]
    AtomixCUDACoreExt = "CUDACore"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDACore = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "94eab0b3ccdcac361188cc661daf69d4433c1818"
registries = "General"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.2.0"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "e9f53743fbe41a7e794d58f79d7a13699d9748fa"
registries = "General"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.2"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "8c290a1b223deaeea9aea44b235d24546da8eb98"
registries = "General"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.4.0"

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

[[deps.CUDA]]
deps = ["CUDACore", "CUDATools", "Reexport", "cuBLAS", "cuFFT", "cuRAND", "cuSOLVER", "cuSPARSE"]
git-tree-sha1 = "3da65a61794b3a5c01ef1eb1eb4879da4a594ac3"
registries = "General"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "6.4.2"

[[deps.CUDACore]]
deps = ["Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVPTX_LLVM_Backend_jll", "PrecompileTools", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "StaticArrays"]
git-tree-sha1 = "0293a2d0776306cd21fddbf3f02e7bdeb417da5a"
registries = "General"
uuid = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
version = "6.4.2"

    [deps.CUDACore.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDACore.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDATools]]
deps = ["CUDACore", "CUDA_Compiler_jll", "CUPTI", "Crayons", "GPUCompiler", "LLVM", "NVML", "NVTX", "PrecompileTools", "Preferences", "PrettyTables", "Printf", "Statistics", "demumble_jll"]
git-tree-sha1 = "e6d1e58ba4b1f292c8532b8527a08a721c7ada8b"
registries = "General"
uuid = "9ec180c6-1c07-47c7-9e6e-ebefa4d1f6d0"
version = "6.4.2"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "0383f1022db3f335ead10b4c3a31b1ad15173dc2"
registries = "General"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.6.3+1"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "e1545697ad04e71fc9d2d95b8f5f54c1244fb607"
registries = "General"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.3.5+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "159b1c1f03e6355bb9c6d8954e5f51019788dd26"
registries = "General"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "2.1.1"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Compiler_jll", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "73ee54b64592c4157e7240e237a13c65049048c3"
registries = "General"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.25.0+1"

[[deps.CUPTI]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox"]
git-tree-sha1 = "e67f5c53e4651aad1055305f24cc4f5c67058f62"
registries = "General"
uuid = "9e67e8f6-ba02-4b6c-a7db-3b11ae1e7ab7"
version = "6.4.2"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
registries = "General"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "7b841680738c19948120f6e4cf8d0200518bd564"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
registries = "General"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Cobweb]]
deps = ["DefaultApplication", "OrderedCollections", "Scratch"]
git-tree-sha1 = "6665ec6b16446379fb76ad58a2a7b65687c77271"
registries = "General"
uuid = "ec354790-cf28-43e8-bb59-b484409b7bad"
version = "0.7.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "61761f58648aa7217445f24f841839b78c712232"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.3"
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
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "291665b547f137df070e4dd83e432b5fee8cc4a0"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.2"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerCaching]]
deps = ["LMDB", "SHA", "Scratch"]
git-tree-sha1 = "b032c1ce89bc82adf2bef8181c2fb75d6bce7f4a"
registries = "General"
uuid = "9db33cc3-5358-4881-8759-fa4194144afd"
version = "0.5.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.ComputationGraphExplorer]]
deps = ["LinearAlgebra", "Luxor", "Typstry", "UUIDs"]
git-tree-sha1 = "40d5b2258f764301a76a6820e7736522ac7f8910"
registries = "General"
uuid = "c9fc7d07-15e8-4fd8-b152-23c2424d2de2"
version = "0.2.2"
weakdeps = ["NNlib"]

    [deps.ComputationGraphExplorer.extensions]
    ComputationGraphExplorerNNlibExt = "NNlib"

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

[[deps.DefaultApplication]]
deps = ["InteractiveUtils"]
git-tree-sha1 = "c0dfa5a35710a193d83f03124356eef3386688fc"
registries = "General"
uuid = "3f0dd361-4fe0-5fc6-8523-80b14ec94d85"
version = "1.1.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6123b906924c935459624131058fab6a587fc6c1"
registries = "General"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "3.2.3+0"

[[deps.EasyConfig]]
deps = ["JSON3", "OrderedCollections", "StructTypes"]
git-tree-sha1 = "11fa8ecd53631b01a2af60e16795f8b4731eb391"
registries = "General"
uuid = "acab07b0-f158-46d4-8913-50acef6d41fe"
version = "0.1.16"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.ExprTools]]
git-tree-sha1 = "d2e49e7efd29719d6f28b891b0e0e159daa9d2b4"
registries = "General"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.11"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "7bd13840b4148949e290748071bbe6826b3eabcd"
registries = "General"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

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

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

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

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "574e443e2545f8622b716676c9358ca8a7ea57b4"
registries = "General"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.5.16"

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

    [deps.GPUArrays.weakdeps]
    JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "0b2b3e8b8fb7c1f5ddbf498be2eb2b3b37c9c0ed"
registries = "General"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.1"

[[deps.GPUCompiler]]
deps = ["CompilerCaching", "ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "REPL", "ScopedValues", "TOML", "Tracy", "UUIDs", "tree_sitter_gcn_jll", "tree_sitter_llvm_jll", "tree_sitter_ptx_jll", "tree_sitter_spirv_jll"]
git-tree-sha1 = "df0aaa99581473fde47162eca00ce00c6e607903"
registries = "General"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "2.12.0"

    [deps.GPUCompiler.extensions]
    HighlightsExt = "Highlights"

    [deps.GPUCompiler.weakdeps]
    AMDGPU_LLVM_Backend_jll = "cc5c0156-bd05-5a77-8a68-bb0aafb29019"
    Highlights = "eafb193a-b7ab-5a9e-9068-77385905fa72"
    LLVMDowngrader_jll = "f52de702-fb25-5922-94ba-81dd59b07444"
    NVPTX_LLVM_Backend_jll = "ef6e0fe3-e6ef-59c0-bde6-4989574699e0"
    SPIRV_LLVM_Backend_jll = "4376b9bf-cff8-51b6-bb48-39421dff0d0c"
    SPIRV_LLVM_Translator_jll = "4a5d46fc-d8cf-5151-a261-86b458210efb"

[[deps.GPUToolbox]]
git-tree-sha1 = "ac858e66e7bee15ccc5931967260db8fe051e2a6"
registries = "General"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "3.3.3"

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

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
registries = "General"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

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

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
registries = "General"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

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

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
registries = "General"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
registries = "General"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "920671fc8e3be4daf278442bf0bd5cfb46ebac48"
registries = "General"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.43"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

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

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "f5264c787a233f5b5a239c424480712f243be730"
registries = "General"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "10.1.0"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "2226731b1de936c7154f0d19b6a7daa2b717bfea"
registries = "General"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.51+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
registries = "General"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LMDB]]
deps = ["CEnum", "LMDB_jll"]
git-tree-sha1 = "84e5cba87693714bfd67f589a91f24a996ff3e69"
registries = "General"
uuid = "11f193de-5e89-5f17-923a-7d207d56daf9"
version = "3.0.1"

[[deps.LMDB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56b8af9f260b5a5ea4cd3c12cb8faa028cf8fa15"
registries = "General"
uuid = "6206cf0b-f360-5984-af49-5437264c140e"
version = "1.0.0+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
registries = "General"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

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
version = "1.11.104+0"

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d4e20500d210247322901841d4eafc7a0c52642d"
registries = "General"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.13.1+0"

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
weakdeps = ["LaTeXStrings", "MathTeXEngine", "Typstry"]

    [deps.Luxor.extensions]
    LuxorExtLatex = ["LaTeXStrings", "MathTeXEngine"]
    LuxorExtTypstry = ["Typstry"]

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "aa1078778be5a8e5259ff04fbc3d258b3e78d464"
registries = "General"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.9"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.8.13"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "BFloat16s", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "d450844d195714a2d29b38c231193c0297cf90e8"
registries = "General"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.45"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreCUDNNExt = ["EnzymeCore", "CUDA", "cuDNN"]
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibMooncakeCUDAExt = ["Mooncake", "CUDA"]
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NVML]]
deps = ["CEnum", "CUDACore", "GPUToolbox", "Libdl"]
git-tree-sha1 = "1eb041587b2887ee12675c7e20c96c5ed44a9dcf"
registries = "General"
uuid = "611af6d1-644e-4c5d-bd58-854d7d1254b9"
version = "6.4.2"

[[deps.NVPTX_LLVM_Backend_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4e4913b46c28b0e09ecd11a357fb2b8632504d27"
registries = "General"
uuid = "ef6e0fe3-e6ef-59c0-bde6-4989574699e0"
version = "23.1.2+1"

[[deps.NVTX]]
deps = ["JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "a9083c3e469e63cca454d1fc3b19472d9d92c14a"
registries = "General"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.3"
weakdeps = ["Colors"]

    [deps.NVTX.extensions]
    NVTXColorsExt = "Colors"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "af2232f69447494514c25742ba1503ec7e9877fe"
registries = "General"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.2.2+0"

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

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "9510d7008275fc5b33fc72a73f8fddef0b5430c6"
registries = "General"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.11"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

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
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

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
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "ba0dc8a8a67cacac4842631f960c046e4e563675"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.8"

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

[[deps.PlotlyLight]]
deps = ["Artifacts", "Cobweb", "Dates", "Downloads", "EasyConfig", "JSON3", "REPL"]
git-tree-sha1 = "ed95b3125e681e5209221a035cfc46058cfcd88f"
registries = "General"
uuid = "ca7969ec-10b3-423e-8d99-40f33abb42bf"
version = "0.13.1"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "90b41ced6bacd8c01bd05da8aed35c5458891749"
registries = "General"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.7"

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
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "StyledStrings", "Tables"]
git-tree-sha1 = "99fd8f32ab6728ff205cddd24e2d9ea6cbb498d3"
registries = "General"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.5.0"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"
    PrettyTablesWriteDocxExt = "WriteDocx"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    WriteDocx = "d049ceea-54ee-41d7-a26f-ba29db3b6599"
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

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
registries = "General"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
registries = "General"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
registries = "General"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "39e70e0ab5d7f89833a62ab7c79df15d4fc417c1"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.22"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

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
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StringManipulation]]
deps = ["PrecompileTools", "StyledStrings"]
git-tree-sha1 = "9e3a02d73a8f0a9be04e677dbef758aab40f8bea"
registries = "General"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.6.1"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "9537ef82c42cdd8c5d443cbc359110cbb36bae10"
registries = "General"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.21"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
registries = "General"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

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

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
registries = "General"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

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

[[deps.Typst_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "OpenSSL_jll"]
git-tree-sha1 = "0fc7d5d3083a3753abe4b4eb10a87c0146fa5f20"
registries = "General"
uuid = "eb4b1da6-20f6-5c66-9826-fdb8ad410d0e"
version = "0.14.2+0"

[[deps.Typstry]]
deps = ["Artifacts", "Dates", "PrecompileTools", "Typst_jll"]
git-tree-sha1 = "3b7b7e04e2ef20793d3fe40f4cc3e55645f2b033"
registries = "General"
uuid = "f0ed7684-a786-439e-b1e3-3b82803b501e"
version = "0.7.0"
weakdeps = ["LaTeXStrings", "Markdown"]

    [deps.Typstry.extensions]
    LaTeXStringsExtension = "LaTeXStrings"
    MarkdownExtension = "Markdown"

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

[[deps.UnsafeAtomics]]
deps = ["LLVM"]
git-tree-sha1 = "74ac5e0e702972da953ffc981094617070be582d"
registries = "General"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.4.0"
weakdeps = ["GPUCompiler"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsGPUCompilerExt = "GPUCompiler"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.cuBLAS]]
deps = ["Adapt", "BFloat16s", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUArrays", "GPUToolbox", "LLVM", "LinearAlgebra"]
git-tree-sha1 = "f91d19c18de7f2b1c93868229b1a0cf605c857d1"
registries = "General"
uuid = "182d3088-87b7-4494-8cad-fc6afaa545bc"
version = "6.4.2"

    [deps.cuBLAS.extensions]
    EnzymeCoreExt = "EnzymeCore"

    [deps.cuBLAS.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.cuFFT]]
deps = ["AbstractFFTs", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "5cc11e026fef3fc5124179f9e0d299db06910945"
registries = "General"
uuid = "533571aa-0936-420e-b4be-9c66f5f626ca"
version = "6.4.2"

[[deps.cuRAND]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "Random", "Random123", "RandomNumbers"]
git-tree-sha1 = "47438c00d3565a2361ecc3e04e5b23eacb127d76"
registries = "General"
uuid = "20fd9a0b-12d5-4c2f-a8af-7c34e9e60431"
version = "6.4.2"

[[deps.cuSOLVER]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "SparseArrays", "cuBLAS", "cuSPARSE"]
git-tree-sha1 = "2538fd2f19a33366ad26b215dc275e49c831e610"
registries = "General"
uuid = "887afef0-6a32-4de5-add4-7827692ba8fc"
version = "6.4.2"

[[deps.cuSPARSE]]
deps = ["Adapt", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUArrays", "GPUToolbox", "KernelAbstractions", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "63824c10451249430f14435452790ef1d1754521"
registries = "General"
uuid = "b26da814-b3bc-49ef-b0ee-c816305aa060"
version = "6.4.2"

    [deps.cuSPARSE.extensions]
    SparseMatricesCSRExt = "SparseMatricesCSR"

    [deps.cuSPARSE.weakdeps]
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
registries = "General"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.gdk_pixbuf_jll]]
deps = ["Artifacts", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Xorg_libX11_jll", "libpng_jll"]
git-tree-sha1 = "895f21b699121d1a57ecac57e65a852caf569254"
registries = "General"
uuid = "da03df04-f53b-5353-a52f-6a8b0620ced0"
version = "2.42.13+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1210ba774d3427387d307bf1f416d699b7c39417"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.1+0"

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
git-tree-sha1 = "1fcd6cf05d23adab0b5ffc496d020537215751dc"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.59+1"

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

[[deps.tree_sitter_gcn_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7aea9f731552967bda93cb7cbf6925de65bc38d5"
registries = "General"
uuid = "8b5cbfcf-8811-596e-8e87-0e51e05ee4b2"
version = "0.1.0+0"

[[deps.tree_sitter_llvm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2fb7d0e3a8abf532fa922766ca130bf814df1e70"
registries = "General"
uuid = "44208993-ee63-5069-9443-8e43b04a9b30"
version = "1.1.0+0"

[[deps.tree_sitter_ptx_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e6ad264eebe3e34f03bca6b5e811d80e0f4797a2"
registries = "General"
uuid = "71e3f6e6-c059-5e7c-a2f2-d560fdad7ce5"
version = "0.1.0+0"

[[deps.tree_sitter_spirv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d8c03949bab84b23fe0500d89979e786d54a653"
registries = "General"
uuid = "f0e86581-c468-54df-a4de-3266e11a3c86"
version = "0.1.0+0"

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
# ╟─40baa108-eb68-433f-9917-ac334334f198
# ╟─77a7de14-87d2-11ef-21ef-937b8239db5b
# ╟─e46fb3ff-b26f-4efb-aaa1-760e80017797
# ╟─af404768-0663-4bc3-81dd-6931b3a486be
# ╟─2ae277d5-8be4-4ea9-a3fc-8ad601577c3a
# ╟─fa20b8db-9ac7-490d-b8d8-8d57469d24e4
# ╟─586c2c6b-4a53-46d5-924b-c843e4c09859
# ╟─1c44706b-fd7d-4826-84a3-73e2db5adadd
# ╠═45051873-3f0d-49b0-a9a6-bfde240594aa
# ╠═3803804a-f44d-4f56-bd59-fb1401d8fb9e
# ╠═63bdc7f5-89a5-4061-9e42-6588e4cd96c6
# ╠═4174d41c-877b-4878-b76e-442991907af0
# ╠═8b8682e4-8adb-4fb1-a013-054b1d9750d7
# ╠═e211118d-eba8-467e-ae4a-665f1df02934
# ╠═bc93fb96-1097-4b22-a077-558f5662efec
# ╟─85fc455c-36cf-4a4c-aa64-84a827884693
# ╟─cc2a09a1-c949-4b09-816b-b49ba7ca8983
# ╟─277bd2ce-fa7f-4288-be8a-0ddd8f23635c
# ╟─fa5dba01-a3f7-452c-877e-352d578ecf51
# ╟─69c08fab-c317-462c-817c-3f841a8a0941
# ╟─8deca676-8a0b-41eb-b7a0-4d65e1158b0b
# ╟─885bc5c9-aefc-4d8a-a4da-6062c64eaa41
# ╟─d2b8fa5c-c604-4093-a2dd-5c95f2eaa676
# ╟─5aff8e66-787d-4dc5-a9b1-0fdec25ce0f0
# ╟─d1dbdd3f-9782-4fba-8c4e-819f152e6c30
# ╟─83ef86e0-bcfb-42ee-a574-16758606423a
# ╟─90850509-463d-44c7-88ae-4406aebd4be1
# ╟─c4e91a07-2b8d-4f1e-9a63-7d05be21c8f4
# ╟─673c3acc-0009-416a-91bb-f57c1fe8eefc
# ╟─28df733e-7db9-4e78-9121-52d8e6ca7591
# ╟─626abc7c-87ef-4838-9f0a-294cf0a4be6a
# ╟─6c60f9ca-ba04-41e2-9625-c9e10f1a853b
# ╟─1d56075c-e28d-46c9-9a0a-210079172388
# ╟─19578219-c6dd-4322-a2f4-44088ef640af
# ╟─d18fb5c2-6e47-4a90-b3d1-90c7af4e2b16
# ╟─e52c7a3b-8d19-4c60-a7f2-31b6ec9d5a08
# ╟─f9a37c14-5b62-4e8d-96a0-2c41db73e65f
# ╟─7f75e3f3-c4e2-402d-be7b-336a4f65042a
# ╟─bab3a3cb-0ad2-4ea5-a15c-6593fc22e496
# ╟─c73f79c6-a28f-4c7a-89e5-8d70a245a210
# ╟─ac52550e-3287-427f-b957-ac61bc850f4d
# ╟─f4d1ee7c-4a01-4b2d-aa9b-ec41ceb0ad0f
# ╟─e8c60922-5bbf-45b5-8311-18c8f8525623
# ╟─73ba544c-616a-4db1-b91d-0b20a7b8924b
# ╟─2f8baccc-19d1-44d6-b71f-0243fd8696ba
# ╟─11eadcc1-68bf-4ca7-9603-ef46468b4779
# ╟─dc4feb58-d2cf-4a97-aaed-7f4593fc9732
# ╟─74063eb5-be06-466a-a2f1-e266c35295ea
# ╟─607000ef-fb7f-4204-b543-3cb6bb75ed71
# ╟─88534196-9f4a-430c-a534-805177ba718d
# ╟─2b631fcd-2703-42df-8a75-2fdff64b3311
# ╠═9988fc4a-cedc-499b-a334-048cc13de000
# ╠═ceaeb177-7a6a-4062-9659-56bebce0e77b
# ╠═3556d366-0bc7-4239-b4f6-3f9bd28780e0
# ╠═69ae57b4-4e4c-44a2-aca7-d0fff89b9566
# ╠═e50f8f52-a73f-4186-af5e-b4ca2c021142
# ╠═9862c791-31e8-4d59-8610-a929d72ea9c3
# ╟─6bd5ea51-54c4-46ba-8eb3-6427225e5249
# ╟─2e8dd18a-b784-4f7d-ac7d-e1f88ba2ca56
# ╟─ee3299e2-3367-4c68-a050-c26650a97c8c
# ╟─4d4e11a2-b6a1-490f-9c7b-ad617dbb395b
# ╟─df1f29ce-026d-4935-83aa-0d597cb166dd
# ╟─533bf4bc-494b-47c2-91b6-7b30221765f5
# ╟─9a07a949-4ade-4325-8ae6-b05bdbbfae00
# ╟─9661a3b9-ae42-466e-967d-69230b9769b3
# ╟─895f4b81-51fe-4a1c-b958-1294743e3a9b
# ╟─3f1068e0-97e2-444a-961d-2af55a4d797b
# ╟─221a9c57-0072-4fb6-b61f-d7e3ab9c4be2
# ╟─e121f72b-fe6d-491a-ab03-ef92154c61ca
# ╟─54457653-f029-4368-8bc0-15d69f22bfcd
# ╟─8194cce9-115f-44b4-9f28-caa88b764185
# ╟─292b182f-b8a4-4ac6-89ad-ef71d40eca1f
# ╟─e9c14738-2c1f-43d8-94f9-fb5440d74092
# ╟─b92d17a9-8481-458a-bc0a-efb7333cbc6e
# ╟─9527686f-24e1-40bb-9a5d-22575aafec9b
# ╟─cd6d807d-6238-44ce-9267-1614679f527a
# ╟─0980ef52-526d-421d-8c63-04272f16cbdd
# ╟─191f33ab-b535-4cb1-96b3-6b2f10fd633a
# ╟─c1ae773e-0f2d-4c1c-bae2-25f45bb28fbb
# ╟─c62eb6b6-aca2-4c27-ab7e-e3374515d9aa
# ╠═310d6a10-b887-4dd0-ba93-f901726ddbbc
# ╠═c616136c-f29f-41c2-bef0-516c05831748
# ╟─068ef840-04cb-4de4-9512-721f57e5750e
# ╟─7331c447-c0a6-4a47-af00-36efc449d9d9
# ╟─bebe2bb8-cf74-4eaa-85d1-712b990b5985
# ╟─347a1500-43aa-4078-9ea3-480bc8a4c291
# ╟─29287c62-e892-448f-a9d5-12785ae4a02f
# ╟─c733ca7e-b57e-4218-9bd4-238ab5749143
# ╟─5f6529a1-4ace-4dd0-a7e2-f51070eab695
# ╟─33bdaa23-707e-4227-b936-c5d7aaf2c48e
# ╟─402f0d1d-0a3f-404a-8019-afc92d154c85
# ╟─e9d6af05-63f7-4d31-85e2-b0233df23608
# ╟─802edb3a-4809-4c50-920b-25f7bdc255dd
# ╟─98db9022-f8ff-4af3-9c81-89cf09771928
# ╟─7921c4c6-56b8-4c6f-a9be-cd1d9984680b
# ╟─edad5700-9b04-47c6-90f8-b6bac0897340
# ╟─8c202da6-1e13-43b8-a22b-94badcef2934
# ╟─40ed6c94-d2f9-4225-80b3-9060f04f8971
# ╟─1994bf51-adf1-4b07-ab4c-f47552d90826
# ╟─9ca3b930-3d9d-460c-9f20-e3c135477b05
# ╟─e27db9a0-09ff-4ee5-a807-d98933b6bcf1
# ╟─95eb9960-89f0-4edc-8943-77a75bce2b80
# ╟─9afd31c9-e938-417a-8c3f-e0d1ba88f95b
# ╟─f010e781-f41e-4861-af1c-32cf5a76ce4d
# ╟─906e5199-f2d2-4816-a195-6d2b1dee9403
# ╠═0a9d0913-a1bb-4f47-8b85-db8714d0f6ab
# ╠═db7aaa4c-5568-4619-a9c4-00c7916c321a
# ╠═6cfc019e-7b97-4bf1-b1f5-de9468b9ab4e
# ╠═f5d3714d-3900-4dbe-9079-978a44584d1d
# ╠═0bcadb3a-4880-4e6c-bccb-b09df8ad8fa3
# ╠═2fcf25d2-fd51-4c13-b57c-86236aceead2
# ╟─6ddc06c0-3f5d-4cc9-8060-dd6997e0f662
# ╟─722ad63a-c2ac-4ed6-b268-41d0f8b745f1
# ╠═b5c3e2ef-3d47-4f44-b968-d04734be2f16
# ╠═a580ef44-234a-4ed1-b007-920651415427
# ╟─0e13e63d-fd08-4cc1-aa37-851c537afbef
# ╠═2adc9595-8829-4d35-be90-a7718c2e7ce7
# ╠═53b21ec0-28e9-46cd-a92e-8afc189c3a11
# ╠═778c40ff-4c9e-42fb-92a6-1e376837f6ef
# ╠═9b4a78d8-e6da-41dd-b922-b35c895eee1a
# ╟─43d2559f-8902-4c54-8fdf-cb268b6f868c
# ╟─4317bba0-723f-4cdc-9d52-67033540a8d2
# ╠═35f8cf4f-3fcb-4e27-9462-244406d7800e
# ╠═87c6a5bc-82bf-44a5-b4d6-6d50285348c0
# ╟─17c91ea8-acb7-4bbd-b0b0-0f8193f45303
# ╠═85303791-bdc4-468a-bc40-48ef2a186282
# ╟─6ded46e6-1c89-4875-addb-8c709e949bb1
# ╟─2ca19ff6-ec22-4327-aea2-80bdca55ccef
# ╟─9bbbda1f-74a6-458b-a084-9d034d6c291f
# ╟─03b6aa6d-7517-4906-9430-302516d0653b
# ╟─78d8c8c9-568d-472a-9f06-a50b1cf2384b
# ╟─f23ca90f-b567-4257-ae41-ec15c57c1f3f
# ╟─7d79ff81-59e0-41f0-b2fe-70b41f44591f
# ╟─da5895e7-af99-46ff-9f53-36529d1ca456
# ╟─9415a6ed-c05e-4487-b0be-f342ec7424cd
# ╟─4d76422e-711c-4f3e-87ab-0ce851bac064
# ╟─3a2132e7-7d69-42d7-89d3-b2d7679ad74f
# ╟─24d52ef8-927e-46c8-b455-74ccdb33d3ca
# ╟─4de9baec-f444-47b8-b9e6-7f3d9e9609b1
# ╟─5a79c09e-2a33-43d1-a5dc-caba3db467dd
# ╟─16f611e8-8e41-43f3-830c-9976cb720b9f
# ╟─fa6dd3f7-7b57-483b-ba0f-90c9bb7bb6a6
# ╟─200e2b1b-065a-4d60-b5fd-86700e4c811a
# ╟─e4a9c57b-c811-428b-a6b8-191b78d5f361
# ╟─c1da4130-5936-499f-bb9b-574e01136eca
# ╟─5b85a063-cf0f-4afa-89f4-420d7350ecc3
# ╟─b16f6225-1949-4b6d-a4b0-c5c230eb4c7f
# ╠═f1ba3d3c-d0a5-4290-ab73-9ce34bd5e5f6
# ╠═b7d4a91e-3c52-4f86-9a0d-5e8c1f2b6d47
# ╠═a06be2d9-73c1-4f85-b2e7-18d3c5a90f47
# ╠═d26cba1c-27d8-4ee7-99e5-6c0e0fc6b6e5
# ╠═7f459370-eb6f-4d82-bf99-1528e136ca5c
# ╟─e8b1c40d-27a6-4f39-b95e-6d3a0f81c72b
# ╟─b3c8d70e-9a41-4d26-85fb-6e02f19ca4d3
# ╟─c71e4f82-0d35-49ba-97c6-84a1bd50e739
# ╟─d492a165-3e70-4c18-b0d9-57fc2e8a1b96
# ╟─cbfc0129-9361-4edb-a467-1456a1f3aeae
# ╟─81deb227-a822-4857-a584-a51cc8ff51f4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
