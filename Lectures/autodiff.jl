### A Pluto.jl notebook ###
# v0.20.21

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
using Plots, PlutoUI, PlutoUI.ExperimentalLayout, HypertextLiteral, PlutoTeachingTools, DataFrames, MLDatasets, Statistics, CUDA, OneHotArrays

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
md"# Reverse differentiation"

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
& \mathrel{\raise{0.19ex}{\scriptstyle+}} = r_1 \cdot \partial f_3\\
& \mathrel{\raise{0.19ex}{\scriptstyle+}} = r_1 \cdot
\begin{bmatrix}
	\partial_1 f_3 & \partial_2 f_3
\end{bmatrix}\\
& \mathrel{\raise{0.19ex}{\scriptstyle+}} =
\begin{bmatrix}
	r_1 \cdot\partial_1 f_3 & r_1 \cdot\partial_2 f_3
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
\end{bmatrix} & \mathrel{\raise{0.19ex}{\scriptstyle+}} = \partial f_3^* \cdot r_1\\
& \mathrel{\raise{0.19ex}{\scriptstyle+}} =
\begin{bmatrix}
	\partial_1 f_3\\ \partial_2 f_3
\end{bmatrix} \cdot r_1\\
& \mathrel{\raise{0.19ex}{\scriptstyle+}} =
\begin{bmatrix}
	\partial_1 f_3 \cdot r_1 \\ \partial_2 f_3 \cdot r_1
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
md"## Expression graph"

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

# ╔═╡ 607000ef-fb7f-4204-b543-3cb6bb75ed71
let
	x = range(-1, stop = 1, length = 11)
	p = plot(x, abs, label = "|x|")
	for λ in range(0, stop = 1, length = 11)
		plot!([0, λ/2 - (1 - λ)/2], [0, -1/2], color = :orange, arrow = Plots.arrow(:closed), label = "")
	end
	p
end

# ╔═╡ ceaeb177-7a6a-4062-9659-56bebce0e77b
abs_bis(x) = ifelse(x > 0, x, -x)

# ╔═╡ e50f8f52-a73f-4186-af5e-b4ca2c021142
abs(Dual(0, 1))

# ╔═╡ 9862c791-31e8-4d59-8610-a929d72ea9c3
abs_bis(Dual(0, 1))

# ╔═╡ e121f72b-fe6d-491a-ab03-ef92154c61ca
md"""
# Neural network

Two equivalent approaches, ``b_k`` is a **column** vector, ``S_i, X, W_i, Y`` are matrices.
"""

# ╔═╡ b92d17a9-8481-458a-bc0a-efb7333cbc6e
hbox([md"""
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
	Div(html" ", style = Dict("flex-grow" => "1")),
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
"""])

# ╔═╡ 9527686f-24e1-40bb-9a5d-22575aafec9b
md"## Evaluation"

# ╔═╡ 29287c62-e892-448f-a9d5-12785ae4a02f
md"""## Matrix multiplication (Vectorized way)

Useful: ``\text{vec}(AXB) = (B^\top \otimes A) \text{vec}(X)``
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

The adjoint of a linear map ``A`` for a given scalar product ``\langle \cdot, \cdot \rangle`` is the linear map ``A^*`` such that
```math
\forall x, y, \qquad \langle A(x), y \rangle = \langle x, A^*(y) \rangle.
```
For the scalar product
```math
\langle X, Y \rangle
=
\sum_{i,j} X_{ij} Y_{ij}
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

# ╔═╡ 2202f572-8a5f-4c11-a14f-53cfa161e8e2
wine = MLDatasets.Wine(; as_df = false)

# ╔═╡ f5d3714d-3900-4dbe-9079-978a44584d1d
function normalise(x)
  μ = Statistics.mean(x, dims=2)
  σ = Statistics.std(x, dims=2, mean=μ)
  return (x .- μ) ./ σ
end

# ╔═╡ 0bcadb3a-4880-4e6c-bccb-b09df8ad8fa3
X = Float32.(normalise(wine.features))

# ╔═╡ 2fcf25d2-fd51-4c13-b57c-86236aceead2
y = Float32.(wine.targets .- 2)

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

# ╔═╡ 87c6a5bc-82bf-44a5-b4d6-6d50285348c0
@time reverse_diff(W, X, y)

# ╔═╡ 17c91ea8-acb7-4bbd-b0b0-0f8193f45303
md"## 🚀 GPU acceleration ⚡"

# ╔═╡ 2ca19ff6-ec22-4327-aea2-80bdca55ccef
h_slider = @bind h Slider(10:1000, default = 16, show_value = true);

# ╔═╡ 722ad63a-c2ac-4ed6-b268-41d0f8b745f1
md"`h` = $(h_slider)"

# ╔═╡ b5c3e2ef-3d47-4f44-b968-d04734be2f16
W = [rand(Float32, h, size(X, 1)), rand(Float32, size(y, 1), h)]

# ╔═╡ a580ef44-234a-4ed1-b007-920651415427
sum((W[2] * tanh.(W[1] * X) - y).^2) / size(y, 2)

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
* Given ``v \in \mathbb{R}^{d_{k-1}}``, by the product ``(\partial^2 f_k \cdot v) \in \mathbb{R}^{d_k \times d_{k-1}}`` , we denote the contraction of the the 3rd (or 2nd since the tensor is symmetric over its last 2 dimensions) dimension:
```math
(\partial^2 f_k \cdot v)_{ij} = \sum_{l = 1}^{d_{k-1}} (\partial^2 f_k)_{ijl} \cdot v_l
```
* Given ``u \in \mathbb{R}^{d_k}``, by the product ``(u \cdot \partial^2 f_k) \in \mathbb{R}^{d_{k-1} \times d_{k-1}}`` , we denote the contraction of the the 1st dimension.
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
* Second part: ``\partial s_3/\partial s_1 \mathrel{\raise{0.19ex}{\scriptstyle+}} = 1``
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

# ╔═╡ 74063eb5-be06-466a-a2f1-e266c35295ea
qa(md"Is the function ``|x|`` is differentiable at ``x = 0`` ?.", md"No, if we approach from the left (that is, ``x < 0``, the function is ``-x``), then the derivative is ``-1``.
If we approach from the right (that is, ``x > 0``, the function is ``x``), then the derivative is ``1``.
There is no valid gradient!")

# ╔═╡ 88534196-9f4a-430c-a534-805177ba718d
qa(md"What about returning a convex combination of the derivative from the left and right ?", md"Any number between ``-1`` and ``1`` is a valid **subgradient**!
Whereas the gradient is the normal to the **unique** tangent, the subgradient is an element of the **tangent cone**, depicted below. For convex functions, the notion of subgradient appropriately generalizes the notion of gradient for nonsmooth functions.

Note that the notion of subgradient is not defined for nonconvex functions. So we may say that we compute the local subgradient of some local nonsmooth ``f_i`` but we cannot deduce from it that the resulting vector is a subgradient of ``f`` if ``f`` is nonconvex.")

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
r_2 \cdot J_2
& = \text{Dual}(
\frac{\partial f}{\partial s_2} \cdot J_2,
\frac{\partial^2 f}{\partial s_2 \partial x_j} \cdot J_2 +
\frac{\partial f_1}{\partial x_i} \cdot H_{2j})\\
& = \text{Dual}(
\frac{\partial f}{\partial s_1},
\frac{\partial^2 f}{\partial s_1 \partial x_j})
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
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CUDA = "~5.9.0"
DataFrames = "~1.8.1"
HypertextLiteral = "~0.9.5"
MLDatasets = "~0.7.18"
OneHotArrays = "~0.2.10"
Plots = "~1.41.1"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.72"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "aa99bcff8430e0d486a62f08c13fbd4562a1a179"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "29bb0eb6f578a587a49da16564705968667f5fa8"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.2"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.AtomsBase]]
deps = ["LinearAlgebra", "PeriodicTable", "Preferences", "Printf", "Requires", "StaticArrays", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "5f76425eb977584353191c41d739e7783f036b90"
uuid = "a963bdd2-2df7-4f54-a1ee-49d51e6be12a"
version = "0.5.1"

    [deps.AtomsBase.extensions]
    AtomsBaseAtomsViewExt = "AtomsView"

    [deps.AtomsBase.weakdeps]
    AtomsView = "ee286e10-dd2d-4ff2-afcb-0a3cd50c8041"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "3b642331600250f592719140c60cf12372b82d66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.5.1"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BufferedStreams]]
git-tree-sha1 = "6863c5b7fc997eadcabdbaf6c5f201dc30032643"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics", "demumble_jll"]
git-tree-sha1 = "d5840b32b52a201ca90ac9d538c1d3a1641bfa2d"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.9.0"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SparseMatricesCSRExt = "SparseMatricesCSR"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8c4f340dd6501a93c4b99b690797772e4a203099"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.2.1+0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e6a1d9f5518122c186fd27786b61d2053cfa1b0c"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.0.1+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "f9a521f52d236fe49f1028d69e549e7f2644bb72"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "1.0.0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "e24c6de116c0735c37e83b8bc05ed60d4d359693"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.19.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Chemfiles]]
deps = ["AtomsBase", "Chemfiles_jll", "DocStringExtensions", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "8d4217428ee7c64605d1217a8ea810436fd03742"
uuid = "46823bd8-5fb3-5f92-9aa0-96921f3dd015"
version = "0.10.43"

[[deps.Chemfiles_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f3743181e30d87c23d9c8ebd493b77f43d8f1890"
uuid = "78a364fa-1a3c-552a-b4bb-8fa0f9c1fcca"
version = "0.10.4+0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "980f01d6d3283b3dbdfd7ed89405f96b7256ad57"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "2.0.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

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

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "Scratch", "p7zip_jll"]
git-tree-sha1 = "8ae085b71c462c2cb1cfedcb10c3c877ec6cf03f"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.13"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

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
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7bb1361afdb33c7f2b085aa49ea8fe1b0fb14e58"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.1+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "83dc665d0312b41367b7263e8a4d172eac1897f4"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3a948313e7a41eb1db7a1e733e6335f17b4ab3c4"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "7.1.1+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "Statistics"]
git-tree-sha1 = "8ddb438e956891a63a5367d7fab61550fc720026"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.6"
weakdeps = ["JLD2"]

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "9a8b92a457f55165923fcfe48997b7b93b712fca"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.7.2"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "5bfe837129bf49e2e049b4f1517546055cc16a93"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "0.3.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "1828eb7275491981fa5f1752a5e126e8f26f8741"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.17"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "27299071cc29e409488ada41ec7643e0ab19091f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.17+0"

[[deps.GZip]]
deps = ["Libdl", "Zlib_jll"]
git-tree-sha1 = "0085ccd5ec327c077ec5b91a5f937b759810ba62"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.6.2"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "50c11ffab2a3d50192a228c313f05b5b5dc5acb2"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.0+0"

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "e94f84da9af7ce9c6be049e9067e511e17ff89ec"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.6+0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e6fe50ae7f23d171f44e311c2960294aaa0beb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.19"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "3d468106a05408f9f7b6f161d9e7715159af247b"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.12.2+0"

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
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

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

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues", "TranscodingStreams"]
git-tree-sha1 = "d97791feefda45729613fafeccc4fbef3f539151"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.15"

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

    [deps.JLD2.weakdeps]
    UnPack = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "277779adfedf4a30d66b64edc75dc6bb6d52a16e"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.6"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "83c617e9e9b02306a7acab79e05ec10253db7c87"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.38"

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
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "ce8614210409eaa54ed5968f4b50aa96da7ae543"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.4"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8e76807afb59ebb833e9b131ebf1a8c006510f33"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.38+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.11.1+1"

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

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d2bc4e1034b2d43076b50f0e34ea094c2cb0a717"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.9.1+6"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3acf07f130a76f87c041cfb2ff7d7284ca67b072"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "e24491cb83551e44a69b9106c50666dea9d953ab"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.4.4"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "1d2dd9b186742b0f317f2530ddcbf00eebb18e96"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.7"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLDatasets]]
deps = ["CSV", "Chemfiles", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "FixedPointNumbers", "GZip", "Glob", "HDF5", "ImageShow", "JLD2", "JSON3", "LazyModules", "MAT", "MLUtils", "NPZ", "Pickle", "Printf", "Requires", "SparseArrays", "Statistics", "Tables"]
git-tree-sha1 = "361c2692ee730944764945859f1a6b31072e275d"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.7.18"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "d72d0ecc3f76998aac04e446547259b9ae4c265f"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.1+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e214f2a20bdd64c04cd3e4ff62d3c9be7e969a59"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.4+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3cce3511ca2c6f87b19c34ffc623417ed2798cbd"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.10+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.5.20"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "eb6eb10b675236cee09a81da369f94f16d77dc2f"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.31"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NPZ]]
deps = ["FileIO", "ZipFile"]
git-tree-sha1 = "60a8e272fe0c5079363b28b0953831e2dd7b7e6f"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.3"

[[deps.NVTX]]
deps = ["Colors", "JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "6b573a3e66decc7fc747afd1edbf083ff78c813a"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.1"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "af2232f69447494514c25742ba1503ec7e9877fe"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.2.2+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "bfe8e84c71972f77e775f75e6d8048ad3fdbe8bc"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.10"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "ec764453819f802fc1e144bfe750c454181bd66d"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.8+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1f7f9bbd5f7a2e5a9f7d96e51c9754454ea7f60b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.4+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.PeriodicTable]]
deps = ["Base64", "Unitful"]
git-tree-sha1 = "238aa6298007565529f911b734e18addd56985e1"
uuid = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"
version = "1.2.1"

[[deps.Pickle]]
deps = ["BFloat16s", "DataStructures", "InternedStrings", "Mmap", "Serialization", "SparseArrays", "StridedViews", "StringEncodings", "ZipFile"]
git-tree-sha1 = "b10600c3a4094c9a35a81c4d109ad5da8a99875f"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.6"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "12ce661880f8e309569074a61d3767e5756a199f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.1"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoLinks", "PlutoUI"]
git-tree-sha1 = "8252b5de1f81dc103eb0293523ddf917695adea1"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.3.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "f53232a27a8c1c836d3998ae1e17d898d4df2a46"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.72"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

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

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "6b8e2f0bae3f678811678065c09571c1619da219"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.1.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "34f7e5d2861083ec7596af8b8c092531facf2192"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "8f528b0851b5b7025032818eb5abbeb8a736f853"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "2d155af8d27cc03e39771aac4468695bcedb6ca7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.10.0"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

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
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "425158c52aa58d42593be6861befadf8b2541e9b"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.4.1"
weakdeps = ["CUDA", "PtrArrays"]

    [deps.StridedViews.extensions]
    StridedViewsCUDAExt = "CUDA"
    StridedViewsPtrArraysExt = "PtrArrays"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "b765e46ba27ecf6b44faf70df40c57aa3a547dcb"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

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

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

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

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "cec2df8cf14e0844a8c4d770d12347fda5931d72"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.25.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.UnitfulAtomic]]
deps = ["Unitful"]
git-tree-sha1 = "903be579194534af1c4b4778d1ace676ca042238"
uuid = "a7773ee8-282e-5fa2-be4e-bd808c38a91a"
version = "1.0.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4909eb8f1cbf6bd4b1c30dd18b2ead9019ef2fad"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.18.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa23f01927b2dac46db77a56b31088feee0a491"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.4+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
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
# ╟─673c3acc-0009-416a-91bb-f57c1fe8eefc
# ╟─28df733e-7db9-4e78-9121-52d8e6ca7591
# ╟─626abc7c-87ef-4838-9f0a-294cf0a4be6a
# ╟─6c60f9ca-ba04-41e2-9625-c9e10f1a853b
# ╟─7f75e3f3-c4e2-402d-be7b-336a4f65042a
# ╟─bab3a3cb-0ad2-4ea5-a15c-6593fc22e496
# ╟─c73f79c6-a28f-4c7a-89e5-8d70a245a210
# ╟─ac52550e-3287-427f-b957-ac61bc850f4d
# ╟─f4d1ee7c-4a01-4b2d-aa9b-ec41ceb0ad0f
# ╟─e8c60922-5bbf-45b5-8311-18c8f8525623
# ╟─73ba544c-616a-4db1-b91d-0b20a7b8924b
# ╟─2f8baccc-19d1-44d6-b71f-0243fd8696ba
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
# ╟─e121f72b-fe6d-491a-ab03-ef92154c61ca
# ╟─b92d17a9-8481-458a-bc0a-efb7333cbc6e
# ╟─9527686f-24e1-40bb-9a5d-22575aafec9b
# ╟─cd6d807d-6238-44ce-9267-1614679f527a
# ╟─29287c62-e892-448f-a9d5-12785ae4a02f
# ╟─c733ca7e-b57e-4218-9bd4-238ab5749143
# ╟─5f6529a1-4ace-4dd0-a7e2-f51070eab695
# ╟─33bdaa23-707e-4227-b936-c5d7aaf2c48e
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
# ╠═2202f572-8a5f-4c11-a14f-53cfa161e8e2
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
# ╟─cbfc0129-9361-4edb-a467-1456a1f3aeae
# ╟─81deb227-a822-4857-a584-a51cc8ff51f4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
