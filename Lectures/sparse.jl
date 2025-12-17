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

# ╔═╡ 75597493-3e69-437f-9408-b43a89b35559
using PlutoUI, PlutoUI.ExperimentalLayout, HypertextLiteral, PlutoTeachingTools

# ╔═╡ 6bd04f2f-ed5f-4ecf-91dd-8868ec5b83e1
using SparseArrays, Images, SparseMatrixColorings

# ╔═╡ 4f8de444-dcb4-411a-b087-6948699614e0
using Graphs, GraphPlot, LinearAlgebra, StableRNGs

# ╔═╡ 39e06fde-d734-11f0-a308-89fba7e8abd6
@htl("""
<p align=center style=\"font-size: 40px;\">Sparse AD</p><p align=right><i>Benoît Legat</i></p>
$(PlutoTeachingTools.ChooseDisplayMode())
$(PlutoUI.TableOfContents(depth=1))
""")

# ╔═╡ e2eb1894-06ad-47a0-9ee0-c65b24a669b5
md"""
* [What color is your Jacobian ? Graph Coloring for  Computing Derivatives](https://epubs.siam.org/doi/10.1137/S0036144504444711)
* [An Illustrated Guide to Automatic Sparse Differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/)
* [Revisiting Sparse Matrix Coloring and Bicoloring](https://arxiv.org/abs/2505.07308)
"""

# ╔═╡ 2578b84a-365a-47b6-b58f-1696517c5d02
md"""
# Sparse Jacobian

Suppose the Jacobian is **sparse**.

* The *sparsity pattern* (rows and columns of nonzeros) is **known**
* You want to determine the **value** of these nonzeros with AD
"""

# ╔═╡ 874b6593-7310-4788-beb0-2f18c0ca09a7
md"## Three columns in just one JVP"

# ╔═╡ aeb8b749-006a-400f-9b7d-05333fd7214f
md"## Whole Jacobian in just two JVP"

# ╔═╡ a7e8569e-2fed-4069-aa84-ced67849ab15
md"## The same applies to reverse mode"

# ╔═╡ 18594f1f-6036-4a95-8682-cf415c998311
md"# Sparsity detection of Jacobian"

# ╔═╡ 0a91e27f-3cea-415e-aed4-123482da375a
md"Simple [operator overleading implementation](https://adrianhill.de/SparseConnectivityTracer.jl/dev/internals/how_it_works/)"

# ╔═╡ 97c94c5a-fab5-4226-b803-10c85a298f2a
struct MyGradientTracer
    indexset::Set{Int}
end

# ╔═╡ 8e2045f3-ee09-4439-b504-d2ca9f58a3dd
Base.:+(a::MyGradientTracer, b::MyGradientTracer) = MyGradientTracer(union(a.indexset, b.indexset))

# ╔═╡ fa2a8104-2a3f-438c-bc63-4236a9f5ebeb
Base.:*(a::MyGradientTracer, b::MyGradientTracer) = MyGradientTracer(union(a.indexset, b.indexset))

# ╔═╡ 9df9c76a-48af-4ca6-83f1-b8798d6d4e34
Base.:/(a::MyGradientTracer, b::Number) = a

# ╔═╡ b18a3bf6-102e-49f5-af8c-f1c35d2e2a8b
md"The sign function is constant over ``x > 0`` and ``x < 0`` so its output carries **no** variables"

# ╔═╡ f8e9d672-e6ad-4910-8307-2b56616b11bc
Base.sign(x::MyGradientTracer) = MyGradientTracer(Set()) # return empty index set

# ╔═╡ 7fec005c-11b9-4a75-a93c-195d456db190
md"## Scalar example"

# ╔═╡ d46dcdb3-ac4c-4ba9-ac48-6d6a693630d3
f(x) = x[1] + x[2]*x[3] * sign(x[4])

# ╔═╡ 2e3cdd4b-9d04-477c-8c6a-fec6ec846919
xtracer = [
    MyGradientTracer(Set(1)),
    MyGradientTracer(Set(2)),
    MyGradientTracer(Set(3)),
    MyGradientTracer(Set(4)),
]

# ╔═╡ 1167ae71-b63d-4ccc-a120-31c8483eca85
ytracer = f(xtracer)

# ╔═╡ 45f4edeb-d934-413f-9f45-568465111b6e
md"## Sparsity detection with multiple outputs"

# ╔═╡ a3bf5961-323b-4adc-99a0-9a456a0bfd48
F(x) = [x[1]*x[2]+sign(x[3]), sign(x[3]) * x[4]/2]

# ╔═╡ a546edb2-6a72-4c5f-95f7-cc7709bbb505
F(xtracer)

# ╔═╡ 1617c79d-3abe-4ffc-9ee1-3312f2b28eb5
md"""
## Coloring problems

Given a graph ``G(V,E)``,
* Nodes ``u, v`` are distance ``k`` neighbors if there exists a path from ``u`` to ``v`` of length at most ``k``.
* A *distance-*``k`` coloring : mapping ``\phi:V \to \{1, \ldots, p\}`` such that ``\phi(u) \neq \phi(v)`` whenever ``u, v`` are distance-``k`` neighbors.
* ``k``*-chromatic number* ``\xi_k(G)`` : mininum ``p`` such that ``\exists`` distance-``k`` coloring with ``p`` colors.
* Distance-``k`` coloring problem : Find distance-``k`` coloring with fewest colors.
* For every fixed integer ``k \ge 1``, the distance-``k`` graph coloring problem [is NP-hard](https://doi.org/10.1137/S089548019120016X).

See [Section 2.1, 2.2, 3.2 of What color is your Jacobian ?](https://epubs.siam.org/doi/10.1137/S0036144504444711)
"""

# ╔═╡ 76162106-fb90-4883-b2d2-b87e1419c072
md"""
## Formulation as coloring problem
"""

# ╔═╡ 5892dfbb-b375-407b-a0a7-da66adb71b67
md"""
[Theorem 3.5](https://epubs.siam.org/doi/10.1137/S0036144504444711) A coloring of the columns is distance-2 in the bipartite graph iff columns of the same color are structurally orthogonal.

[Lemma 3.7](https://epubs.siam.org/doi/10.1137/S0036144504444711) The column intersection graph is the square of the biparted graph restricted to its column vertices.

[Lemma 2.1](https://epubs.siam.org/doi/10.1137/S0036144504444711) A coloring is distance-``k`` in ``G`` iff it is distance-1 in ``G^k``.
"""

# ╔═╡ e882ebee-36ad-43d9-8ab7-20f7e8b2d9fe
md"""
## Example

Consider the following sparsity pattern of the Jacobian matrix:
"""

# ╔═╡ f29078e8-242a-4b88-881f-781829c16ed5
md"Taken from a tutorial of the [SparseMatrixColorings' doc](https://gdalle.github.io/SparseMatrixColorings.jl/stable/tutorial/#Coloring-results)"

# ╔═╡ 5a6aec48-54fb-4d88-a8a1-b06a2bb99108
S = sparse([
    0 0 1 1 0 1
    1 0 0 0 1 0
    0 1 0 0 1 0
    0 1 1 0 0 0
])

# ╔═╡ 19dd6563-1a4e-479f-881d-9bcbc720ea36
md"""
# What color is your Hessian ?

How many HVP do we need to find the values of this sparse **symmetric** matrix ?
"""

# ╔═╡ 32440932-322a-46ce-911d-5fa6a1136bca
ijkl = sparse([
	1 2 3 0
	2 4 0 5
	3 0 6 0
	0 5 0 7
])

# ╔═╡ c2e4982a-3798-4c45-8831-5a76b3b29f6e
md"[Section 2.4](https://epubs.siam.org/doi/10.1137/S0036144504444711) Consider the *adjacency graph* ``G`` of a matrix ``A`` be the graph whose adjacency matrix has same sparsity pattern as ``A``. So ``i`` and ``j`` are adjacent iff ``a_{ij}`` is nonzero."

# ╔═╡ e6697119-2801-48e2-b978-facc00af636d
md"""
## Need 3 colors for a path of 4 vertices

```math
D^\top = \begin{bmatrix}
  \color{yellow}\mathbf{1} &  &  & \\
   & \color{pink}\mathbf{1} & \color{pink}\mathbf{1} & \\
   &  &  & \color{blue}\mathbf{1}
\end{bmatrix} \qquad\qquad\qquad
AD = \begin{bmatrix}
  \color{yellow}a_1 & \color{pink}a_2 + a_3 &\\
  \color{yellow}a_2 & \color{pink}a_4 & \color{blue}a_5\\
  \color{yellow}a_3 & \color{pink}a_6 &\\
   & a_5 & \color{blue}a_7
\end{bmatrix}
```
"""

# ╔═╡ a7b85908-404b-475b-9875-c1190de5ff71
md"""
* Each off-diagonal entry ``a_{uv}`` corresponds to an edge in the graph.
* The edge links nodes ``u, v`` of 2 different colors ``\phi(u)`` and ``\phi(v)``
* If ``u`` is not adjacent to any other nodes of color ``\phi(v)`` then ``a_{uv}`` is the **only term** at the row ``u`` of the HVP of color ``\phi(v)``
"""

# ╔═╡ 760ef943-e9d9-4cdc-8977-932cf6fcb78a
md"## Coloring of a star"

# ╔═╡ 285d3797-5086-4adc-839c-2204128585e1
md"""
* For any pink node ``u > 1``, the offdiagonal entry ``a_{u,1}`` can be obtained from the yellow HVP.
"""

# ╔═╡ 0445aae6-2f7e-42b8-b30b-f31298cdeed0
md"`n` = $(@bind star_n Slider(3:10, default = 6, show_value = true))"

# ╔═╡ 95d7e9cd-b4a1-4205-b7d4-8c10916be8a2
star = let
	I = collect(1:star_n)
	J = ones(Int, star_n)
	for i in 2:star_n
		push!(I, 1)
		push!(J, i)
		push!(I, i)
		push!(J, i)
	end
	sparse(I, J, ones(Int, length(I)))
end;

# ╔═╡ a4ecf92d-6c45-4112-8f0c-06a6d227cd84
md"""## Star coloring

[Definition 4.5](https://epubs.siam.org/doi/10.1137/S0036144504444711) A mapping ``\phi: V \to \{1, \ldots, p\}`` is a *star coloring* if

1. ``\phi`` is a distance-1 coloring
2. every **path of 4 vertices** uses at least 3 colors

[Theorem 4.6](https://epubs.siam.org/doi/10.1137/S0036144504444711)
Let ``A`` be a symmetric matrix with **nonzero diagonal elements**,
``G`` be its adjacency matrix. A mapping ``\phi`` is a star coloring of the adjacency graph iff it induces a symmetrically structurally orthogonal partition of the columns of ``A``.

Name star coloring comes from the fact that the subgraph induced by any pair of colors is a star.
"""

# ╔═╡ 29d9f355-26aa-4714-9f9e-3d020016662e
md"## Small Example"

# ╔═╡ d0ba0c62-c788-4871-959c-fe7044519849
md"""
# Less colors but nontrivial substitution

```math
\begin{bmatrix}
  a_{1} & a_{2} & &\\
  a_2 & a_3 & a_4 &\\
  & a_4 & a_5 & a_6\\
  & & a_6 & a_7
\end{bmatrix}
```
With star coloring, 3 colors:
"""

# ╔═╡ fab75cf6-21cc-4001-84b4-9e66f55d5a6a
ijkl2 = sparse([
	1 2 0 0
	2 3 4 0
	0 4 5 6
	0 0 6 7
]);

# ╔═╡ b33bf212-5f88-47ab-ba10-a2815d17f235
md"""
## 2 colors with substitutions

With 2 colors, our two forward tangents are
```math
D^\top = \begin{bmatrix}
  1 & 0 & 1 & 0\\
  0 & 1 & 0 & 1
\end{bmatrix} \qquad\qquad\qquad
AD = \begin{bmatrix}
  a_1 & a_2\\
  a_2 + a_4 & a_3\\
  a_5 & a_4 + a_6\\
  a_6 & a_7
\end{bmatrix}
```
"""

# ╔═╡ e3ae5697-6b0b-4d5f-99a0-d579b6eff655
md"""## Acyclic coloring

[Definition 6.3](https://epubs.siam.org/doi/10.1137/S0036144504444711) A mapping ``\phi: V \to \{1, \ldots, p\}`` is an *acyclic coloring* if

1. ``\phi`` is a distance-1 coloring
2. every **cycle** uses at least 3 colors

[Theorem 4.6](https://epubs.siam.org/doi/10.1137/S0036144504444711)
Let ``A`` be a symmetric matrix with **nonzero diagonal elements**,
``G`` be its adjacency matrix. A mapping ``\phi`` is an acyclic coloring of the adjacency graph iff it induces a substitutable partition of the columns of ``A``.

Name acyclic coloring comes from the fact that the subgraph induced by any pair of colors is a forest.
"""

# ╔═╡ f2418632-3261-4acb-8cc6-df333b5480a5
md"## Illustrative example"

# ╔═╡ 2bd9ea13-362e-47ac-b26d-8e36d33e04ea
md"For each color ``c``, we consider the submatrix ``B_c = A_{:,I}`` where ``I = \{i \mid \phi(i) = c\}`` is the set of columns corresponding to color-``c`` nodes."

# ╔═╡ e3dec4ca-69c0-4955-9303-5f8a239546f2
md"Let ``h_c`` be the sum of the columns of ``B_c`` hence the result of the corresponding HVP."

# ╔═╡ d5b7f51b-e451-40a7-aa41-a8006decba33
md"""## Red-Blue subgraph
To compute an off-diagonal entry ``A_{ij}``, consider the subgraph induced by the two colors ``\phi(i)`` and ``\phi(j)``. The entry will be computed, from ``B_{\phi(i)}`` and ``B_{\phi(j)}``, possible after substitution.

Consider below left ``B_\text{red}`` and right ``B_\text{blue}``. The rows corresponding to diagonal entries have been omitted as we already found how to compute them in the previous slide. The rows corresponding to green nodes are in bold as they will be computed from the subgraph induced by red/green or blue/green so we can ignore them for now.
"""

# ╔═╡ 6c174bac-faef-424a-9de7-e4c7be8ddb27
md"## Larger example : star vs acyclic coloring"

# ╔═╡ dbef0e7d-b731-4d83-98e3-104b23f26042
md"`n` = $(@bind n Slider(10:20, show_value = true, default = 16))"

# ╔═╡ 2fe7f4eb-ca33-4338-ba8e-29b769b0dcce
large = sparse(Symmetric(sprand(StableRNG(0), Bool, n, n, 0.2) + I));

# ╔═╡ 1efb0f40-fb9e-4e5b-be59-84910afbf376
md"""
## Comparison of chromatic numbers

[Theorem 7.1](https://epubs.siam.org/doi/10.1137/S0036144504444711) For every graph ``G``,
```math
\xi_1(G) \le \xi_\text{acyclic}(G) \le \xi_\text{star}(G) \le \xi_2(G) = \xi_1(G^2)
```
"""

# ╔═╡ 4da92dbb-fcbd-4991-a2f1-f866778a27ef
md"## Utils"

# ╔═╡ d23e7238-5edb-42d3-96d9-0dffa9e7f326
hbox([
	md"scale = $(@bind(scale, Slider(1:15, default = 10)))",
	md"pad = $(@bind(pad, Slider(1:10, default = 3)))",
	md"border = $(@bind(border, Slider(1:5, default = 2)))",
])

# ╔═╡ e9135060-75d4-4f5a-9315-2e5fbb493cdf
function colored_plots(A; decompression, structure, partition = :column, kws...)
	problem = ColoringProblem(; structure, partition)
	algo = GreedyColoringAlgorithm(; decompression)
	result = coloring(A, problem, algo)
	background_color = RGBA(0, 0, 0, 0)
	border_color = RGB(0, 0, 0)
	colorscheme = distinguishable_colors(
    	ncolors(result),
    	[convert(RGB, background_color), convert(RGB, border_color)];
    	dropseed=true,
	)
	A_img, B_img = SparseMatrixColorings.show_colors(result; colorscheme, scale, pad, border, background_color, border_color)
	if structure == :symmetric
		S = A
	else
		if partition == :column
			S = A' * A
		else
			S = A * A'
		end
	end
	adj = SparseMatrixColorings.AdjacencyGraph(A)
	gp = gplot(Graphs.SimpleDiGraph(S - Diagonal(diag(S))); nodefillc = colorscheme[result.color], kws...)
	A_img, gp, B_img
end

# ╔═╡ 2728dc2c-ca17-4989-9259-451f95a24bd2
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

# ╔═╡ 352c92ec-14da-4beb-85c4-f66d77fa42d6
img("https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/compute_graph.png", :width => 330)

# ╔═╡ a5769449-96cd-477a-bceb-ad9439a37430
md"""
# What color is your Jacobian ?

Given a sparse matrix ``A``, two graphs represent its sparsity:

* Column intersection graph, used in [Software for estimating sparse Jacobian matrices, 1984](https://dl.acm.org/doi/abs/10.1145/1271.1610)
* Bipartite graph, used in [Colpack](https://github.com/CSCsw/ColPack) and [SparseMatrixColoring](https://github.com/gdalle/SparseMatrixColorings.jl)

$(img("Gebremedhin_Figure_3_1"))

See [Section 3.4 of What color is your Jacobian ?](https://epubs.siam.org/doi/10.1137/S0036144504444711)
"""

# ╔═╡ cabeb42c-f40b-4549-b9a2-3bda7a988672
img("Gebremedhin_Figure_6_1")

# ╔═╡ 22c1030b-bef5-4af5-9d5f-6afb4a7ae699
img("Gebremedhin_Figure_6_1_mat")

# ╔═╡ e820e135-2df2-4fa3-a09b-8ce7e5a1f745
two_columns(left, right) = hbox([
	left,
	Div(html" ", style = Dict("flex-grow" => "1")),
	right,
])

# ╔═╡ dc23180e-2b0f-41ae-aebf-0159a8f8677a
two_columns(
	md"""
Consider the ``4 \times 5`` sparse matrix on the right:

* Forward mode would require 5 JVP as there are 5 columns
* Reverse mode would require 4 VJP as there are 4 rows

Can we do better using the known sparsity pattern ?
""",
	img("https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/sparse_map.svg", :width => 200),
)

# ╔═╡ 9e24f674-42af-4fa8-bf3e-42e14d855a58
two_columns(
	md"""
* The columns 1, 2 and 5 do not share any rows with nonzero entries.
* So their entries can be recovered unambiguously with just one JVP!
	""",
	img("https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/sparse_ad.svg", :width => 400),
)

# ╔═╡ a01688ec-8f16-4d39-b39d-6d1acd24d7ad
two_columns(
	img("https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/sparse_ad_forward_full.svg", :width => 300),
	img("https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/sparse_ad_forward_decompression.svg", :width => 300),
)

# ╔═╡ 726208c6-e856-42a2-b027-555388bc505e
two_columns(
	img("https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/sparse_ad_reverse_full.svg"),
	img("https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/sparse_ad_reverse_decompression.svg"),
)

# ╔═╡ 12ffe7c2-2bf6-4949-ae28-f3d8e03a5da5
two_columns(
	md"Color's columns are structurally orthogonal",
	md"Jacobian from 3 JVP",
)

# ╔═╡ 21898d8d-f822-4128-ad56-d9a10865fb1d
three_columns(left, center, right) = hbox([
	left,
	Div(html" ", style = Dict("flex-grow" => "1")),
	center,
	Div(html" ", style = Dict("flex-grow" => "1")),
	right,
])

# ╔═╡ dcf92bc9-5e0f-4716-bb07-ca0d9b73b166
viz(args...; kws...) = three_columns(colored_plots(args...; kws...)...)

# ╔═╡ 91226321-fc72-4bd9-8d78-9c2bff5c760a
viz(S, structure = :nonsymmetric, decompression = :direct, plot_size = (3cm, 3cm))

# ╔═╡ 33b368e7-d71e-40ff-8170-a18d11db98fc
viz(ijkl, decompression = :direct, structure = :symmetric, plot_size = (3cm, 3cm))

# ╔═╡ 70296870-49af-4564-bdb7-bd2d49f9114b
viz(star, decompression = :direct, structure = :symmetric, plot_size = (5cm, 5cm))

# ╔═╡ 3e6fc4af-19f4-4ff9-b9cf-759e831ff2a4
viz(SparseMatrixColorings.what_fig_41().A, decompression = :direct, structure = :symmetric, plot_size = (5cm, 5cm))

# ╔═╡ 884f9025-10e9-4657-88ec-55e6c9b6d530
viz(ijkl2, decompression = :direct, structure = :symmetric, plot_size = (3cm, 3cm))

# ╔═╡ 647c372e-c094-4dfc-8a48-7b7f93fb94dd
viz(ijkl, decompression = :substitution, structure = :symmetric, plot_size = (4cm, 4cm))

# ╔═╡ a413c12c-72c1-4f46-b7f2-0b0118c1b232
viz(large, decompression = :direct, structure = :symmetric, plot_size = (6cm, 6cm))

# ╔═╡ 9b427766-5290-4dee-8dd5-5aab20080d0a
viz(large, decompression = :substitution, structure = :symmetric, plot_size = (6cm, 6cm))

# ╔═╡ dfe3dffd-926b-4083-9e66-8536e5df198d
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

# ╔═╡ 592510fd-a0e4-42e5-ae5b-cac301e74f0a
qa(md"How is the subgraph induced by a pair of colors ?", md"""
It does not contain paths of more than 3 vertices and adjacent vertices have opposite colors.

So it is a union of disjoint connected components and each component is a stars with one root node of a given color connected with leaf nodes of the other color.
""")

# ╔═╡ 26191290-2b2d-43fe-b2bf-d690fd0bbe15
qa(md"From which HVP do we determine the off-diagonal entries ?",
md"""
* For the edges 34 and 64, respectively between the yellow nodes 3, 6 and the blue nodes 4, they are part of the star centered at the blue node 4 so they are obtained from the entries 3 and 6 blue HVP.
* For the edges 12, 32, 52 and 62, respectively between the yellow nodes 1, 3, 5, 6 and the pink node 2, they are part of the star centered at the pink node 2 so they are obtained from the entries 1, 3, 5 and 6 of the pink HVP.
""")

# ╔═╡ 04ea8b79-63ba-4eb4-938c-98a21668924a
qa(md"How is the subgraph induced by a pair of colors ?", md"""
It does not contain cycles so it is a union of disjoint trees or in other words a forest.

The concept of tree generalizes the concept of stars. Indeed, a star is a tree and, when rooted at the center of the star, the depth of the tree is 1.
Since the depth is one, every edge is indicent to a leaf so no substitutions are needed. With acyclic coloring, the edges incident to a leaf will be obtained directly but the edges that are not incident to any tree leaves will need the entries corresponding to the edges deeper in their tree to be determined first.
""")

# ╔═╡ e721600c-1887-4e99-b474-a427b665e591
qa(md"What are the nonzero entries of ``i``th row of ``B_{\phi(i)}`` ?",
md"""
It contains ``A_{ii}`` but can it contain other nonzero entries ?
Suppose that ``A_{ij}`` is nonzero with ``\phi(j) = \phi(i)``.
This would mean that ``i`` and ``j`` are adjacent and of same color which
contradicts the fact that ``\phi`` is a distance-1 coloring.
So the only nonzero entry of the ``i``th row of ``B_{\phi(i)}`` is ``A_{ii}``.
""")

# ╔═╡ 44a6b427-bd4b-4fa1-95d8-a8ea55719192
qa(md"How can the diagonal entries be determined ?", md"""
The value of ``(h_{\phi(i)})_i`` is the sum of the entries of the ``i``th row of ``B_{\phi(i)}``.
Since the only nonzero entry of this row is ``A_{ii}``, we have ``A_{ii} = (h_{\phi(i)})_i``.
""")

# ╔═╡ 15cf368c-cada-4a72-b59b-a0f9dd0115b4
qa(
	md"How to compute the non-bold entries ?",
md"""
For star graphs, the entry ``a_{uv}`` was the edge of a star with either center ``u`` and leaf ``v`` or center ``v`` and leaf ``u``.
It was then obtained from the HVP of the color of the center.

We need to generalize this for arbitrary trees.
For any edge from a leaf ``u`` to a node ``v``, we can compute
``a_{uv}`` from the ``u``th entry of the HVP of color ``\phi(v)``.
Then, we subtract ``a_{uv}`` from the ``v``th row of the HVP of color ``\phi(u)``.
So we can do
* ``a_{71} \gets (h_\text{red})_7`` then ``(h_\text{blue})_1 \gets (h_\text{blue})_1 - a_{71}``
* ``a_{43} \gets (h_\text{red})_4`` then ``(h_\text{blue})_3 \gets (h_\text{blue})_3 - a_{43}``
* ``a_{52} \gets (h_\text{blue})_5`` then ``(h_\text{red})_2 \gets (h_\text{red})_2 - a_{52}``

Then, we can remove these leaves and edges. Doing so, some their parent in the tree becomes leaves and we can apply the same procedure recursively. So we get
* ``a_{12} \gets (h_\text{blue})_5`` then ``(h_\text{red})_2 \gets (h_\text{red})_2 - a_{12}``
* ``a_{32} \gets (h_\text{blue})_5`` then ``(h_\text{red})_2 \gets (h_\text{red})_2 - a_{32}``
""",
)

# ╔═╡ 79910840-ec59-4433-bfb3-ae2d8da40d22
qa(md"""
What is the 1-chromatic number of planar graphs ?
""",
   md"""
The [four color theorem](https://en.wikipedia.org/wiki/Four_color_theorem) shows that the 1-chromatic number of any planar graph ``G`` is ``\xi_1(G) \le 4``.
This theorem is related to our context as it applies to the **same** coloring problem but the column intersection graphs of sparse matrices are not necessarily scalar hence the chromatic number can be as large as the number of columns of the matrices here.
""")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
GraphPlot = "a2cc645c-3eea-5389-862e-a155d0052231"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
StableRNGs = "860ef19b-820b-49d6-a774-d7a799459cd3"

[compat]
GraphPlot = "~0.6.2"
Graphs = "~1.13.1"
HypertextLiteral = "~0.9.5"
Images = "~0.26.2"
PlutoTeachingTools = "~0.4.6"
PlutoUI = "~0.7.75"
SparseMatrixColorings = "~0.4.23"
StableRNGs = "~1.0.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.3"
manifest_format = "2.0"
project_hash = "d0f350d99ab1208cd99a29ffb7cc7fb7a62f7213"

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

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

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

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "d81ae5489e13bc03567d4fbbb06c546a5e53c857"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.22.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Preferences", "Static"]
git-tree-sha1 = "f3a21d7fc84ba618a779d1ed2fcca2e682865bab"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.7"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "1a3ad7e16a321667698a19e77362b35a1e94c544"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.1"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "cee8104904c53d39eb94fd06cbe60cb5acde7177"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.0.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

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

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

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

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "b137aa32bfe5b89996f8f87825b64ac41b9f2e16"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.6"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "a692f5e257d332de1e554e4566a4e5a8a72de2b2"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.4"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

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

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.GraphPlot]]
deps = ["ArnoldiMethod", "Colors", "Compose", "DelimitedFiles", "Graphs", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "066c87e33a8fcc3518c9e9970a1cbf85aa79fd6c"
uuid = "a2cc645c-3eea-5389-862e-a155d0052231"
version = "0.6.2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7a98c6502f4632dbe9fb1973a4244eaa3324e84d"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.1"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HistogramThresholding]]
deps = ["ImageBase", "LinearAlgebra", "MappedArrays"]
git-tree-sha1 = "7194dfbb2f8d945abdaf68fa9480a965d6661e69"
uuid = "2c695a8d-9458-5d45-9878-1b8a99cf7853"
version = "0.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Preferences", "Static"]
git-tree-sha1 = "af9ab7d1f70739a47f03be78771ebda38c3c71bf"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.18"

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

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageBinarization]]
deps = ["HistogramThresholding", "ImageCore", "LinearAlgebra", "Polynomials", "Reexport", "Statistics"]
git-tree-sha1 = "33485b4e40d1df46c806498c73ea32dc17475c59"
uuid = "cbc4b850-ae4b-5111-9e64-df94c024a13d"
version = "0.3.1"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageCorners]]
deps = ["ImageCore", "ImageFiltering", "PrecompileTools", "StaticArrays", "StatsBase"]
git-tree-sha1 = "24c52de051293745a9bad7d73497708954562b79"
uuid = "89d5987c-236e-4e32-acd0-25bd6bd87b70"
version = "0.1.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "52116260a234af5f69969c5286e6a5f8dc3feab8"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.12"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "8e64ab2f0da7b928c8ae889c514a52741debc1c2"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.4.2"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Bzip2_jll", "FFTW_jll", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Zlib_jll", "Zstd_jll", "libpng_jll", "libwebp_jll", "libzip_jll"]
git-tree-sha1 = "2c232857f2eb9ecfa3ab534df7f060c9afbeb187"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "7.1.2011+0"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "cffa21df12f00ca1a365eb8ed107614b40e8c6da"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.6"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "7196039573b6f312864547eb7a74360d6c0ab8e6"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.9.0"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "dfde81fafbe5d6516fb864dc79362c5c6b973c82"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.2"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "a49b96fd4a8d1a9a718dfd9cde34c154fc84fcd5"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.2"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "b842cbff3f44804a84fda409745cc8f04c029a20"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.6"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalSets]]
git-tree-sha1 = "d966f85b3b7a8e49d034d27a189e9a4874b4391a"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.13"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "8f8ff711442d1f4cfc0d86133e7ee03d62ec9b98"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.3"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

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

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

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

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

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

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "8e6a74641caf3b84800f2ccd55dc7ab83893c10b"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.17.0+0"

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

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "a9fc7883eb9b5f04f46efb9a540833d1fad974b3"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.173"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    ForwardDiffNNlibExt = ["ForwardDiff", "NNlib"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "3a8f462a180a9d735e340f4e8d5f364d411da3a4"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.8.1"

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

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NearestNeighbors]]
deps = ["AbstractTrees", "Distances", "StaticArrays"]
git-tree-sha1 = "2949f294f82b5ad7192fd544a988a1e785438ee2"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.26"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

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

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "libpng_jll"]
git-tree-sha1 = "215a6666fee6d6b3a6e75f2cc22cb767e2dd393a"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.5.5+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

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

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "972089912ba299fba87671b025cd0da74f5f54f7"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.0"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

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

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "4d8c1b7c3329c1885b857abb50d08fa3f4d9e3c8"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.7"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "5680a9276685d392c87407df00d57c9924d9f11e"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.7.1"
weakdeps = ["RecipesBase"]

    [deps.Rotations.extensions]
    RotationsRecipesBaseExt = "RecipesBase"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.SciMLPublic]]
git-tree-sha1 = "ed647f161e8b3f2973f24979ec074e8d084f1bee"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "749a2b719ec7f34f280c0d97ac3dab5c89818631"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.5.1"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

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

[[deps.SparseMatrixColorings]]
deps = ["ADTypes", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "Random", "SparseArrays"]
git-tree-sha1 = "6ed48d9a3b22417c765dc273ae3e1e4de035e7c8"
uuid = "0a514795-09f3-496d-8182-132a7b665d35"
version = "0.4.23"

    [deps.SparseMatrixColorings.extensions]
    SparseMatrixColoringsCUDAExt = "CUDA"
    SparseMatrixColoringsCliqueTreesExt = "CliqueTrees"
    SparseMatrixColoringsColorsExt = "Colors"
    SparseMatrixColoringsJuMPExt = ["JuMP", "MathOptInterface"]

    [deps.SparseMatrixColorings.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "49440414711eddc7227724ae6e570c7d5559a086"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.3.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

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

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "be5733d4a2b03341bdcab91cea6caa7e31ced14b"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.9"

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

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "d969183d3d244b6c33796b5ed01ab97328f2db85"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.5"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "98b9352a24cb6a2066f9ababcc6802de9aed8ad8"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.6"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

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

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "d1d9a935a26c475ebffd54e9c7ad11627c43ea85"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.72"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

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

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "de8ab4f01cb2d8b41702bab9eaad9e8b7d352f73"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.53+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.libzip_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "86addc139bca85fdf9e7741e10977c45785727b7"
uuid = "337d8026-41b4-5cde-a456-74a10e5b31d1"
version = "1.11.3+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─39e06fde-d734-11f0-a308-89fba7e8abd6
# ╟─e2eb1894-06ad-47a0-9ee0-c65b24a669b5
# ╟─2578b84a-365a-47b6-b58f-1696517c5d02
# ╟─dc23180e-2b0f-41ae-aebf-0159a8f8677a
# ╟─874b6593-7310-4788-beb0-2f18c0ca09a7
# ╟─9e24f674-42af-4fa8-bf3e-42e14d855a58
# ╟─aeb8b749-006a-400f-9b7d-05333fd7214f
# ╟─a01688ec-8f16-4d39-b39d-6d1acd24d7ad
# ╟─a7e8569e-2fed-4069-aa84-ced67849ab15
# ╟─726208c6-e856-42a2-b027-555388bc505e
# ╟─18594f1f-6036-4a95-8682-cf415c998311
# ╟─0a91e27f-3cea-415e-aed4-123482da375a
# ╠═97c94c5a-fab5-4226-b803-10c85a298f2a
# ╠═8e2045f3-ee09-4439-b504-d2ca9f58a3dd
# ╠═fa2a8104-2a3f-438c-bc63-4236a9f5ebeb
# ╠═9df9c76a-48af-4ca6-83f1-b8798d6d4e34
# ╟─b18a3bf6-102e-49f5-af8c-f1c35d2e2a8b
# ╠═f8e9d672-e6ad-4910-8307-2b56616b11bc
# ╟─7fec005c-11b9-4a75-a93c-195d456db190
# ╠═d46dcdb3-ac4c-4ba9-ac48-6d6a693630d3
# ╠═2e3cdd4b-9d04-477c-8c6a-fec6ec846919
# ╠═1167ae71-b63d-4ccc-a120-31c8483eca85
# ╟─45f4edeb-d934-413f-9f45-568465111b6e
# ╠═a3bf5961-323b-4adc-99a0-9a456a0bfd48
# ╠═a546edb2-6a72-4c5f-95f7-cc7709bbb505
# ╟─352c92ec-14da-4beb-85c4-f66d77fa42d6
# ╟─a5769449-96cd-477a-bceb-ad9439a37430
# ╟─1617c79d-3abe-4ffc-9ee1-3312f2b28eb5
# ╟─76162106-fb90-4883-b2d2-b87e1419c072
# ╟─5892dfbb-b375-407b-a0a7-da66adb71b67
# ╟─e882ebee-36ad-43d9-8ab7-20f7e8b2d9fe
# ╟─f29078e8-242a-4b88-881f-781829c16ed5
# ╟─5a6aec48-54fb-4d88-a8a1-b06a2bb99108
# ╟─12ffe7c2-2bf6-4949-ae28-f3d8e03a5da5
# ╟─91226321-fc72-4bd9-8d78-9c2bff5c760a
# ╟─19dd6563-1a4e-479f-881d-9bcbc720ea36
# ╟─32440932-322a-46ce-911d-5fa6a1136bca
# ╟─c2e4982a-3798-4c45-8831-5a76b3b29f6e
# ╟─e6697119-2801-48e2-b978-facc00af636d
# ╠═33b368e7-d71e-40ff-8170-a18d11db98fc
# ╟─a7b85908-404b-475b-9875-c1190de5ff71
# ╟─760ef943-e9d9-4cdc-8977-932cf6fcb78a
# ╟─70296870-49af-4564-bdb7-bd2d49f9114b
# ╟─285d3797-5086-4adc-839c-2204128585e1
# ╟─0445aae6-2f7e-42b8-b30b-f31298cdeed0
# ╟─95d7e9cd-b4a1-4205-b7d4-8c10916be8a2
# ╟─a4ecf92d-6c45-4112-8f0c-06a6d227cd84
# ╟─592510fd-a0e4-42e5-ae5b-cac301e74f0a
# ╟─29d9f355-26aa-4714-9f9e-3d020016662e
# ╟─3e6fc4af-19f4-4ff9-b9cf-759e831ff2a4
# ╟─26191290-2b2d-43fe-b2bf-d690fd0bbe15
# ╟─d0ba0c62-c788-4871-959c-fe7044519849
# ╟─884f9025-10e9-4657-88ec-55e6c9b6d530
# ╟─fab75cf6-21cc-4001-84b4-9e66f55d5a6a
# ╟─b33bf212-5f88-47ab-ba10-a2815d17f235
# ╟─647c372e-c094-4dfc-8a48-7b7f93fb94dd
# ╟─e3ae5697-6b0b-4d5f-99a0-d579b6eff655
# ╟─04ea8b79-63ba-4eb4-938c-98a21668924a
# ╟─f2418632-3261-4acb-8cc6-df333b5480a5
# ╟─cabeb42c-f40b-4549-b9a2-3bda7a988672
# ╟─2bd9ea13-362e-47ac-b26d-8e36d33e04ea
# ╟─e721600c-1887-4e99-b474-a427b665e591
# ╟─e3dec4ca-69c0-4955-9303-5f8a239546f2
# ╟─44a6b427-bd4b-4fa1-95d8-a8ea55719192
# ╟─d5b7f51b-e451-40a7-aa41-a8006decba33
# ╟─22c1030b-bef5-4af5-9d5f-6afb4a7ae699
# ╟─15cf368c-cada-4a72-b59b-a0f9dd0115b4
# ╟─6c174bac-faef-424a-9de7-e4c7be8ddb27
# ╟─a413c12c-72c1-4f46-b7f2-0b0118c1b232
# ╟─9b427766-5290-4dee-8dd5-5aab20080d0a
# ╟─dbef0e7d-b731-4d83-98e3-104b23f26042
# ╟─2fe7f4eb-ca33-4338-ba8e-29b769b0dcce
# ╟─1efb0f40-fb9e-4e5b-be59-84910afbf376
# ╟─79910840-ec59-4433-bfb3-ae2d8da40d22
# ╟─4da92dbb-fcbd-4991-a2f1-f866778a27ef
# ╟─d23e7238-5edb-42d3-96d9-0dffa9e7f326
# ╠═75597493-3e69-437f-9408-b43a89b35559
# ╠═6bd04f2f-ed5f-4ecf-91dd-8868ec5b83e1
# ╠═4f8de444-dcb4-411a-b087-6948699614e0
# ╠═dcf92bc9-5e0f-4716-bb07-ca0d9b73b166
# ╠═e9135060-75d4-4f5a-9315-2e5fbb493cdf
# ╟─2728dc2c-ca17-4989-9259-451f95a24bd2
# ╟─e820e135-2df2-4fa3-a09b-8ce7e5a1f745
# ╟─21898d8d-f822-4128-ad56-d9a10865fb1d
# ╟─dfe3dffd-926b-4083-9e66-8536e5df198d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
