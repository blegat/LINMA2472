using LinearAlgebra
using SparseConnectivityTracer, SparseMatrixColorings
using Graphs, GraphPlot, Colors
using Compose, Cairo, Fontconfig

detector = TracerSparsityDetector()
x = rand(3)
f(x, y, z, λ, γ) = sum(x.^2) - λ * sum(x) + λ^2 + γ^2 + sum(sin(γ - z) for z in z) + dot(x - z, y)
v(x) = f(x[1:n], x[n+1:2n], x[2n+1:3n], x[3n+1], x[3n+2])
f(x, y, λ) = sum(x.^2) + sum(y.^2) + λ^2 + dot(y .- λ, x)
n = 3
v(x) = f(x[1:n], x[n+1:2n], x[2n+1])

v(x) = sum(x.^2) + sum(x[2i] * (x[2i-1] - x[2n+1]) for i in 1:n)
x = rand(2n+1)
S = hessian_sparsity(v, x, detector)
G = Graphs.SimpleDiGraph(S - Diagonal(diag(S)))
adjacency_graph = gplot(G, nodelabel = eachindex(x))
draw(PNG(joinpath(@__DIR__, "adjacency_graph.png"), 16cm, 16cm), adjacency_graph)

problem = ColoringProblem(; structure=:symmetric, partition=:column)

star_algo = GreedyColoringAlgorithm(; decompression=:direct)
star_result = coloring(S, problem, star_algo)
adj = SparseMatrixColorings.AdjacencyGraph(S)
background_color = RGBA(0, 0, 0, 0)
border_color = RGB(0, 0, 0)
colorscheme = distinguishable_colors(
    ncolors(result),
    [convert(RGB, background_color), convert(RGB, border_color)];
    dropseed=true,
)
star_coloring = gplot(G; nodelabel = eachindex(x), nodefillc = colorscheme[star_result.color])
draw(PNG(joinpath(@__DIR__, "star_coloring.png"), 16cm, 16cm), star_coloring)

acyclic_algo = GreedyColoringAlgorithm(; decompression=:substitution)
acyclic_result = coloring(S, problem, acyclic_algo)
acyclic_coloring = gplot(G; nodelabel = eachindex(x), nodefillc = colorscheme[acyclic_result.color])
draw(PNG(joinpath(@__DIR__, "acyclic_coloring.png"), 16cm, 16cm), acyclic_coloring)
