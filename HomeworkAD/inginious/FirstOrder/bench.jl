using BenchmarkTools

include(joinpath(@__DIR__, "test.jl"))
include(joinpath(@__DIR__, "test_more_layers.jl"))

## First order
num_data = 200
num_features = 10
n_hidden = 12
using Random
Random.seed!(0)
num_hidden = rand(2:10, n_hidden-1) 
X, y, w = generate_data(num_data, num_features, num_hidden, false)
# Identity activation
L = loss(cross_entropy, NeuralNetwork(relu, true), X, y)

include(joinpath(@__DIR__, "reverse_vectorized.jl"))
println(@belapsed $VectReverse.gradient($L, $w))