include("test.jl")

# Reference implementation we test against
include("forward.jl")

## First order
include("reverse_vectorized.jl")
run_gradient_tests(Forward.gradient, VectReverse.gradient)