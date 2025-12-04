include("test.jl")

# Reference implementation we test against
include("forward.jl")

## Second order
include("reverse_vectorized.jl")
run_gradient_tests(Forward.hessian, VectReverse.hessian, hessian = true)