LabAD = joinpath(dirname(@__DIR__), "LabAD")

include(joinpath(LabAD, "test", "test.jl"))

# Reference implementation we test against
include(joinpath(LabAD, "solution", "forward.jl"))

## First order
include(joinpath(@__DIR__, "reverse_vectorized.jl"))
run_gradient_tests(Forward.gradient, VectReverse.gradient)

## Second order
# We only test `hessian` and not `hvp` but if `hessian` is implemented
# by reusing `hvp`, this is testing both at the same time.
run_gradient_tests(Forward.hessian, VectReverse.hessian, hessian = true)
