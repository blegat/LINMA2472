LabAD = dirname(@__DIR__)

include(joinpath(LabAD, "test", "test.jl"))

# Reference implementation we test against
include(joinpath(LabAD, "solution", "forward.jl"))

## First order
include(joinpath(LabAD, "solution", "reverse_simple.jl"))
run_gradient_tests(Forward.gradient, SimpleReverse.gradient)

include(joinpath(LabAD, "solution", "reverse_ifelse.jl"))
run_gradient_tests(Forward.gradient, IfElseReverse.gradient)

include(joinpath(LabAD, "solution", "reverse_jacstoring.jl"))
run_gradient_tests(Forward.gradient, JacobianStoringReverse.gradient)
