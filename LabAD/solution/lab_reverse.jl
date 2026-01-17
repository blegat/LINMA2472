# See the README for what to install to run this file.

########## Warm-up ############

function quad(x)
    I = eachindex(x)
    y = x - I
    return sum(I .* y.^2)
end

include(joinpath(@__DIR__, "forward.jl"))
include(joinpath(@__DIR__, "reverse_simple.jl"))
include(joinpath(@__DIR__, "reverse_ifelse.jl"))
include(joinpath(@__DIR__, "reverse_jacstoring.jl"))

x = rand(2)
∇f = @time Forward.gradient(quad, x)
∇r = @time SimpleReverse.gradient(quad, x)
∇r_ifelse = @time IfElseReverse.gradient(quad, x)
∇r_jacs = @time JacobianStoringReverse.gradient(quad, x)

# We should get a difference at the order of `1e-15` unless we got it wrong:
norm.(∇f .- ∇r)
norm.(∇f .- ∇r_ifelse)
norm.(∇f .- ∇r_jacs)
@test all(∇f .≈ ∇r)
@test all(∇f .≈ ∇r_ifelse)
@test all(∇f .≈ ∇r_jacs)

########## Stretching ############

include(joinpath(dirname(@__DIR__), "data.jl"))
include(joinpath(dirname(@__DIR__), "models.jl"))
using Test, LinearAlgebra

num_data = 100
X, y = random_moon(num_data)
num_hidden = 10

w = random_weights(X, y, num_hidden)
L = loss(mse, identity_activation, X, y)

∇f = @time Forward.gradient(L, w)
∇r = @time SimpleReverse.gradient(L, w)
∇r_ifelse = @time IfElseReverse.gradient(L, w)
∇r_jacs = @time JacobianStoringReverse.gradient(L, w)

# We should get a difference at the order of `1e-15` unless we got it wrong:
norm.(∇f .- ∇r)
norm.(∇f .- ∇r_ifelse)
norm.(∇f .- ∇r_jacs)
@test all(∇f .≈ ∇r)
@test all(∇f .≈ ∇r_ifelse)
@test all(∇f .≈ ∇r_jacs)

########## tanh ############

L = loss(mse, tanh_activation, X, y)

# ## Exercise 1

∇f = @time Forward.gradient(L, w)
∇r = @time SimpleReverse.gradient(L, w)
∇r_ifelse = @time IfElseReverse.gradient(L, w)
∇r_jacs = @time JacobianStoringReverse.gradient(L, w)

# We should get a difference at the order of `1e-15` unless we got it wrong:
norm.(∇f .- ∇r)
norm.(∇f .- ∇r_ifelse)
norm.(∇f .- ∇r_jacs)
@test all(∇f .≈ ∇r)
@test all(∇f .≈ ∇r_ifelse)
@test all(∇f .≈ ∇r_jacs)

########## ReLU ############

L = loss(mse, relu_activation, X, y)

# ## Exercise 2

∇f = @time Forward.gradient(L, w)
∇r = @time SimpleReverse.gradient(L, w)
∇r_ifelse = @time IfElseReverse.gradient(L, w)
∇r_jacs = @time JacobianStoringReverse.gradient(L, w)

norm.(∇f .- ∇r)
@test all(∇f .≈ ∇r)

########## Cross entropy ############

Y_encoded = one_hot_encode(y)
w = random_weights(X, Y_encoded, num_hidden)
L = loss(cross_entropy, relu_softmax, X, Y_encoded)

# ## Exercise 3

∇f = @time Forward.gradient(L, w)
∇r = @time SimpleReverse.gradient(L, w)
∇r_ifelse = @time IfElseReverse.gradient(L, w)
∇r_jacs = @time JacobianStoringReverse.gradient(L, w)

norm.(∇f .- ∇r)
norm.(∇f .- ∇r_ifelse)
norm.(∇f .- ∇r_jacs)
@test all(∇f .≈ ∇r)
@test all(∇f .≈ ∇r_ifelse)
@test all(∇f .≈ ∇r_jacs)

# ## Exercise 4

# Already finished the first three exercises ?
# Try to compute the hessian of the loss function.
# *Hint:* You can compute it as the Jacobian of the gradient like
#         in `lab_forward.jl`. In `lab_forward.jl`, you used forward
#         for both the Jacobian and the gradient. This time, using
#         reverse for both the Jacobian and gradient is probably
#         a bit ambitious but try using reverse of at least one of
#         them. This is called doing *forward-over-reverse*.

# ## Exercise 5:

# It's best to first focus on correctness but once everything is correct you
# can try looking at performance (the fun part)

# Let's first get a more accurate benchmarking by executing the benchmark many times, do:
using BenchmarkTools
@benchmark $Forward.gradient($L, $w)
@benchmark $SimpleReverse.gradient($L, $w)
@benchmark $IfElseReverse.gradient($L, $w)
@benchmark $JacobianStoringReverse.gradient($L, $w)

# The `$` are needed because BenchmarkTools prevents accessing global variables
# since that is often a performance pitfall.
# If you want to investigate where the the time is spent, use (run it twice and
# discard the first plot as it probably also contains traces corresponding to compilation)
# Note that `@profview` and `@profview_allocs` only work in VS code.

@profview SimpleReverse.gradient(L, w)

# If you see a high number of allocations, this may also be a sign of performance issue.
# You can investigate where they come from with

@profview_allocs SimpleReverse.gradient(L, w)

# Now that we have done some benchmarking, we can start trying to improve performance.

# ## Exercise 6:

# In `reverse.jl`, when creating the expression graph, we construct a Node for each of the variables and constants of the expression.
# Are the derivatives with respect to ALL these Nodes useful? If not, modify the code so as to avoid the unnecessary computations.

# ## Exercise 7:

# In `reverse.jl`, the local jacobians are computed and stored during the backward pass.
# This means that we have to check the symbol of the operation of each Node during the backward pass,
# implying a costly if-else during the backward pass. Note however that the local jacobians could be
# computed during the forward pass (in which we have to check (if-else) the symbol of the operation of
# each Node anyway), and stored in their respective Nodes.
# Then the backward pass would only consists of multiplying the local jacobians together, with no need to know the symbols.
# The backward pass is therefore faster. Modify the code so as to implement this 'jacobian-storing' version of Reverse mode.
# Is there any downside to this version?
