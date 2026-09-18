# See the README for what to install to run this file.

########## Warm-up ############

using Revise # This will allow the changes you make to LabAD to take effect even after executing `using LabAD`.`
using LabAD
import ComputationGraphExplorer as CGE

x = rand(2)

univariate(x) = x[1]^2 + x[1]

∇f = @time Forward.gradient(univariate, x)

# For the reverse mode, we will build the computation graph with `ComputationGraphExplorer`.
# We can visualize the computation graph by looking at the REPL output when executing the following line:

graph = univariate(map(Reverse.Node, x))

# We can visualize the values of the reverse tangents after the backward pass:

CGE.backward!(graph)

# We can visualize the computation graph as an image in the VS Code plot pane as follows:

CGE.visualize(graph)

# Note that the node correspinding to `x` is used twice (as illustrated by the `↩` in the text representation of the graph),
# so it's a DAG (Directed Acyclic Graph), not a tree!

## Multivariate function


# (x1 - 1)^2 + 2(x2 - 2)^2
function quad(x)
    I = eachindex(x)
    y = x - I
    return sum(I .* y.^2)
end

∇f = @time Forward.gradient(quad, x)

# For the reverse mode, we will build the computation graph with `ComputationGraphExplorer`.
# We can visualize the computation graph as follows:

graph = quad(map(Reverse.Node, x))

# The backward pass is going to because we haven't `CGE.pullback!` backward pass for all operations yet.

CGE.backward!(graph)

# But we can still visualize

CGE.visualize(graph)

# The gradient computation also fails since it is using `CGE.backward!` internally

∇r = @time Reverse.gradient(quad, x)

########## Stretching ############

using Test, LinearAlgebra

num_data = 100
X, y = random_moon(num_data)
num_hidden = 10

w = random_weights(X, y, num_hidden)
L = loss(mse, identity_activation, X, y)

∇f = @time Forward.gradient(L, w)
∇r = @time Reverse.gradient(L, w)

# We should get a difference at the order of `1e-15` unless we got it wrong:
norm.(∇f .- ∇r)
@test all(∇f .≈ ∇r)

########## tanh ############

L = loss(mse, tanh_activation, X, y)

# ## Exercise 1

∇f = @time Forward.gradient(L, w)
∇r = @time Reverse.gradient(L, w)

# We should get a difference at the order of `1e-15` unless we got it wrong:
norm.(∇f .- ∇r)
@test all(∇f .≈ ∇r)

########## ReLU ############

L = loss(mse, relu_activation, X, y)

# ## Exercise 2

∇f = @time Forward.gradient(L, w)
∇r = @time Reverse.gradient(L, w)

norm.(∇f .- ∇r)
@test all(∇f .≈ ∇r)

########## Cross entropy ############

Y_encoded = one_hot_encode(y)
w = random_weights(X, Y_encoded, num_hidden)
L = loss(cross_entropy, relu_softmax, X, Y_encoded)

# ## Exercise 3

∇f = @time Forward.gradient(L, w)
∇r = @time Reverse.gradient(L, w)

norm.(∇f .- ∇r)
@test all(∇f .≈ ∇r)

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
@benchmark $Reverse.gradient($L, $w)

# The `$` are needed because BenchmarkTools prevents accessing global variables
# since that is often a performance pitfall.
# If you want to investigate where the the time is spent, use (run it twice and
# discard the first plot as it probably also contains traces corresponding to compilation)
# Note that `@profview` and `@profview_allocs` only work in VS code.

@profview Reverse.gradient(L, w)

# If you see a high number of allocations, this may also be a sign of performance issue.
# You can investigate where they come from with

@profview_allocs Reverse.gradient(L, w)

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


# ## Takeaway

# Use ComputationGraphExplorer.jl to visualize the expression graph of the loss function.
# We'll first need to reduce the size significantly:
# You can do this by running

num_data_tiny = 4
X_tiny, y_tiny = random_moon(num_data_tiny)
num_hidden_tiny = 2

w_tiny = random_weights(X_tiny, y_tiny, num_hidden_tiny)
L_tiny = loss(mse, identity_activation, X_tiny, y_tiny)

graph = L_tiny(map(Reverse.Node, w_tiny))

CGE.backward!(graph)

CGE.visualize(graph)

# You can observe that the number of nodes of the computation graph grows with `num_data_tiny` and
# `num_hidden_tiny`. Is that an issue in terms of performance for computing the gradient ?
# Think about it, it is a teaser for the project ;)
