# See the README for what to install to run this file.

########## Warm-up ############

# Let's start by minimizing a quadratic function:
# ∑ᵢ i(xᵢ - i)^2

function quad(x)
    I = eachindex(x)
    y = x - I
    return sum(I .* y.^2)
end

# WIP implementation of forward differentiation to be completed in this lab!
include(joinpath(@__DIR__, "forward.jl"))
include(joinpath(@__DIR__, "train.jl"))

# Let's take a 2D example so that it's easy to visualize
using Plots
heatmap(range(-1, stop=3, length=100), range(0, stop=4, length=100), quad ∘ vcat)
x = rand(2)
xs = [copy(x)]
Forward.gradient(quad, x)
# We see that the gradient is not zero, let's train 10 times with 10 steps each
for _ in 1:10
    losses = train!(Forward.gradient!, quad, x, num_iters = 1)
    push!(xs, copy(x))
end
Forward.gradient(quad, x)
scatter!(getindex.(xs, 1), getindex.(xs, 2), label = "")

########## Stretching ############

# Let's now move to a neural network model but first
# without any activation function

include("data.jl")
include("models.jl")

# Number of data samples
num_data = 100
X, y = random_moon(num_data)

# We use a simple neural network with 1 hidden layer of
# `num_hidden` neurons.
num_hidden = 10
w = random_weights(X, y, num_hidden);

plot_moon(identity_activation, w, X, y)

# Although the cross-entropy loss would be appropriate here,
# we use the mean-squared error loss to start simple.
# Our forward differentiation implementation can compute the gradient
# as follows:

L = loss(mse, identity_activation, X, y)
L(w)
Forward.gradient(L, w)

# Let's load some training utilities that will follow the gradient:

# Let's do 10 gradient steps with a step-size of `0.1` (`0.1` is the default,
# see https://fluxml.ai/Optimisers.jl/stable/api/#Optimisers.Descent)

losses = train!(Forward.gradient!, L, w)

using Plots
plot(eachindex(losses), losses, label = "")

# We can see in the plot that we are far from converged.
# Let's do 10 more gradient steps. We pass the `losses` as additional argument
# so that the new losses are appended to the plot

losses = train!(Forward.gradient!, L, w; losses)
plot(eachindex(losses), log.(losses), label = "")

# We can see that the linear model now does its best to 
# differentiate each category.

plot_moon(identity_activation, w, X, y)

########## tanh ############

# We now turn into a more realistic situation in which
# we use a `tanh` activation function.
# Let's start with new random weights

w = random_weights(X, y, num_hidden)
L = loss(mse, tanh_activation, X, y)

# We see now that the model is nonlinear but still untrained

plot_moon(tanh_activation, w, X, y)

# ## Exercise 1

# If we try to compute the gradient, we get an error, replace it with
# a correct implementation.
# *Hint:* ``tanh(x)' = 1 - tanh(x)^2``
Forward.gradient(L, w)

losses = train!(Forward.gradient!, L, w)

# Train more!

train!(Forward.gradient!, L, w; losses)

plot(eachindex(losses), log.(losses), label = "")

plot_moon(tanh_activation, w, X, y)

########## ReLU ############

# We now replace the `tanh` activation by a ReLU activation.
# This function is not differentiable at 0 so we are not actually
# computing proper gradients but we'll still name our function
# `gradient` as it is common in Automatic Differentiation.

w = random_weights(X, y, num_hidden)
plot_moon(relu_activation, w, X, y)

# ## Exercise 2

# FIXME This time we didn't add a placeholder function throwing an error so we get
# a `MethodError` saything that the method `isless(::Dual, ::Int)` is missing
# 3 choices here:
# 1) implement `isless(::Dual, ::Real)`
# 2) implement `max(::Real, ::Dual)`
# 3) implement `relu(::Dual)`
L = loss(mse, relu_activation, X, y)
Forward.gradient(L, w)

losses = train!(Forward.gradient!, L, w)

# Train more !

train!(Forward.gradient!, L, w; losses)

plot(eachindex(losses), log.(losses), label = "")

plot_moon(relu_activation, w, X, y)

########## Cross entropy ############

Y_encoded = one_hot_encode(y)

w = random_weights(X, Y_encoded, num_hidden)

plot_moon(relu_softmax, w, X, y)

# ## Exercise 3

# If we try computing the gradient, we get errors!
# Fix them to complete this exercise

L = loss(cross_entropy, relu_softmax, X, Y_encoded)
Forward.gradient(L, w)

losses = train!(Forward.gradient!, L, w)

# Train more !

train!(Forward.gradient!, L, w; losses)

plot(eachindex(losses), log.(losses), label = "")

plot_moon(relu_softmax, w, X, y)

# ## Exercise 4:

# Already finished the three first exercises ?
# Try to compute the hessian of the loss function.
# *Hint:* The hessian is the **Jacobian** of the **gradient**
#         so you can start by implementing a Jacobian
#         function and then combine it with the existing gradient
#         function to get the hessian.
