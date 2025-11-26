include("flatten.jl")

########## Warm-up ############

mse(y_est, y) = sum((y_est - y).^2) / length(y)

function last_layer(::AbstractVector, num_hidden::Integer)
    return rand(num_hidden)
end

function random_weights(X, y, num_hidden::Integer)
    W1 = rand(size(X, 2), num_hidden)
    W2 = last_layer(y, num_hidden)
    return Flatten(W1, W2)
end

function identity_activation(W::Flatten, X)
    W1, W2 = W.components
    return X * W1 * W2
end

function loss(dist, model, X, y)
    return (w::Flatten) -> dist(model(w, X), y)
end

########## tanh ############

function tanh_activation(W, X)
    W1, W2 = W.components
    hidden_layer = tanh.(X * W1)
    return hidden_layer * W2
end

########## ReLU ############

function relu(x)
    return max(0, x)
end

function relu_activation(W::Flatten, X)
    W1, W2 = W.components
    hidden_layer = relu.(X * W1)
    return hidden_layer * W2
end

########## Cross entropy ############

function last_layer(Y::AbstractMatrix, num_hidden::Integer)
    return rand(num_hidden, size(Y, 2))
end

function softmax(x)
    exps = exp.(x .- maximum(x, dims = 2))
    # Le -max(x) est utilisé dans la fonction softmax pour éviter les problèmes 
    # d'instabilité numérique lorsque les entrées sont grandes. Cela améliore la 
    # stabilité des calculs sans affecter les résultats finaux.
    return exps ./ sum(exps, dims = 2)
end

function one_hot_encode(labels::Vector)
    classes = unique!(sort(labels))
    return classes' .== labels
end

function relu_softmax(W, X)
    W1, W2 = W.components
    hidden_layer = relu.(X * W1)
    logits = hidden_layer * W2
    return softmax(logits)
end

function cross_entropy(Y_est, Y)
    return -sum(log.(Y_est) .* Y) / size(Y, 1)
end
