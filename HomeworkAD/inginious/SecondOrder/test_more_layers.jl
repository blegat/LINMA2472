function random_weights(X, y, num_hidden::Vector)
    w = [rand(size(X, 2), num_hidden[1])]
    for i in 2:length(num_hidden)
        push!(w, rand(num_hidden[i - 1], num_hidden[i]))
    end
    # Can't do push in case `last_layer` is a `Vector`
    # since `w` is a `Vector` of `Matrix`
    w = [w; [last_layer(y, num_hidden[end])]]
    return Flatten(w)
end

function generate_random_more_layers(classification,hessian)
    if hessian
        n = 4  # number of layers
        num_data = rand(1:5)
        num_features = rand(1:3)
        num_hidden = rand(1:5, n-1) # generates n random hidden layer sizes between 2 and 10
    else 
        n = 6 # number of layers
        num_data = rand(5:10)
        num_features = rand(2:5)
        num_hidden = rand(5:10, n-1) # generates n random hidden layer sizes between 5 and 30
    end
    return generate_data(num_data, num_features, num_hidden, classification)
end

# -------------------------
# Generic test runner
# -------------------------
function run_gradient_tests_more_layers(Reference_diff, Given_diff, loss_function, activation; 
                            name="Test", 
                            classification=false, 
                            hessian=false,
                            num_repeats=5, 
                            rtol=1e-8) 
    @testset "$name with more layers" begin # do not works -> redefine activation
        # 9. Random tests with more than 2 layers 
        for _ in 1:num_repeats
            X, y, w = generate_random_more_layers(classification, hessian)
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
        end
    end
end

struct NeuralNetwork{A}
    activation::A
    softmax::Bool
end

function (model::NeuralNetwork)(W::Flatten, X)
    σ = model.activation
    wds = W.components
    y = X * wds[1]
    for i in 2:length(wds)
        if model.activation === identity
            x = y # Shortcut so that you don't need to implement broadcasting with `identity` :)
        else
            x = σ.(y)
        end
        y = x * wds[i]
    end
    if model.softmax
        y = softmax(y)
    end
    return y
end

function run_gradient_tests_more_layers(Reference_diff, Given_diff; kws...)
    # 1) Identity activation
    run_gradient_tests_more_layers(Reference_diff, Given_diff, mse, NeuralNetwork(identity, false), name="Identity"; kws...)

    # 2) Tanh activation
    run_gradient_tests_more_layers(Reference_diff, Given_diff, mse, NeuralNetwork(tanh, false), name="Tanh"; kws...)

    # 3) ReLU
    run_gradient_tests_more_layers(Reference_diff, Given_diff, mse, NeuralNetwork(relu, false), name="ReLU"; kws...)

    # 4) Softmax Tanh
    run_gradient_tests_more_layers(Reference_diff, Given_diff, cross_entropy, NeuralNetwork(tanh, true), name="Softmax Tanh"; classification = true, kws...)

    # 4) Softmax ReLU
    run_gradient_tests_more_layers(Reference_diff, Given_diff, cross_entropy, NeuralNetwork(relu, true), name="Softmax ReLU"; classification = true, kws...)
    return
end
