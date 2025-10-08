using Test
using LinearAlgebra

include(joinpath(dirname(@__DIR__), "models.jl"))

# -------------------------
# Utils
# -------------------------
function flatten_gradient(g)
    if isa(g, Float64)
        return [g]
    elseif hasproperty(g, :components)
        return reduce(vcat, vec.(g.components))
    elseif isa(g, AbstractArray) && !isa(g, Vector)
        # If it's a matrix of Flatten, flatten each element
        return reduce(vcat, flatten_gradient.(g))
    else
        return vec(g)
    end
end

function check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol=1e-8, test=true)
    L = loss(loss_function, activation, X, y)
    ∇f = Reference_diff(L, deepcopy(w))
    ∇r = Given_diff(L, deepcopy(w))
    ∇f = flatten_gradient(∇f)
    ∇r = flatten_gradient(∇r)
    if ! test
        @test_skip "Not doing this test" 
    else
        @test isapprox(norm(∇f - ∇r), 0.0; atol = rtol * norm(∇f))
    end
end

function generate_data(num_data, num_features, num_hidden, classification)
    X = rand(num_data, num_features)
    y = rand(num_data)
    if classification
        y = one_hot_encode(y)
    end
    w = random_weights(X, y, num_hidden)
    X .= X .* 2 .- 1
    w.components .= [c .* 2 .- 1 for c in w.components]
    if !classification
        y .= y .* 2 .- 1
    end
    return X, y, w
end

function generate_random(classification, hessian)
    if hessian
        num_data = rand(2:10)
        num_features = rand(2:4)
        num_hidden = rand(2:10)
    else 
        num_data = rand(5:30)
        num_features = rand(2:20)
        num_hidden = rand(5:30)
    end
    return generate_data(num_data, num_features, num_hidden, classification)
end

# -------------------------
# Generic test runner
# -------------------------
# If `zero_test` is `true`, the input of activation layers might be zero.
# If the activation is `ReLU`, this will test that `Reference_diff` and `Given_diff`
# give the same gradient even though several gradient are valid which is tricky.
# Use `zero_test = false` to deactivate them. We don't require you to pass the
# tests with `zero_test = true` for the project.
function run_gradient_tests(Reference_diff, Given_diff, loss_function, activation; 
                            name="Test",
                            classification=false,
                            num_repeats=5,
                            rtol=1e-8, hessian=false,
                            zero_test=true) 
    @testset "$name" begin
        # 1. Random small tests
        for _ in 1:num_repeats
            # with only positive values
            X, y, w = generate_random(classification, hessian)
            X .= abs.(X)
            y .= abs.(y)
            w.components .= [abs.(c) for c in w.components]
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
            # with some negative value
            X, y, w = generate_random(classification, hessian)
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
            # with negative values
            X, y, w = generate_random(classification, hessian)
            X .= -abs.(X)
            w.components .= [-abs.(c) for c in w.components]
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
            X, y, w = generate_random(classification, hessian)
            X .= -abs.(X)
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
            X, y, w = generate_random(classification, hessian)
            w.components .= [-abs.(c) for c in w.components]
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
            # edge case scalar
            num_data = 1
            num_hidden = 1
            num_features = 1
            X, y, w = generate_data(num_data, num_features, num_hidden, classification)
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
        end
    end

    @testset "$name with identical labels" begin
        # 2. Random tests with identical labels
        for _ in 1:num_repeats
            X, y, w = generate_random(classification, hessian)
            y .= 1.0
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
        end
    end

    @testset "$name with small numbers" begin 
        # 3. Random tests with small numbers
        for _ in 1:num_repeats
            X, y, w = generate_random(classification, hessian)
            X = X*10^-5
            y = y*10^-5
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
        end
    end
    if !classification # cross_entropy fail with big numbers
        @testset "$name with big numbers" begin 
            # 4. Random tests with big numbers
            for _ in 1:num_repeats
                X, y, w = generate_random(classification, hessian)
                X = X*10^3
                y = y*10^3
                check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
            end
        end
    end
    if !classification
        @testset "$name with extreme numbers" begin
            # 5. Random tests with extreme numbers (big for X, small for y)
            for _ in 1:num_repeats 
                X, y, w = generate_random(classification, hessian)
                X .= X * 1e10
                y .= y * 1e-10
                check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
            end
        end
    end

    if !hessian # too slow for Hessian
        @testset "$name with big matrix" begin 
            # 6. Random tests with big matrix
            for _ in 1:num_repeats
                if classification # otherwise too slow for cross_entropy
                    num_data = rand(10:30)
                    num_hidden = rand(10:30)
                    num_features = rand(5:10)
                else    
                    num_data = rand(30:50)
                    num_hidden = rand(30:50)
                    num_features = rand(5:20)
                end
                X, y, w = generate_data(num_data, num_features, num_hidden, classification)

                check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol)
            end
        end
    end

    @testset "$name with sparse weights" begin # Fail for IfElseReverse
        # 7. Random tests with sparse weights
        for _ in 1:num_repeats
            X, y, w = generate_random(classification, hessian)
            mask = rand(length(w)) .> 0.7
            offset = 0
            for i in eachindex(w.components)
                n = length(w.components[i])
                w.components[i] .= w.components[i] .* reshape(mask[offset+1:offset+n], size(w.components[i]))
                # Remplace -0.0 par 0.0
                w.components[i] .= map(x -> x == 0.0 && signbit(x) ? 0.0 : x, w.components[i])
                offset += n
            end # 70% of weights sont zero et -0.0 remplacés par 0.0
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol, zero_test)
        end
    end
    @testset "$name with zero matrix" begin
        for _ in 1:num_repeats
            # 8. Random tests with zero matrix
            X, y, w = generate_random(classification, hessian)
            for i in eachindex(w.components)
                w.components[i] .= 0.0
            end
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol, zero_test)

            X, y, w = generate_random(classification, hessian)
            X .= 0.0
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol, zero_test)

            X, y, w = generate_random(classification, hessian)
            y .= 0.0
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol, zero_test)

            X, y, w = generate_random(classification, hessian)
            X .= 0.0
            y .= 0.0
            check_case(X, y, w, loss_function, activation, Reference_diff, Given_diff, rtol, zero_test)
        end
    end  
end

# We default to `test_zero_relu=false`, passing the tests with `test_zero_relu=true`
# is out of the scope of this project.
function run_gradient_tests(Reference_diff, Given_diff, test_zero_relu=false; kws...)
    # 1) Identity activation
    run_gradient_tests(Reference_diff, Given_diff, mse, identity_activation, name="Identity"; kws...)

    # 2) Tanh activation
    run_gradient_tests(Reference_diff, Given_diff, mse, tanh_activation, name="Tanh"; kws...)

    # 3) ReLU
    run_gradient_tests(Reference_diff, Given_diff, mse, relu_activation, name="ReLU", zero_test=test_zero_relu; kws...)

    # 4) Softmax
    run_gradient_tests(Reference_diff, Given_diff, cross_entropy, relu_softmax, name="Softmax", zero_test=test_zero_relu; classification = true, kws...)
    return
end
