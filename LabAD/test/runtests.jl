# Copyright (c) 2026 Benoît Legat
# SPDX-License-Identifier: MIT

import ComputationGraphExplorer as CGE
import Random
using LabAD
using Test

@testset "Flatten" begin
    first = reshape(collect(1.0:4.0), 2, 2)
    second = [5.0, 6.0]
    flattened = Flatten(first, second)

    @test length(flattened) == 6
    @test size(flattened) == (6,)
    @test collect(flattened) == collect(1.0:6.0)
    @test unflatten_index(flattened, 5) == (2, 1)
    @test_throws BoundsError flattened[7]

    flattened[5] = 7.0
    @test second[1] == 7.0
    @test collect(zero(flattened)) == zeros(6)
    @test collect(map(x -> 2x, flattened)) == 2 .* collect(flattened)

    destination = similar(flattened, Float64)
    map!(identity, destination, flattened)
    @test collect(destination) == collect(flattened)

    columns = reduce(hcat, [Flatten([1.0, 2.0]), Flatten([3.0, 4.0])])
    @test columns == [1.0 3.0; 2.0 4.0]
end

function quadratic(x)
    indices = eachindex(x)
    displacement = x - indices
    return sum(indices .* displacement .^ 2)
end

@testset "Forward lab warm-up" begin
    x = [2.0, 3.0]
    @test Forward.gradient(quadratic, x) == [2.0, 4.0]

    gradient = similar(x)
    @test Forward.gradient!(quadratic, gradient, x) === gradient
    @test gradient == [2.0, 4.0]

    trained = zeros(2)
    losses = train!(Forward.gradient!, quadratic, trained; num_iters = 3)
    @test length(losses) == 4
    @test losses[end] < losses[begin]
end

@testset "Data and models used by both labs" begin
    Random.seed!(1234)
    X, y = random_moon(20)
    @test size(X) == (20, 2)
    @test size(y) == (20,)
    @test all(label -> label in (-1.0, 1.0), y)

    weights = random_weights(X, y, 3)
    @test size.(weights.components) == [(2, 3), (3,)]
    prediction = identity_activation(weights, X)
    @test size(prediction) == size(y)

    objective = loss(mse, identity_activation, X, y)
    @test objective(weights) == mse(prediction, y)
    gradient = Forward.gradient(objective, weights)
    @test size.(gradient.components) == size.(weights.components)
    @test all(all(isfinite, component) for component in gradient.components)

    encoded = one_hot_encode([-1.0, 1.0, -1.0, 1.0])
    @test size(encoded) == (4, 2)
    @test vec(sum(encoded; dims = 2)) == ones(Int, 4)

    probabilities = softmax([1000.0 1001.0; -1000.0 -999.0])
    @test all(isfinite, probabilities)
    @test vec(sum(probabilities; dims = 2)) ≈ ones(2)
    @test isfinite(cross_entropy(probabilities, encoded[1:2, :]))

    classification_weights = random_weights(X, encoded, 3)
    @test size.(classification_weights.components) == [(2, 3), (3, 2)]
end

@testset "Reverse lab warm-up" begin
    univariate(x) = x[1]^2 + x[1]
    x = [2.0, 3.0]

    graph = univariate(map(Reverse.Node, x))
    @test graph.value == univariate(x)
    text = sprint(show, graph)
    @test occursin("↩", text)
    svg = repr(MIME"image/svg+xml"(), CGE.visualize(graph))
    @test occursin("<svg", svg)

    bilinear(x) = x[1] * x[2] + x[1]
    @test Reverse.gradient(bilinear, x) == [4.0, 2.0]
end
