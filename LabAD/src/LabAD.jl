# Copyright (c) 2026 Benoît Legat
# SPDX-License-Identifier: MIT

module LabAD

import Random

include("flatten.jl")
include("forward.jl")
include("reverse.jl")
include("models.jl")
include("train.jl")
include("data.jl")

export Flatten,
    unflatten_index,
    Forward,
    Reverse,
    random_weights,
    identity_activation,
    tanh_activation,
    relu_activation,
    relu_softmax,
    mse,
    softmax,
    cross_entropy,
    one_hot_encode,
    loss,
    train!,
    random_moon,
    plot_moon

end
