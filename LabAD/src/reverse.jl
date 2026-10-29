module Reverse

import ComputationGraphExplorer as CGE

mutable struct ReverseData
    derivative::Float64
end
CGE.metadata(::Type{ReverseData}, ::Float64) = ReverseData(0.0)
CGE.metadata_rows(data::ReverseData) = ["r" => data.derivative]

const Node = CGE.Node{Float64,ReverseData}

function CGE.seed_metadata!(data::ReverseData, is_output::Bool)
    data.derivative = is_output ? 1.0 : 0.0
end

function CGE.pullback!(::typeof(+), f::Node, args::Node...)
    for arg in args
        arg.metadata.derivative += f.metadata.derivative
    end
end

function CGE.pullback!(::typeof(-), f::Node, x::Node, y::Node)
    x.metadata.derivative += f.metadata.derivative
    y.metadata.derivative -= f.metadata.derivative
end

function CGE.pullback!(::typeof(*), f::Node, x::Node, y::Node)
    x.metadata.derivative += f.metadata.derivative * y.value
    y.metadata.derivative += f.metadata.derivative * x.value
end

function CGE.pullback!(op, f::Node, args...)
    error("$op is not implemented yet, this is the purpose of the practice session!")
end

function gradient!(f, g, x)
    x_nodes = map(Node, x)
    expr = f(x_nodes)
    CGE.backward!(expr)
    return map!(node -> node.metadata.derivative, g, x_nodes)
end

gradient(f, x) = gradient!(f, zero(x), x)

end
