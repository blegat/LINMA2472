module IfElseReverse

import ComputationGraphExplorer as CGE

mutable struct ReverseData
    derivative::Float64
end
CGE.metadata(::Type{ReverseData}, ::Float64) = ReverseData(0.0)
CGE.metadata_rows(data::ReverseData) =
    ["r" => data.derivative]

const Node = CGE.Node{Float64,ReverseData}

function CGE.seed_metadata!(data::ReverseData, is_output::Bool)
    data.derivative = is_output ? 1.0 : 0.0
end

# Unlike SimpleReverse, constants are kept as ordinary numbers whenever
# possible. The operation symbol and arity therefore distinguish, for
# example, Node * Node from Node * Number.
Base.zero(::Node) = Node(0.0)
Base.:*(x::Node, y::Node) = Node(:*, Node[x, y], x.value * y.value)
function Base.:*(x::Node, y::Number)
    if iszero(x.value)
        # Recovering y from (x * y) / x is impossible at x == 0.
        return x * Node(y)
    end
    return Node(:*, Node[x], x.value * y)
end
Base.:*(x::Number, y::Node) = iszero(y.value) ? Node(x) * y : Node(:*, Node[y], x * y.value)
Base.:+(x::Node, y::Node) = Node(:+, Node[x, y], x.value + y.value)
Base.:+(x::Node, y::Number) = Node(:+, Node[x], x.value + y)
Base.:+(x::Number, y::Node) = Node(:+, Node[y], x + y.value)
Base.:-(x::Node, y::Node) = Node(:-, Node[x, y], x.value - y.value)
Base.:-(x::Node, y::Number) = Node(:+, Node[x], x.value - y)
Base.:-(x::Number, y::Node) = Node(:-, Node[y], x - y.value)
Base.:-(x::Node) = Node(:-, Node[x], -x.value)
Base.:/(x::Node, y::Node) = Node(:/, Node[x, y], x.value / y.value)
Base.:/(x::Node, y::Number) = x * inv(y)
Base.:^(x::Node, n::Integer) = Base.power_by_squaring(x, n)
Base.tanh(x::Node) = Node(:tanh, Node[x], tanh(x.value))
Base.exp(x::Node) = Node(:exp, Node[x], exp(x.value))
Base.log(x::Node) = Node(:log, Node[x], log(x.value))

function CGE.pullback!(::typeof(+), f::Node, args::Node...)
    for arg in args
        arg.metadata.derivative += f.metadata.derivative
    end
end

function CGE.pullback!(::typeof(-), f::Node, x::Node, y::Node)
    x.metadata.derivative += f.metadata.derivative
    y.metadata.derivative -= f.metadata.derivative
end

function CGE.pullback!(::typeof(-), f::Node, x::Node)
    x.metadata.derivative -= f.metadata.derivative
end

function CGE.pullback!(::typeof(*), f::Node, x::Node, y::Node)
    x.metadata.derivative += f.metadata.derivative * y.value
    y.metadata.derivative += f.metadata.derivative * x.value
end

function CGE.pullback!(::typeof(*), f::Node, x::Node)
    if !iszero(x.value)
        x.metadata.derivative += f.metadata.derivative * f.value / x.value
    end
end

function CGE.pullback!(::typeof(/), f::Node, x::Node, y::Node)
    x.metadata.derivative += f.metadata.derivative / y.value
    y.metadata.derivative -= f.metadata.derivative * x.value / y.value^2
end

function CGE.pullback!(::typeof(tanh), f::Node, x::Node)
    x.metadata.derivative += f.metadata.derivative * (1 - tanh(x.value)^2)
end

function CGE.pullback!(::typeof(exp), f::Node, x::Node)
    x.metadata.derivative += f.metadata.derivative * exp(x.value)
end

function CGE.pullback!(::typeof(log), f::Node, x::Node)
    x.metadata.derivative += f.metadata.derivative / x.value
end

function gradient!(f, g, x)
    x_nodes = map(Node, x)
    CGE.backward!(f(x_nodes))
    return map!(node -> node.metadata.derivative, g, x_nodes)
end

gradient(f, x) = gradient!(f, zero(x), x)

end
