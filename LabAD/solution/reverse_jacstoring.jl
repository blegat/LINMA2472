module JacobianStoringReverse

import ComputationGraphExplorer as CGE

mutable struct ReverseData
    derivative::Float64
    local_jacobian::Vector{Float64}
end
CGE.metadata(::Type{ReverseData}, ::Float64) = ReverseData(0.0, Float64[])
CGE.metadata_rows(data::ReverseData) = [
    "r" => data.derivative,
    "J" => data.local_jacobian,
]

const Node = CGE.Node{Float64,ReverseData}

function CGE.seed_metadata!(data::ReverseData, is_output::Bool)
    data.derivative = is_output ? 1.0 : 0.0
end

function operation(op, args::Vector{Node}, value, local_jacobian)
    return Node(op, args, Float64(value), ReverseData(0.0, local_jacobian))
end

# The local scalar Jacobians are computed together with the primal value.
Base.zero(::Node) = Node(0.0)
Base.:*(x::Node, y::Node) = operation(:*, Node[x, y], x.value * y.value, [y.value, x.value])
Base.:*(x::Node, y::Number) = operation(:*, Node[x], x.value * y, [Float64(y)])
Base.:*(x::Number, y::Node) = operation(:*, Node[y], x * y.value, [Float64(x)])
Base.:+(x::Node, y::Node) = operation(:+, Node[x, y], x.value + y.value, [1.0, 1.0])
Base.:+(x::Node, y::Number) = operation(:+, Node[x], x.value + y, [1.0])
Base.:+(x::Number, y::Node) = operation(:+, Node[y], x + y.value, [1.0])
Base.:-(x::Node, y::Node) = operation(:-, Node[x, y], x.value - y.value, [1.0, -1.0])
Base.:-(x::Node, y::Number) = operation(:-, Node[x], x.value - y, [1.0])
Base.:-(x::Number, y::Node) = operation(:-, Node[y], x - y.value, [-1.0])
Base.:-(x::Node) = operation(:-, Node[x], -x.value, [-1.0])
Base.:/(x::Node, y::Node) = operation(:/, Node[x, y], x.value / y.value, [1 / y.value, -x.value / y.value^2])
Base.:/(x::Node, y::Number) = x * inv(y)
Base.:^(x::Node, n::Integer) = Base.power_by_squaring(x, n)
Base.tanh(x::Node) = operation(:tanh, Node[x], tanh(x.value), [1 - tanh(x.value)^2])
Base.exp(x::Node) = operation(:exp, Node[x], exp(x.value), [exp(x.value)])
Base.log(x::Node) = operation(:log, Node[x], log(x.value), [1 / x.value])

function CGE.pullback!(f::Node)
    for (arg, local_derivative) in zip(f.args, f.metadata.local_jacobian)
        arg.metadata.derivative += f.metadata.derivative * local_derivative
    end
    return f
end

function gradient!(f, g, x)
    x_nodes = map(Node, x)
    CGE.backward!(f(x_nodes))
    return map!(node -> node.metadata.derivative, g, x_nodes)
end

gradient(f, x) = gradient!(f, zero(x), x)

end
