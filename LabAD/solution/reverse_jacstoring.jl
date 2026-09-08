module JacobianStoringReverse

mutable struct Node
    op::Union{Nothing,Symbol}
    args::Vector{Node}
    value::Float64
	localjac::Vector{Float64} 
    derivative::Float64
end
Node(op, args, value, localjac) = Node(op, args, value, localjac, 0.0)
Node(value) = Node(nothing, Node[], value, Float64[])
Base.zero(::Node) = Node(0)
Base.:*(x::Node, y::Node) = Node(:*, [x, y], x.value * y.value, [y.value, x.value])
Base.:*(x::Node, y::Number) = Node(:*, [x], x.value * y, [y])
Base.:*(x::Number, y::Node) = Node(:*, [y], x * y.value, [x])
Base.:+(x::Node, y::Node) = Node(:+, [x, y], x.value + y.value, [1.0, 1.0])
Base.:+(x::Node, y::Number) = Node(:+, [x], x.value + y, [1.0])
Base.:+(x::Number, y::Node) = Node(:+, [y], x + y.value, [1.0])
Base.:-(x::Node, y::Node) = Node(:-, [x, y], x.value - y.value, [1.0, -1.0])
Base.:-(x::Node, y::Number) = Node(:subtract_cst, [x], x.value - y, [1.0])
Base.:-(x::Number, y::Node) = Node(:substract_node, [y], x - y.value, [-1.0])
Base.:-(x::Node) = Node(:-, [x], -x.value, [-1.0])
Base.:/(x::Node, y::Node) = Node(:/, [x, y], x.value / y.value, [1.0 / y.value, -x.value / y.value^2])
Base.:/(x::Node, y::Number) = x * inv(y)
Base.:^(x::Node, n::Integer) = Base.power_by_squaring(x, n)
Base.tanh(x::Node) = Node(:tanh, [x], tanh(x.value), [1.0 - tanh(x.value)^2])
Base.isless(x::Node, y::Number) = x.value < y
Base.isless(x::Number, y::Node) = x < y.value
Base.isless(x::Node, y::Node) = x.value < y.value
Base.exp(x::Node) = Node(:exp, [x], exp(x.value), [exp(x.value)])
Base.log(x::Node) = Node(:log, [x], log(x.value), [1.0 / x.value])

function topo_sort!(visited, topo, f::Node)
	if !(f in visited)
		push!(visited, f)
		for arg in f.args
			topo_sort!(visited, topo, arg)
		end
		push!(topo, f)
	end
end

function _backward!(f::Node)
	if isnothing(f.op)
		return
	elseif !isnothing(f.localjac)
		for (i, arg) in enumerate(f.args)
			arg.derivative += f.derivative * f.localjac[i]
		end
	else
		error("Operator `$(f.op)` not supported yet")
	end
end

function backward!(f::Node)
	topo = typeof(f)[]
	topo_sort!(Set{typeof(f)}(), topo, f)
	reverse!(topo)
	for node in topo
		node.derivative = 0
	end
	f.derivative = 1
	for node in topo
		_backward!(node)
	end
	return f
end

function gradient!(f, g, x)
	x_nodes = map(Node, x)
	expr = f(x_nodes)
	backward!(expr)
	return map!(node -> node.derivative, g, x_nodes)
end

gradient(f, x) = gradient!(f, zero(x), x)

end
