module SimpleReverse

mutable struct Node
    op::Union{Nothing,Symbol}
    args::Vector{Node}
    value::Float64
    derivative::Float64
end
Node(op, args, value) = Node(op, args, value, 0.0)
Node(value) = Node(nothing, Node[], value)
Base.zero(::Node) = Node(0)
Base.:*(x::Node, y::Node) = Node(:*, [x, y], x.value * y.value)
Base.:*(x::Node, y::Number) = x * Node(y)
Base.:*(x::Number, y::Node) = Node(x) * y
Base.:+(x::Node, y::Node) = Node(:+, [x, y], x.value + y.value)
Base.:-(x::Node, y::Node) = Node(:-, [x, y], x.value - y.value)
Base.:-(x::Node, y::Number) = x - Node(y)
Base.:-(x::Node) = Node(:-, [x], -x.value)
Base.:/(x::Node, y::Node) = Node(:/, [x, y], x.value / y.value)
Base.:/(x::Node, y::Number) = x * Node(inv(y))
Base.:^(x::Node, n::Integer) = Base.power_by_squaring(x, n)
Base.tanh(x::Node) = Node(:tanh, [x], tanh(x.value))
Base.exp(x::Node) = Node(:exp, [x], exp(x.value))
Base.log(x::Node) = Node(:log, [x], log(x.value))
Base.isless(x::Node, y::Number) = x.value < y
Base.isless(x::Number, y::Node) = x < y.value
Base.isless(x::Node, y::Node) = x.value < y.value

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
	elseif f.op == :+
		for arg in f.args
			arg.derivative += f.derivative
		end
	elseif f.op == :- && length(f.args) == 2
		f.args[1].derivative += f.derivative
		f.args[2].derivative -= f.derivative
	elseif f.op == :- && length(f.args) == 1
		f.args[1].derivative -= f.derivative
	elseif f.op == :* && length(f.args) == 2
		f.args[1].derivative += f.derivative * f.args[2].value
		f.args[2].derivative += f.derivative * f.args[1].value
	elseif f.op == :/ && length(f.args) == 2
		f.args[1].derivative += f.derivative / f.args[2].value
		f.args[2].derivative -= f.derivative * f.args[1].value / f.args[2].value^2
	elseif f.op == :tanh && length(f.args) == 1
		f.args[1].derivative += f.derivative * (1 - tanh(f.args[1].value)^2)
	elseif f.op == :exp && length(f.args) == 1
		f.args[1].derivative += f.derivative * exp(f.args[1].value)
	elseif f.op == :log && length(f.args) == 1
		f.args[1].derivative += f.derivative / f.args[1].value
	else
		error("Operator $(f.op) not supported yet")
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
