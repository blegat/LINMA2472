module IfElseReverse

export reverse_diff

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
function Base.:*(x::Node, y::Number)
	if iszero(x.value)
		# In `_backward!`, need to recompute `y` from
		# `(x.value * y) / x.value` but this won't work
		# if `x.value` is zero so we create a node for `y`
		x * Node(y)
	else
		Node(:*, [x], x.value * y)
	end
end
function Base.:*(x::Number, y::Node)
	if iszero(y.value)
		# Same as previous method
		Node(x) * y
	else
		Node(:*, [y], x * y.value)
	end
end
Base.:+(x::Node, y::Node) = Node(:+, [x, y], x.value + y.value)
Base.:+(x::Node, y::Number) = Node(:+, [x], x.value + y)
Base.:+(x::Number, y::Node) = Node(:+, [y], x + y.value)
Base.:-(x::Node, y::Node) = Node(:-, [x, y], x.value - y.value)
Base.:-(x::Node, y::Number) = Node(:subtract_cst, [x], x.value - y)
Base.:-(x::Number, y::Node) = Node(:substract_node, [y], x - y.value)
Base.:-(x::Node) = Node(:-, [x], -x.value)
Base.:/(x::Node, y::Node) = Node(:/, [x, y], x.value / y.value)
Base.:/(x::Node, y::Number) = x * inv(y)
Base.:^(x::Node, n::Integer) = Base.power_by_squaring(x, n)
Base.tanh(x::Node) = Node(:tanh, [x], tanh(x.value))
Base.isless(x::Node, y::Number) = x.value < y
Base.isless(x::Number, y::Node) = x < y.value
Base.isless(x::Node, y::Node) = x.value < y.value
Base.exp(x::Node) = Node(:exp, [x], exp(x.value))
Base.log(x::Node) = Node(:log, [x], log(x.value))
function Base.show(io::IO, node::Node)
    print(io, "Node(")
    print(io, node.op)
    if !isempty(node.args)
        print(io, ", [")
        for (i, arg) in enumerate(node.args)
            show(io, arg)
            if i < length(node.args)
                print(io, ", ")
            end
        end
        print(io, "]")
    end
    print(io, ")")
end

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
	elseif f.op == :subtract_cst && length(f.args) == 1
		f.args[1].derivative += f.derivative
	elseif f.op == :substract_node && length(f.args) == 1
		f.args[1].derivative -= f.derivative
	elseif f.op == :* && length(f.args) == 2
		f.args[1].derivative += f.derivative * f.args[2].value
		f.args[2].derivative += f.derivative * f.args[1].value
	elseif f.op == :* && length(f.args) == 1
		# Multiplication of `x` with a constant `a`, so `a * x`.
		# In this implementation, we don't store the local derivative
		# so we don't have `a`. However we do have `f.value = a * x` so we can
		# recompute `a` as `f.value / x` where `x = f.args[1].value`
		if f.args[1].value != 0
			f.args[1].derivative += f.derivative * f.value / f.args[1].value
		end
	elseif f.op == :tanh
		f.args[1].derivative += f.derivative * (1 - tanh(f.args[1].value)^2)
	elseif f.op == :exp
		f.args[1].derivative += f.derivative * exp(f.args[1].value)
	elseif f.op == :log
		f.args[1].derivative += f.derivative / f.args[1].value
	elseif f.op == :/
		f.args[1].derivative += f.derivative / f.args[2].value
		f.args[2].derivative -= f.derivative * f.args[1].value / f.args[2].value^2
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
