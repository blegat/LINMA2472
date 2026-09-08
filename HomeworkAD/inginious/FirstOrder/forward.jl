module Forward

# We need `Union{Float64,Dual}` for the hessian and hessian-vector product
struct Dual
    value::Union{Float64,Dual}
    derivative::Union{Float64,Dual}
end
Dual(x::Number, y::Number) = Dual(Float64(x), Float64(y))

Base.broadcastable(d::Dual) = Ref(d)
Base.zero(::Dual) = Dual(0, 0)
Base.zero(::Type{Dual}) = Dual(0, 0)
Base.one(::Dual) = Dual(1, 0)
# Addition and subtraction
Base.:+(x::Dual, y::Dual) = Dual(x.value + y.value, x.derivative + y.derivative)
Base.:+(x::Dual, y::Number) = Dual(x.value + y, x.derivative)
Base.:+(x::Number, y::Dual) = Dual(x + y.value, y.derivative)
Base.:-(x::Dual, y::Dual) = Dual(x.value - y.value, x.derivative - y.derivative)
Base.:-(x::Dual, y::Number) = Dual(x.value - y, x.derivative)
Base.:-(x::Number, y::Dual) = Dual(x - y.value, -y.derivative)
Base.:-(x::Dual) = Dual(-x.value, -x.derivative)
# Scalar multiplication and division
Base.:*(α::Number, x::Dual) = Dual(α * x.value, α * x.derivative)
Base.:*(x::Dual, α::Number) = Dual(x.value * α, x.derivative * α)
Base.:/(x::Dual, α::Number) = Dual(x.value / α, x.derivative / α)
# Dual multiplication, division and power
Base.:*(x::Dual, y::Dual) = Dual(x.value * y.value, x.value * y.derivative + x.derivative * y.value)
Base.:/(x::Dual, y::Dual) = Dual(x.value / y.value, (x.derivative * y.value - x.value * y.derivative) / y.value^2)
Base.:^(x::Dual, n::Integer) = Base.power_by_squaring(x, n)
# Specific functions and operations
Base.tanh(x::Dual) = Dual(tanh(x.value), (1 - tanh(x.value)^2) * x.derivative)
Base.:exp(x::Dual) = Dual(exp(x.value), exp(x.value) * x.derivative)
Base.:log(x::Dual) = Dual(log(x.value), x.derivative / x.value)
# Solution 
Base.isless(x::Dual, y::Number) = x.value < y
Base.isless(x::Number, y::Dual) = x < y.value
Base.isless(x::Dual, y::Dual) = x.value < y.value
Base.show(io::IO, d::Dual) = print(io, "Dual(", d.value, ", ", d.derivative, ")")

function onehot(v, i)
    z = zero(v)
    z[i] = one(z[i])
    return z
end

function gradient(f, x, i::Integer)
    dx = map(Dual, x, onehot(x, i))
    return f(dx).derivative
end

function gradient!(f, g, x)
    return map!(g, eachindex(x)) do i
        gradient(f, x, i)
    end
end

gradient(f, x) = gradient!(f, zero(x), x)

# Jacobian-vector product `J(x) * tx`
function pushforward(f, x, tx)
    dW = map(Dual, x, tx)
    return map(y -> y.derivative, f(dW))
end

function jacobian(f, x, i::Integer)
    return pushforward(f, x, onehot(x, i))
end

# We don't know in advance the dimension of the output of `F`
# so we cannot easily redirect to a `jacobian!`
function jacobian(f, x)
    return reduce(hcat, map(i -> jacobian(f, x, i), eachindex(x)))
end

function hessian(f, x)
    return jacobian(z -> gradient(f, z), x)
end

# Hessian-vector product
function hvp(f, x, tx)
    return pushforward(z -> gradient(f, z), x, tx)
end

end
