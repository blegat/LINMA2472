"""
    struct Flatten{A}
        components::Vector{A}
    end

Lazily represent the vector obtained by concatenating the vectorization of the arrays.
That is, it represents `reduce(vcat, vec.(components))`.

## Examples

The following lazily represents the concatenation of the vectorization of two
matrices:
```julia
julia> f = Flatten(ones(2, 3), zeros(1, 2))
Flatten{Matrix{Float64}}([[1.0 1.0 1.0; 1.0 1.0 1.0], [0.0 0.0]])

julia> collect(f)
8-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 0.0
 0.0

julia> reduce(vcat, vec.(f.components))
8-element Vector{Float64}:
 1.0
 1.0
 1.0
 1.0
 1.0
 1.0
 0.0
 0.0
```
"""
struct Flatten{A}
    components::Vector{A}

    function Flatten{A}(components::Vector) where {A}
        return new{A}(components)
    end

    function Flatten(components::Vector)
        # `Flatten([ones(2, 2), zeros(2)])` would give a `Flatten{Array{Float64}}`.
        # `Array{Float64}` is abstract so it's going to cause dynamic dispatch
        # every time it is accessed. We should use the small union `VecOrMat{Float64}`
        # as Julia handles it much more efficiently, see https://julialang.org/blog/2018/08/union-splitting/
        A = Union{unique(typeof.(components))...}
        return Flatten{A}(components)
    end
end
# If we make a `Flatten` with only one argument and this argument is a vector,
# then we would expect `Flatten(args...)` to be called but it's rather
# `Flatten(::Vector)` that will be called. This additional method below fixes
# this.
function Flatten(components::Vector{<:Number})
    return Flatten([components])
end
Flatten(args...) = Flatten(collect(args))
Base.length(v::Flatten) = sum(length, v.components)
Base.size(v::Flatten) = (length(v),)
Base.eachindex(v::Flatten) = Base.axes1(v)
# We could do this but we want to allow `A` to not define
# `eltype`
#Base.eltype(::Type{Flatten{A}}) where {A} = eltype(A)
#Base.eltype(v::Flatten) = eltype(typeof(v))
Base.IteratorEltype(::Type{<:Flatten}) = Base.EltypeUnknown()
# `Flatten` is not a subtype of `AbstractVector` so the fallback for
# `reduce` does not work. Here we give up on the `Flatten` structure
# because we have no-way to represent a Matrix/2D Flatten.
function Base.reduce(::typeof(hcat), v::Vector{<:Flatten})
    reduce(hcat, collect.(v))
end

function Base.iterate(A::Flatten, state=(eachindex(A),))
    y = iterate(state...)
    y === nothing && return nothing
    A[y[1]], (state[1], Base.tail(y)...)
end

function unflatten_index(f::Flatten, i)
    k = i
    for j in eachindex(f.components)
        if 1 <= k <= length(f.components[j])
            return (j, k)
        end
        k -= length(f.components[j])
    end
    throw(BoundsError(f, i))
end

function Base.getindex(f::Flatten, i)
    j, k = unflatten_index(f, i)
    return f.components[j][k]
end

function Base.setindex!(f::Flatten, v, i)
    j, k = unflatten_index(f, i)
    return f.components[j][k] = v
end

function Base.similar(f::Flatten, ::Type{T}) where {T}
    return Flatten(similar.(f.components, T))
end

function Base.zero(f::Flatten)
    return Flatten(zero.(f.components))
end

# We have to redefine it since `dest` is not an `AbstractArray`
function Base.map!(f, dest::Flatten, A)
    for (i, j) in zip(eachindex(dest),eachindex(A))
        dest[i] = f(A[j])
    end
    return dest
end

function Base.map(op, a::Flatten, b::Flatten)
    return Flatten(map(a.components, b.components) do x, y
        map(op, x, y)
    end)
end

function Base.map(op, a::Flatten)
    return Flatten(map(a.components) do x
        map(op, x)
    end)
end
