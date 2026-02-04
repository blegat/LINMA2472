include(joinpath(@__DIR__, "flatten.jl"))

# See details about these optimisers in the following course:
# "LINMA2474 - High-Dimensional Data Analysis and Optimization"
import Optimisers

function update!(rule, state, W, ∇)
	state, Δ = Optimisers.apply!(rule, state, W, ∇)
	W .= W .- Δ
end

function update!(rule, states, w::Flatten, ∇::Flatten)
	for (state, W, ∇i) in zip(states.components, w.components, ∇.components)
		update!(rule, state, W, ∇i)
	end
end

function Optimisers.init(rule::Optimisers.AbstractRule, w::Flatten)
	return Flatten(map(W -> Optimisers.init(rule, W), w.components))
end

function train!(gradient!, L, w; rule = Optimisers.Descent(), num_iters = 10, losses = [L(w)], states = Optimisers.init(rule, w))
	g = zero(w) # preallocation
	for _ in 1:num_iters
		∇ = gradient!(L, g, w)
		update!(rule, states, w, ∇)
		push!(losses, L(w))
	end
	return losses
end
