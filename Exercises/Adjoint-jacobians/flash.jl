using LinearAlgebra, Random, Test

# Student version.
# Replace each error("TODO ...") expression with the missing Julia expression.
# Notation in the handout:
# dO = r_0, dP = r_1, dS = r_2, dQ = r_3, dK = r_4, dV = r_5.
# Julia broadcasts a vector of length n along the columns of an n-by-b matrix.

# ---------- Reference: naive forward attention ----------
function attn_naive(Q, K, V)
    d = size(Q, 2)
    S = Q * K' ./ sqrt(d)
    P = exp.(S .- maximum(S, dims=2))
    P ./= sum(P, dims=2)
    return P * V, P
end

# ---------- Flash forward pass ----------
function attn_flash_fwd(Q, K, V; Bc=16)
    n, d = size(Q)
    dv = size(V, 2)
    O = zeros(eltype(Q), n, dv)  # Unnormalized output accumulator.
    m = fill(-Inf, n)           # Running row maxima.
    l = zeros(eltype(Q), n)     # Running sums relative to m.

    for j in 1:Bc:n
        jj = j:min(j + Bc - 1, n)
        S = error("TODO F1: scores for the current tile")
        mnew = error("TODO F2: updated row maxima")
        Pblk = error("TODO F3: exponentials relative to mnew (not normalized)")
        corr = error("TODO F4: rescaling factor for previous contributions")
        l .= error("TODO F5: updated running sums")
        O .= error("TODO F6: updated unnormalized output")
        m .= mnew
    end

    O ./= error("TODO F7: final row normalization")
    L = error("TODO F8: row log-sum-exp values")
    return O, L
end

# ---------- Flash backward pass ----------
function attn_flash_back(Q, K, V, O, L, dO; Bc=16)
    n, d = size(Q)
    D = error("TODO B1: compute the row correction from dO and O")
    dQ = zeros(eltype(Q), size(Q))
    dK = zeros(eltype(K), size(K))
    dV = zeros(eltype(V), size(V))

    for j in 1:Bc:n
        jj = j:min(j + Bc - 1, n)
        S = error("TODO B2: recompute the tile scores")
        Pblk = error("TODO B3: reconstruct normalized probabilities using L")
        dV[jj, :] .+= error("TODO B4: contribution to the gradient of V")
        dPblk = error("TODO B5: gradient with respect to the probability tile")
        dSblk = error("TODO B6: apply the row-wise softmax adjoint")
        dQ .+= error("TODO B7: contribution to the gradient of Q")
        dK[jj, :] .+= error("TODO B8: contribution to the gradient of K")
    end

    return dQ, dK, dV
end

# ---------- Provided checks: run after completing the TODOs ----------
# Finite differences of the scalar function <O, dO>.
function finite_difference_gradient(f, X; h=1e-6)
    G = similar(X)
    for i in eachindex(X)
        Xp, Xm = copy(X), copy(X)
        Xp[i] += h
        Xm[i] -= h
        G[i] = (f(Xp) - f(Xm)) / (2h)
    end
    return G
end

function run_checks()
    Random.seed!(0)
    n, d = 37, 5  # The last tile is smaller than Bc.
    Q, K, V = randn(n, d), randn(n, d), randn(n, d)
    dO = randn(n, d)

    Oa, _ = attn_naive(Q, K, V)
    Of, L = attn_flash_fwd(Q, K, V)
    @test isapprox(Of, Oa; rtol=1e-10, atol=1e-10)

    S = Q * K' ./ sqrt(d)
    row_max = vec(maximum(S, dims=2))
    Lref = row_max .+ log.(vec(sum(exp.(S .- row_max), dims=2)))
    @test isapprox(L, Lref; rtol=1e-10, atol=1e-10)

    dQ, dK, dV = attn_flash_back(Q, K, V, Of, L, dO)
    f(Q, K, V) = sum(attn_naive(Q, K, V)[1] .* dO)
    num_dQ = finite_difference_gradient(X -> f(X, K, V), Q)
    num_dK = finite_difference_gradient(X -> f(Q, X, V), K)
    num_dV = finite_difference_gradient(X -> f(Q, K, X), V)

    @test isapprox(dQ, num_dQ; rtol=1e-5, atol=1e-7)
    @test isapprox(dK, num_dK; rtol=1e-5, atol=1e-7)
    @test isapprox(dV, num_dV; rtol=1e-5, atol=1e-7)
    println("Forward, log-sum-exp, and all three gradient checks passed.")
end

run_checks()