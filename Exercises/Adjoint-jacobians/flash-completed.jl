using LinearAlgebra, Random

# ---------- référence : attention naïve, matérialise S et P ----------
function attn_naive(Q, K, V)
    d = size(Q, 2)
    S = Q * K' ./ sqrt(d)
    P = exp.(S .- maximum(S, dims=2))
    P ./= sum(P, dims=2)
    return P * V, P
end

function attn_naive_back(Q, K, V, P, dO)
    d = size(Q, 2)
    dV = P' * dO
    dP = dO * V'
    dS = P .* (dP .- sum(P .* dP, dims=2))
    dQ = dS * K ./ sqrt(d)
    dK = dS' * Q ./ sqrt(d)
    return dQ, dK, dV
end

# ---------- flash : forward en tuiles, ne garde que O et L ----------
function attn_flash_fwd(Q, K, V; Bc=16)
    n, d = size(Q); dv = size(V, 2)
    O = zeros(eltype(Q), n, dv)
    m = fill(-Inf, n)          # max courant par ligne
    l = zeros(eltype(Q), n)    # somme courante par ligne
    for j in 1:Bc:n
        jj = j:min(j+Bc-1, n)
        S = Q * K[jj, :]' ./ sqrt(d)
        mnew = max.(m, vec(maximum(S, dims=2)))
        Pblk = exp.(S .- mnew)
        corr = exp.(m .- mnew)
        l .= corr .* l .+ vec(sum(Pblk, dims=2))
        O .= corr .* O .+ Pblk * V[jj, :]
        m .= mnew
    end
    O ./= l
    L = m .+ log.(l)           # logsumexp par ligne : O(n) stocké
    return O, L
end

# ---------- flash : backward, recalcule S et P tuile par tuile ----------
function attn_flash_back(Q, K, V, O, L, dO; Bc=16)
    n, d = size(Q)
    D = vec(sum(dO .* O, dims=2))      # = rowsum(P .* dP), sans n×n
    dQ = zeros(eltype(Q), size(Q)); dK = zeros(eltype(K), size(K)); dV = zeros(eltype(V), size(V))
    for j in 1:Bc:n
        jj = j:min(j+Bc-1, n)
        S    = Q * K[jj, :]' ./ sqrt(d)   # recalcul
        Pblk = exp.(S .- L)               # recalcul, une passe grâce à L
        dV[jj, :] .+= Pblk' * dO
        dPblk = dO * V[jj, :]'
        dSblk = Pblk .* (dPblk .- D)      # D diffusé sur les colonnes
        dQ .+= dSblk * K[jj, :] ./ sqrt(d)
        dK[jj, :] .+= dSblk' * Q ./ sqrt(d)
    end
    return dQ, dK, dV
end

using Random, LinearAlgebra
Random.seed!(0)
n, d = 37, 5
Q, K, V = randn(n, d), randn(n, d), randn(n, d)
dO = randn(n, d)

Oa, P = attn_naive(Q, K, V)
Of, L = attn_flash_fwd(Q, K, V)
@show norm(Oa - Of)
@show norm(L - vec(log.(sum(exp.(Q*K'./sqrt(d)), dims=2))))

ga = attn_naive_back(Q, K, V, P, dO)
gf = attn_flash_back(Q, K, V, Of, L, dO)
@show norm.(ga .- gf)

# vérif par différences finies sur <O, dO>
f(Q, K, V) = sum(attn_naive(Q, K, V)[1] .* dO)
eps = 1e-6
num_dQ = similar(Q)
for i in eachindex(Q)
    Qp = copy(Q); Qp[i] += eps; Qm = copy(Q); Qm[i] -= eps
    num_dQ[i] = (f(Qp, K, V) - f(Qm, K, V)) / (2eps)
end
@show norm(num_dQ - gf[1]) / norm(num_dQ)

