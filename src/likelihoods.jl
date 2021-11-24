function loglikelihoods!(LL::AbstractMatrix, hmm::AbstractHMM{Univariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        LL[t, i] = logpdf(hmm.B[i], observations[t])
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::AbstractHMM{Multivariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        LL[t, i] = logpdf(hmm.B[i], view(observations, t, :))
    end
end

"""
    loglikelihoods(hmm, observations; robust) -> Matrix

Return the log-likelihood per-state and per-observation.

**Output**
- `Matrix{Float64}`: log-likelihoods matrix (`T x K`).

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000)
LL = likelihoods(hmm, y)
```
"""
function loglikelihoods(hmm::AbstractHMM, observations; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    T, K = size(observations, 1), size(hmm, 1)
    LL = Matrix{Float64}(undef, T, K)

    loglikelihoods!(LL, hmm, observations)
    if robust
        replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    end
    LL
end

function likelihoods!(L::AbstractMatrix, hmm::PeriodicHMM{Univariate}, observations, n2t::AbstractArray{Int})
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    @inbounds for i in OneTo(K), n in OneTo(T)
        t = n2t[n] # periodic t
        L[n, i] = pdf(hmm.B[i, t], observations[n])
    end
end

function likelihoods!(L::AbstractMatrix, hmm::PeriodicHMM{Multivariate}, observations, n2t::AbstractArray{Int})
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    @inbounds for i in OneTo(K), n in OneTo(T)
        t = n2t[n] # periodic t
        L[n, i] = pdf(hmm.B[i, t], view(observations, n, :))
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::PeriodicHMM{Univariate}, observations, n2t::AbstractArray{Int})
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    @inbounds for i in OneTo(K), n in OneTo(N)
        t = n2t[n] # periodic t
        LL[n, i] = logpdf(hmm.B[i, t], observations[n])
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::PeriodicHMM{Multivariate}, observations, n2t::AbstractArray{Int})
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    @inbounds for i in OneTo(K), n in OneTo(N)
        t = n2t[n] # periodic t
        LL[n, i] = logpdf(hmm.B[i, t], view(observations, n, :))
    end
end

## For AbstractHMM + n2t + lag_cat

function loglikelihoods(hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int}; logl = nothing, robust = false,
    past = [0 1 0 1 1 0 1 0 0 0
        1 1 0 1 1 1 1 1 1 1
        1 1 0 1 1 1 0 1 1 1
        1 1 0 1 1 0 0 0 1 0
        1 1 0 1 1 0 0 1 0 1])
    (logl !== nothing) && deprecate_kwargs("logl")
    N, K, size_memory = size(observations, 1), size(hmm, 1), size(hmm, 4)
    LL = Matrix{Float64}(undef, N, K)

    memory = Int(log(size_memory) / log(2))
    lag_cat = conditional_to(observations, memory; past = past)

    loglikelihoods!(LL, hmm, observations, n2t, lag_cat)
    if robust
        replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    end
    LL
end