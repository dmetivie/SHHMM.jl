struct PeriodicHMM{F,T} <: AbstractHMM{F}
    a::Vector{T}
    A::Array{T,3}
    B::Matrix{<:Distribution{F}}
    PeriodicHMM{F,T}(a, A, B) where {F,T} = assert_periodichmm(a, A, B) && new(a, A, B)
end

PeriodicHMM(
    a::AbstractVector{T},
    A::AbstractArray{T,3},
    B::AbstractMatrix{<:Distribution{F}},
) where {F,T} = PeriodicHMM{F,T}(a, A, B)

PeriodicHMM(A::AbstractArray{T,3}, B::AbstractMatrix{<:Distribution{F}}) where {F,T} =
    PeriodicHMM{F,T}(ones(size(A, 1)) ./ size(A, 1), A, B)

function assert_periodichmm(a::AbstractVector, A::AbstractArray{T,3} where {T}, B::AbstractMatrix{<:Distribution})
    @argcheck isprobvec(a)
    @argcheck istransmats(A)
    @argcheck all(length.(B) .== length(B[1])) ArgumentError("All distributions must have the same dimensions")
    @argcheck size(A, 3) == size(B, 2) ArgumentError("Period length must be the same for transition matrix and distribution")
    @argcheck length(a) == size(A, 1) == size(B, 1) ArgumentError("Number of transition rates must match length of chain")
    return true
end


function assert_periodichmm(hmm::PeriodicHMM)
    assert_periodichmm(hmm.a, hmm.A, hmm.B)
end

istransmats(A::AbstractArray{T,3} where {T}) = all(i -> istransmat(@view A[:, :, i]), 1:size(A, 3))

# ==(h1::PeriodicHMM, h2::PeriodicHMM) = (h1.a == h2.a) && (h1.A == h2.A) && (h1.B == h2.B)

function rand(
    rng::AbstractRNG,
    hmm::PeriodicHMM,
    N::Integer;
    init = rand(rng, Categorical(hmm.a)),
    seq = false
)
    T = size(hmm.B, 2)
    z = Vector{Int}(undef, N)
    (T >= 1) && (z[1] = init)
    n2t = CyclicArray(1:T, "1D")
    for n = 2:N
        tm1 = n2t[n-1] # periodic t-1
        z[n] = rand(rng, Categorical(hmm.A[z[n-1], :, tm1]))
    end
    y = rand(rng, hmm, z)
    seq ? (z, y) : y
end

function rand(rng::AbstractRNG, hmm::PeriodicHMM{Univariate}, z::AbstractVector{<:Integer})
    y = Vector{Float64}(undef, length(z))
    T = size(hmm.B, 2)
    n2t = CyclicArray(1:T, "1D")
    for n in eachindex(z)
        t = n2t[n] # periodic t
        y[n] = rand(rng, hmm.B[z[n], t])
    end
    y
end

function rand(
    rng::AbstractRNG,
    hmm::PeriodicHMM{Multivariate},
    z::AbstractVector{<:Integer},
)
    y = Matrix{Float64}(undef, length(z), size(hmm, 2))
    T = size(hmm.B, 2)
    n2t = CyclicArray(1:T, "1D")
    for n in eachindex(z)
        t = n2t[n] # periodic t
        y[n, :] = rand(rng, hmm.B[z[n], t])
    end
    y
end

# should not be necessary with AbstractHMM type
# rand(hmm::PeriodicHMM, N::Integer; kwargs...) = rand(GLOBAL_RNG, hmm, N; kwargs...)
#
# rand(hmm::PeriodicHMM, z::AbstractVector{<:Integer}) = rand(GLOBAL_RNG, hmm, z)

"""
    size(hmm, [dim]) -> Int | Tuple
Return the number of states in `hmm`, the dimension of the observations and the length of the chain.
"""
size(hmm::PeriodicHMM, dim = :) = (size(hmm.B, 1), length(hmm.B[1]), size(hmm.B, 2))[dim]

copy(hmm::PeriodicHMM) = PeriodicHMM(copy(hmm.a), copy(hmm.A), copy(hmm.B))

function nparams(hmm::PeriodicHMM)
    (length(hmm.a) - 1) + (size(hmm.A, 1) * size(hmm.A, 2) - size(hmm.A, 1)) * size(hmm.A, 3)
end


###
function likelihoods!(L::AbstractMatrix, hmm::PeriodicHMM{Univariate}, observations)
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    n2t = CyclicArray(1:T, "1D")
    @inbounds for i in OneTo(K), n in OneTo(T)
        t = n2t[n] # periodic t
        L[n, i] = pdf(hmm.B[i, t], observations[n])
    end
end

function likelihoods!(L::AbstractMatrix, hmm::PeriodicHMM{Multivariate}, observations)
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    n2t = CyclicArray(1:T, "1D")
    @inbounds for i in OneTo(K), n in OneTo(T)
        t = n2t[n] # periodic t
        L[n, i] = pdf(hmm.B[i, t], view(observations, n, :))
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::PeriodicHMM{Univariate}, observations)
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    n2t = CyclicArray(1:T, "1D")
    @inbounds for i in OneTo(K), n in OneTo(N)
        t = n2t[n] # periodic t
        LL[n, i] = logpdf(hmm.B[i, t], observations[n])
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::PeriodicHMM{Multivariate}, observations)
    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck size(LL) == (N, K)
    n2t = CyclicArray(1:T, "1D")
    @inbounds for i in OneTo(K), n in OneTo(N)
        t = n2t[n] # periodic t
        LL[n, i] = logpdf(hmm.B[i, t], view(observations, n, :))
    end
end

####

function update_A!(
    A::AbstractArray{T,3} where {T},
    ??::AbstractArray,
    ??::AbstractMatrix,
    ??::AbstractMatrix,
    LL::AbstractMatrix,
)
    @argcheck size(??, 1) == size(??, 1) == size(LL, 1) == size(??, 1)
    @argcheck size(??, 2) ==
              size(??, 2) ==
              size(LL, 2) ==
              size(A, 1) ==
              size(A, 2) ==
              size(??, 2) ==
              size(??, 3)

    N, K = size(LL)
    T = size(A, 3)
    n2t = CyclicArray(1:T, "1D")
    @inbounds for n in OneTo(N - 1)
        t = n2t[n] # periodic t
        m = vec_maximum(view(LL, n + 1, :))
        c = 0.0

        for i in OneTo(K), j in OneTo(K)
            ??[n, i, j] = ??[n, i] * A[i, j, t] * exp(LL[n+1, j] - m) * ??[n+1, j]
            c += ??[n, i, j]
        end

        for i in OneTo(K), j in OneTo(K)
            ??[n, i, j] /= c
        end
    end

    fill!(A, 0.0)
    ## For periodicHMM only the n observation corresponding to A(t) are used to update A(t)
    t_n = n2t[1:N]
    n_in_t = [findall(t_n .== t) for t = 1:T] # could probably be speeded up

    @inbounds for t in OneTo(T)
        for i in OneTo(K)
            c = 0.0

            for j in OneTo(K)
                for n in setdiff(n_in_t[t], N)
                    A[i, j, t] += ??[n, i, j]
                end
                c += A[i, j, t]
            end

            for j in OneTo(K)
                A[i, j, t] /= c
            end
        end
    end
end

# # In-place update of the observations distributions.
function update_B!(B::AbstractMatrix, ??::AbstractMatrix, observations, estimator)
    @argcheck size(??, 1) == size(observations, 1)
    @argcheck size(??, 2) == size(B, 1)
    N = size(??, 1)
    K = size(B, 1)
    T = size(B, 2)
    ## For periodicHMM only the n observation corresponding to A(t) are used to update A(t)
    n2t = CyclicArray(1:T, "1D")
    t_n = n2t[1:N]
    n_in_t = [findall(t_n .== t) for t = 1:T] # could probably be speeded up. For exemple computed outside the function only once

    @inbounds for t in OneTo(T)
        n_t = n_in_t[t]
        for i in OneTo(K)
            if sum(??[n_t, i]) > 0
                B[i, t] = estimator(typeof(B[i, t]), permutedims(observations[n_t, :]), ??[n_t, i])
            end
        end
    end
end

function fit_mle!(
    hmm::PeriodicHMM,
    observations;
    display = :none,
    maxiter = 100,
    tol = 1e-3,
    robust = false,
    estimator = fit_mle
)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, T = size(observations, 1), size(hmm, 1), size(hmm, 3)
    @argcheck T == size(hmm.B, 2)
    history = EMHistory(false, 0, [])

    # Allocate memory for in-place updates
    c = zeros(N)
    ?? = zeros(N, K)
    ?? = zeros(N, K)
    ?? = zeros(N, K)
    ?? = zeros(N, K, K)
    LL = zeros(N, K)

    loglikelihoods!(LL, hmm, observations)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(??, c, hmm.a, hmm.A, LL)
    backwardlog!(??, c, hmm.a, hmm.A, LL)
    posteriors!(??, ??, ??)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, ??, ??)
        update_A!(hmm.A, ??, ??, ??, LL)
        update_B!(hmm.B, ??, observations, estimator)

        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely observations.
        robust && (hmm.A .+= eps())

        @check isprobvec(hmm.a)
        @check istransmats(hmm.A)

        loglikelihoods!(LL, hmm, observations)
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

        forwardlog!(??, c, hmm.a, hmm.A, LL)
        backwardlog!(??, c, hmm.a, hmm.A, LL)
        posteriors!(??, ??, ??)

        logtotp = sum(c)
        (display == :iter) && println("Iteration $it: logtot = $logtotp")

        push!(history.logtots, logtotp)
        history.iterations += 1

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history.converged = true
            break
        end

        logtot = logtotp
    end

    if !history.converged
        if display in [:iter, :final]
            println("EM has not converged after $(history.iterations) iterations, logtot = $logtot")
        end
    end

    history
end

# In-place forward pass, where ?? and c are allocated beforehand.
function forwardlog!(
    ??::AbstractMatrix,
    c::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractMatrix,
)
    @argcheck size(??, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(??, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    N, K = size(LL)
    T = size(A, 3)

    fill!(??, 0.0)
    fill!(c, 0.0)

    m = vec_maximum(view(LL, 1, :))

    for j in OneTo(K)
        ??[1, j] = a[j] * exp(LL[1, j] - m)
        c[1] += ??[1, j]
    end

    for j in OneTo(K)
        ??[1, j] /= c[1]
    end

    c[1] = log(c[1]) + m

    n2t = CyclicArray(1:T, "1D")
    @inbounds for n = 2:N
        tm1 = n2t[n-1] # periodic t-1
        m = vec_maximum(view(LL, n, :))

        for j in OneTo(K)
            for i in OneTo(K)
                ??[n, j] += ??[n-1, i] * A[i, j, tm1]
            end
            ??[n, j] *= exp(LL[n, j] - m)
            c[n] += ??[n, j]
        end

        for j in OneTo(K)
            ??[n, j] /= c[n]
        end

        c[n] = log(c[n]) + m
    end
end

# In-place backward pass, where ?? and c are allocated beforehand.
function backwardlog!(
    ??::AbstractMatrix,
    c::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractMatrix,
)
    @argcheck size(??, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(??, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    N, K = size(LL)
    T = size(A, 3)
    L = zeros(K)
    (T == 0) && return

    fill!(??, 0.0)
    fill!(c, 0.0)

    for j in OneTo(K)
        ??[end, j] = 1.0
    end
    n2t = CyclicArray(1:T, "1D")

    @inbounds for n = N-1:-1:1
        t = n2t[n] # periodic t
        m = vec_maximum(view(LL, n + 1, :))

        for i in OneTo(K)
            L[i] = exp(LL[n+1, i] - m)
        end

        for j in OneTo(K)
            for i in OneTo(K)
                ??[n, j] += ??[n+1, i] * A[j, i, t] * L[i]
            end
            c[n+1] += ??[n, j]
        end

        for j in OneTo(K)
            ??[n, j] /= c[n+1]
        end

        c[n+1] = log(c[n+1]) + m
    end

    m = vec_maximum(view(LL, 1, :))

    for j in OneTo(K)
        c[1] += a[j] * exp(LL[1, j] - m) * ??[1, j]
    end

    c[1] = log(c[1]) + m
end

function forward(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix)
    m = Matrix{Float64}(undef, size(LL))
    c = Vector{Float64}(undef, size(LL, 1))
    forwardlog!(m, c, a, A, LL)
    m, sum(c)
end

function backward(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix)
    m = Matrix{Float64}(undef, size(LL))
    c = Vector{Float64}(undef, size(LL, 1))
    backwardlog!(m, c, a, A, LL)
    m, sum(c)
end

function posteriors(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix; kwargs...)
    ??, _ = forward(a, A, LL; kwargs...)
    ??, _ = backward(a, A, LL; kwargs...)
    posteriors(??, ??)
end

function viterbi!(
    T1::AbstractMatrix,
    T2::AbstractMatrix,
    z::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    L::AbstractMatrix,
)
    N, K = size(L)
    T = size(A, 3)
    (T == 0) && return

    fill!(T1, 0.0)
    fill!(T2, 0)

    c = 0.0

    for i in OneTo(K)
        T1[1, i] = a[i] * L[1, i]
        c += T1[1, i]
    end

    for i in OneTo(K)
        T1[1, i] /= c
    end

    n2t = CyclicArray(1:T, "1D")
    @inbounds for n = 2:N
        tm1 = n2t[n-1] # t-1
        c = 0.0

        for j in OneTo(K)
            # TODO: If there is NaNs in T1 this may
            # stay to 0 (NaN > -Inf == false).
            # Hence it will crash when computing z[t].
            # Maybe we should check for NaNs beforehand ?
            amax = 0
            vmax = -Inf

            for i in OneTo(K)
                v = T1[n-1, i] * A[i, j, tm1]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end

            T1[n, j] = vmax * L[n, j]
            T2[n, j] = amax
            c += T1[n, j]
        end

        for i in OneTo(K)
            T1[n, i] /= c
        end
    end

    z[N] = argmax(T1[N, :])
    for n = N-1:-1:1
        z[n] = T2[n+1, z[n+1]]
    end
end

function viterbilog!(
    T1::AbstractMatrix,
    T2::AbstractMatrix,
    z::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractMatrix,
)
    N, K = size(LL)
    T = size(A, 3)
    (T == 0) && return

    fill!(T1, 0.0)
    fill!(T2, 0)

    al = log.(a)
    Al = log.(A)

    for i in OneTo(K)
        T1[1, i] = al[i] + LL[1, i]
    end
    n2t = CyclicArray(1:T, "1D")
    @inbounds for n = 2:N
        tm1 = n2t[n-1] # t-1
        for j in OneTo(K)
            amax = 0
            vmax = -Inf

            for i in OneTo(K)
                v = T1[n-1, i] + Al[i, j, tm1]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end

            T1[n, j] = vmax + LL[n, j]
            T2[n, j] = amax
        end
    end

    z[N] = argmax(T1[N, :])
    for n = N-1:-1:1
        z[n] = T2[n+1, z[n+1]]
    end
end
