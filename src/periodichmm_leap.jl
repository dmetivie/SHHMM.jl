function HierarchicalPeriodicHMM_trig(θ_Q::AbstractArray, θ_Y::AbstractArray, T::Int)
    K, D, size_memory = size(θ_Y)
    @argcheck K == size(θ_Q, 1)

    A = zeros(K, K, T)
    [A[k, l, t] = exp(polynomial_trigo(t, θ_Q[k, l, :], T = T)) for k = 1:K, l = 1:K-1, t = 1:T]
    [A[k, K, t] = 1 for k = 1:K, t = 1:T] # last colum is 1/normalization (one could do otherwise)
    normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
    [A[k, l, t] /= normalization_polynomial[k, t] for k = 1:K, l = 1:K, t = 1:T]

    p = [1 / (1 + exp(polynomial_trigo(t, θ_Y[k, s, h, :], T = T))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_memory]

    return HierarchicalPeriodicHMM(A, Bernoulli.(p))
end

function HierarchicalPeriodicHMM_trig(a::AbstractVector, θ_Q::AbstractArray, θ_Y::AbstractArray, T::Int)
    K, D, size_memory = size(θ_Y)
    @assert K == size(θ_Q, 1)

    A = zeros(K, K, T)
    [A[k, l, t] = exp(polynomial_trigo(t, θ_Q[k, l, :], T = T)) for k = 1:K, l = 1:K-1, t = 1:T]
    [A[k, K, t] = 1 for k = 1:K, t = 1:T] # last colum is 1/normalization (one could do otherwise)
    normalization_polynomial = [1 + sum(A[k, l, t] for l = 1:K-1) for k = 1:K, t = 1:T]
    [A[k, l, t] /= normalization_polynomial[k, t] for k = 1:K, l = 1:K, t = 1:T]

    p = [1 / (1 + exp(polynomial_trigo(t, θ_Y[k, s, h, :], T = T))) for k = 1:K, t = 1:T, s = 1:D, h = 1:size_memory]

    return HierarchicalPeriodicHMM(a, A, Bernoulli.(p))
end

function rand(
    rng::AbstractRNG,
    hmm::PeriodicHMM,
    n2t::AbstractArray{Int},
    useless;
    init = rand(rng, Categorical(hmm.a)),
    seq = false
)
    T = size(hmm.B, 2)
    N = length(n2t)

    z = Vector{Int}(undef, N)
    (T >= 1) && (z[1] = init)
    for n = 2:N
        tm1 = n2t[n-1] # periodic t-1
        z[n] = rand(rng, Categorical(hmm.A[z[n-1], :, tm1]))
    end
    y = rand(rng, hmm, n2t, z, useless)
    seq ? (z, y) : y
end

function rand(rng::AbstractRNG, hmm::PeriodicHMM{Univariate}, n2t::AbstractArray{Int}, z::AbstractVector{<:Integer}, useless::String)
    y = Vector(undef, length(z))
    T = size(hmm.B, 2)
    @argcheck length(n2t) == length(z)
    for n in eachindex(z)
        t = n2t[n] # periodic t
        y[n] = rand(rng, hmm.B[z[n], t])
    end
    y
end

function rand(rng::AbstractRNG, hmm::PeriodicHMM{Univariate}, n2t::AbstractArray{Int}, z::AbstractVector{<:Integer}, yini)
    y = Vector(undef, length(z))
    T = size(hmm.B, 2)
    @argcheck length(n2t) == length(z)
    y[1] = yini
    for n in eachindex(z)
        t = n2t[n] # periodic t
        p = yesterday.(params.(hmm.B[z[n], t].v), y[n-1])
        y[n] = rand(rng, Product(Bernoulli.(p)))
    end
    y
end

function rand(
    rng::AbstractRNG,
    hmm::PeriodicHMM{Multivariate},
    n2t::AbstractArray{Int},
    z::AbstractVector{<:Integer},
    useless::String
)
    y = Matrix{eltype(hmm.B[1])}(undef, length(z), size(hmm, 2))
    T = size(hmm.B, 2)
    @argcheck length(n2t) == length(z)
    for n in eachindex(z)
        t = n2t[n] # periodic t
        y[n, :] = rand(rng, hmm.B[z[n], t])
    end
    y
end

function rand(
    rng::AbstractRNG,
    hmm::PeriodicHMM{Multivariate},
    n2t::AbstractArray{Int},
    z::AbstractVector{<:Integer},
    yini
)
    T = size(hmm, 2)
    y = Matrix{eltype(hmm.B[1])}(undef, length(z), T)
    @argcheck length(n2t) == length(z)
    N_ini = size(yini, 2)
    y[1:N_ini, :] = yini
    for n in eachindex(z)[N_ini+1:end]
        t = n2t[n] # periodic t
        p = yesterday.(params.(hmm.B[z[n], t].v), y[n-1, :])
        y[n, :] = rand(rng, Product(Bernoulli.(p)))
    end
    y
end

rand(hmm::PeriodicHMM, n2t::AbstractArray{Int}, useless; kwargs...) = rand(GLOBAL_RNG, hmm, n2t, useless; kwargs...)

rand(hmm::PeriodicHMM, n2t::AbstractArray{Int}, z::AbstractVector{<:Integer}, useless; kwargs...) = rand(GLOBAL_RNG, hmm, n2t, z, useless; kwargs...)


function yesterday(probj::Tuple, yj)
    probj = probj[1]
    probj[day2(yj, 1)] / (probj[day2(yj, 1)] + probj[day2(yj, 0)]) #1-> dd, 2-> wd, 3 -> dw, 4 -> ww
end
day3(b, y, t) = passmissing(Int)(round(1 + 1.4b + 2.4y + 3.4t))
day2(y, t) = (1 + y + 2t)

###
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

####

function update_A!(
    A::AbstractArray{T,3} where {T},
    ξ::AbstractArray,
    α::AbstractMatrix,
    β::AbstractMatrix,
    LL::AbstractMatrix,
    n2t::AbstractArray{Int}
)
    @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1)
    @argcheck size(α, 2) ==
              size(β, 2) ==
              size(LL, 2) ==
              size(A, 1) ==
              size(A, 2) ==
              size(ξ, 2) ==
              size(ξ, 3)

    N, K = size(LL)
    T = size(A, 3)
    @inbounds for n in OneTo(N - 1)
        t = n2t[n] # periodic t
        m = vec_maximum(view(LL, n + 1, :))
        c = 0.0

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] = α[n, i] * A[i, j, t] * exp(LL[n+1, j] - m) * β[n+1, j]
            c += ξ[n, i, j]
        end

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] /= c
        end
    end

    fill!(A, 0.0)
    ## For periodicHMM only the n observation corresponding to A(t) are used to update A(t)
    n_in_t = [findall(n2t .== t) for t = 1:T] # could probably be speeded up

    @inbounds for t in OneTo(T)
        for i in OneTo(K)
            c = 0.0

            for j in OneTo(K)
                for n in setdiff(n_in_t[t], N)
                    A[i, j, t] += ξ[n, i, j]
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
function update_B!(B::AbstractMatrix, γ::AbstractMatrix, observations, estimator, n2t::AbstractArray{Int})
    @argcheck size(γ, 1) == size(observations, 1)
    @argcheck size(γ, 2) == size(B, 1)
    @argcheck size(observations, 1) == size(n2t, 1)
    N = size(γ, 1)
    K = size(B, 1)
    T = size(B, 2)
    ## For periodicHMM only the n observation corresponding to A(t) are used to update A(t)
    C = ncategories(B[1].v[1])
    n_in_t = [findall(n2t .== t) for t = 1:T] # could probably be speeded up. For exemple computed outside the function only once
    if typeof(B[1]) in [Product{Discrete,Categorical{Float64,Vector{Float64}},Vector{Categorical{Float64,Vector{Float64}}}}, Categorical{Float64,Vector{Float64}}]
        @inbounds for t in OneTo(T)
            n_t = n_in_t[t]
            for i in OneTo(K)
                if sum(γ[n_t, i]) > 0
                    B[i, t] = fit_mle((B[i, t]), C, permutedims(observations[n_t, :]), γ[n_t, i])
                end
            end
        end
    else
        @inbounds for t in OneTo(T)
            n_t = n_in_t[t]
            for i in OneTo(K)
                if sum(γ[n_t, i]) > 0
                    B[i, t] = estimator((B[i, t]), permutedims(observations[n_t, :]), γ[n_t, i])
                end
            end
        end
    end
end

function fit_mle(hmm::PeriodicHMM, observations, n2t::AbstractArray{Int}; init = :none, kwargs...)
    hmm = copy(hmm)

    if init == :kmeans
        kmeans_init!(hmm, observations, display = get(kwargs, :display, :none))
    end

    history = fit_mle!(hmm, observations, n2t; kwargs...)
    hmm, history
end

function fit_mle!(
    hmm::PeriodicHMM,
    observations,
    n2t::AbstractArray{Int}
    ;
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

    # n2t = date # dayofyear_Leap.(date)

    # Allocate memory for in-place updates
    c = zeros(N)
    α = zeros(N, K)
    β = zeros(N, K)
    γ = zeros(N, K)
    ξ = zeros(N, K, K)
    LL = zeros(N, K)

    loglikelihoods!(LL, hmm, observations, n2t)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, ξ, α, β, LL, n2t)
        update_B!(hmm.B, γ, observations, estimator, n2t)

        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely observations.
        robust && (hmm.A .+= eps())

        @check isprobvec(hmm.a)
        @check istransmats(hmm.A)

        loglikelihoods!(LL, hmm, observations, n2t)
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))


        forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
        backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
        posteriors!(γ, α, β)

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

# In-place forward pass, where α and c are allocated beforehand.
function forwardlog!(
    α::AbstractMatrix,
    c::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractMatrix,
    n2t::AbstractArray{Int}
)
    @argcheck size(α, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(α, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    N, K = size(LL)
    T = size(A, 3)

    fill!(α, 0.0)
    fill!(c, 0.0)

    m = vec_maximum(view(LL, 1, :))

    for j in OneTo(K)
        α[1, j] = a[j] * exp(LL[1, j] - m)
        c[1] += α[1, j]
    end

    for j in OneTo(K)
        α[1, j] /= c[1]
    end

    c[1] = log(c[1]) + m

    for n = 2:N
        tm1 = n2t[n-1] # periodic t-1
        m = vec_maximum(view(LL, n, :))
        #
        for j in OneTo(K)
            for i in OneTo(K)
                α[n, j] += α[n-1, i] * A[i, j, tm1]
            end
            α[n, j] *= exp(LL[n, j] - m)
            c[n] += α[n, j]
        end
        #
        for j in OneTo(K)
            α[n, j] /= c[n]
        end
        #
        c[n] = log(c[n]) + m
    end
end

# In-place backward pass, where β and c are allocated beforehand.
function backwardlog!(
    β::AbstractMatrix,
    c::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractMatrix,
    n2t::AbstractArray{Int}
)
    @argcheck size(β, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(β, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    N, K = size(LL)
    T = size(A, 3)
    L = zeros(K)
    (T == 0) && return

    fill!(β, 0.0)
    fill!(c, 0.0)

    for j in OneTo(K)
        β[end, j] = 1.0
    end

    @inbounds for n = N-1:-1:1
        t = n2t[n] # periodic t
        m = vec_maximum(view(LL, n + 1, :))

        for i in OneTo(K)
            L[i] = exp(LL[n+1, i] - m)
        end

        for j in OneTo(K)
            for i in OneTo(K)
                β[n, j] += β[n+1, i] * A[j, i, t] * L[i]
            end
            c[n+1] += β[n, j]
        end

        for j in OneTo(K)
            β[n, j] /= c[n+1]
        end

        c[n+1] = log(c[n+1]) + m
    end

    m = vec_maximum(view(LL, 1, :))

    for j in OneTo(K)
        c[1] += a[j] * exp(LL[1, j] - m) * β[1, j]
    end

    c[1] = log(c[1]) + m
end
## For Array
function forward(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix, n2t::AbstractArray{Int})
    m = Matrix{Float64}(undef, size(LL))
    c = Vector{Float64}(undef, size(LL, 1))
    forwardlog!(m, c, a, A, LL, n2t)
    m, sum(c)
end

function backward(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix, n2t::AbstractArray{Int})
    m = Matrix{Float64}(undef, size(LL))
    c = Vector{Float64}(undef, size(LL, 1))
    backwardlog!(m, c, a, A, LL, n2t)
    m, sum(c)
end

function posteriors(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix, n2t::AbstractArray{Int}; kwargs...)
    α, _ = forward(a, A, LL, n2t; kwargs...)
    β, _ = backward(a, A, LL, n2t; kwargs...)
    posteriors(α, β)
end
## For AbstractHMM + n2t (for inhomogeneous HMM with date or period)

function loglikelihoods(hmm::AbstractHMM, observations, n2t::AbstractArray{Int}; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    N, K = size(observations, 1), size(hmm, 1)
    LL = Matrix{Float64}(undef, N, K)

    loglikelihoods!(LL, hmm, observations, n2t)
    if robust
        replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    end
    LL
end

function loglikelihood(hmm::AbstractHMM, observations, n2t::AbstractArray{Int}; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    forward(hmm, observations, n2t, robust = robust)[2]
end

function forward(hmm::AbstractHMM, observations, n2t::AbstractArray{Int}; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations, n2t; robust = robust)
    forward(hmm.a, hmm.A, LL, n2t)
end

function backward(hmm::AbstractHMM, observations, n2t::AbstractArray{Int}; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations; robust = robust)
    backward(hmm.a, hmm.A, LL, n2t)
end

function posteriors(hmm::AbstractHMM, observations, n2t::AbstractArray{Int}; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations, n2t; robust = robust)
    posteriors(hmm.a, hmm.A, LL, n2t)
end

function viterbi(hmm::AbstractHMM, observations, n2t::AbstractArray{Int}; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations, n2t; robust = robust)
    viterbi(hmm.a, hmm.A, LL, n2t)
end

## Viterbi
function viterbi(a::AbstractVector, A::AbstractArray, LL::AbstractMatrix, n2t::AbstractArray{Int}; logl = nothing)
    ## < v1.1 compatibility
    (logl !== nothing) && deprecate_kwargs("logl")
    (logl == false) && (LL = log.(LL))
    ## --------------------
    T1 = Matrix{Float64}(undef, size(LL))
    T2 = Matrix{Int}(undef, size(LL))
    z = Vector{Int}(undef, size(LL, 1))
    viterbilog!(T1, T2, z, a, A, LL, n2t::AbstractArray{Int})
    z
end

function viterbi!(
    T1::AbstractMatrix,
    T2::AbstractMatrix,
    z::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    L::AbstractMatrix,
    n2t::AbstractArray{Int}
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
    n2t::AbstractArray{Int}
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
