struct HierarchicalPeriodicHMM{F,T} <: AbstractHMM{F}
    a::Vector{T}
    A::Array{T,3}
    B::Array{<:Distribution{F},4}
    HierarchicalPeriodicHMM{F,T}(a, A, B) where {F,T} = assert_hierarchicalperiodichmm(a, A, B) && new(a, A, B)
end

HierarchicalPeriodicHMM(
    a::AbstractVector{T},
    A::AbstractArray{T,3},
    B::AbstractArray{<:Distribution{F},4},
) where {F,T} = HierarchicalPeriodicHMM{F,T}(a, A, B)

HierarchicalPeriodicHMM(A::AbstractArray{T,3}, B::AbstractArray{<:Distribution{F},4}) where {F,T} =
    HierarchicalPeriodicHMM{F,T}(ones(size(A, 1)) ./ size(A, 1), A, B)


function assert_hierarchicalperiodichmm(a::AbstractVector, A::AbstractArray{T,3} where {T}, B::AbstractArray{<:Distribution,4})
    @argcheck isprobvec(a)
    @argcheck istransmats(A)
    @argcheck size(A, 3) == size(B, 2) ArgumentError("Period length must be the same for transition matrix and distribution")
    @argcheck length(a) == size(A, 1) == size(B, 1) ArgumentError("Number of transition rates must match length of chain")
    return true
end


function assert_hierarchicalperiodichmm(hmm::HierarchicalPeriodicHMM)
    assert_periodichmm(hmm.a, hmm.A, hmm.B)
end

size(hmm::HierarchicalPeriodicHMM, dim = :) = (size(hmm.B, 1), size(hmm.B, 3), size(hmm.B, 2), size(hmm.B, 4))[dim]
# K, D, T, size_memory
copy(hmm::HierarchicalPeriodicHMM) = HierarchicalPeriodicHMM(copy(hmm.a), copy(hmm.A), copy(hmm.B))

function rand_test(
    rng::AbstractRNG,
    hmm::HierarchicalPeriodicHMM,
    n2t::AbstractArray{Int},
    useless;
    init = rand(rng, Categorical(hmm.a)),
    seq = false,
    kwargs...
)
    T = size(hmm.B, 2)
    N = length(n2t)

    z = Vector{Int}(undef, N)
    (T >= 1) && (z[1] = init)
    for n = 2:N
        tm1 = n2t[n-1] # periodic t-1
        z[n] = rand(rng, Categorical(hmm.A[z[n-1], :, tm1]))
    end
    y = rand(rng, hmm, n2t, z, useless; kwargs...)
    seq ? (z, y) : y
end

function rand_test(
    rng::AbstractRNG,
    hmm::HierarchicalPeriodicHMM,
    n2t::AbstractArray{Int},
    z::AbstractVector{<:Integer},
    yini;
    size_memory = size(hmm, 4)
)
    T = size(hmm, 2)
    max_size_memory = maximum(size_memory)
    y = Matrix{Bool}(undef, length(z), T)
    memory = Int.(log.(size_memory) / log(2))
    D = size(y, 2)

    @argcheck D == size(hmm, 2)
    @argcheck length(n2t) == length(z)

    p = zeros(D)
    for j = 1:D
        if memory[j] > 0
            # One could do some specialized for each value of memory e.g. for memory = 1, we have simply previous_day_cat = y[n-1,:].+1
            N_ini = length(yini[1:memory[j], j])
            y[1:N_ini, j] = yini[1:memory[j], j]

            for n in eachindex(z)[N_ini+1:end]
                t = n2t[n] # periodic t
                previous_day_cat = bin2digit([y[n-m, j] for m = 1:memory[j]])

                p = succprob(hmm.B[z[n], t, j, previous_day_cat])
                y[n, j] = rand(Bernoulli(p))
            end
        else
            for n in eachindex(z)[1:end]
                t = n2t[n] # periodic t
                p = succprob(hmm.B[z[n], t, j, 1])
                y[n, j] = rand(Bernoulli(p))
            end
        end
    end
    y
end

function rand(
    rng::AbstractRNG,
    hmm::HierarchicalPeriodicHMM,
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

function rand(
    rng::AbstractRNG,
    hmm::HierarchicalPeriodicHMM,
    n2t::AbstractArray{Int},
    z::AbstractVector{<:Integer},
    yini
)
    T = size(hmm, 2)
    y = Matrix{Bool}(undef, length(z), T)
    memory = Int(log(size(hmm, 4)) / log(2))
    D = size(y, 2)

    @argcheck D == size(hmm, 2)
    @argcheck length(n2t) == length(z)

    p = zeros(D)
    if memory > 0
        # One could do some specialized for each value of memory e.g. for memory = 1, we have simply previous_day_cat = y[n-1,:].+1
        N_ini = size(yini, 1)
        @argcheck N_ini == memory # Did we gave the correct number of initial conditions
        size(yini, 1) == D
        y[1:N_ini, :] = yini
        previous_day_cat = zeros(Int, D)
        for n in eachindex(z)[N_ini+1:end]
            t = n2t[n] # periodic t
            previous_day_cat[:] = bin2digit.(eachcol([y[n-m, j] for m = 1:memory, j = 1:D]))
            p[:] = succprob.(hmm.B[CartesianIndex.(z[n], t, 1:D, previous_day_cat)])
            y[n, :] = rand(Product(Bernoulli.(p)))
        end
    else
        for n in eachindex(z)[1:end]
            t = n2t[n] # periodic t
            p[:] = succprob.(hmm.B[CartesianIndex.(z[n], t, 1:D, 1)])
            y[n, :] = rand(Product(Bernoulli.(p)))
        end
    end
    y
end

rand(hmm::HierarchicalPeriodicHMM, n2t::AbstractArray{Int}, useless; kwargs...) = rand(GLOBAL_RNG, hmm, n2t, useless; kwargs...)

rand(hmm::HierarchicalPeriodicHMM, n2t::AbstractArray{Int}, z::AbstractVector{<:Integer}, useless; kwargs...) = rand(GLOBAL_RNG, hmm, n2t, z, useless; kwargs...)

rand_test(hmm::HierarchicalPeriodicHMM, n2t::AbstractArray{Int}, useless; kwargs...) = rand(GLOBAL_RNG, hmm, n2t, useless; kwargs...)

rand_test(hmm::HierarchicalPeriodicHMM, n2t::AbstractArray{Int}, z::AbstractVector{<:Integer}, useless; kwargs...) = rand(GLOBAL_RNG, hmm, n2t, z, useless; kwargs...)

function likelihoods!(L::AbstractMatrix, hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int}, lag_cat::Matrix{Int})
    N, K, T, D = size(observations, 1), size(hmm, 1), size(hmm, 3), size(hmm, 2)
    @argcheck size(L) == (N, K)

    @inbounds for i in OneTo(K), n in OneTo(N)
        t = n2t[n] # periodic t
        L[n, i] = pdf(product_distribution(hmm.B[CartesianIndex.(i, t, 1:D, lag_cat[n, :])]), observations[n, :])
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int}, lag_cat::Matrix{Int})
    N, K, T, D = size(observations, 1), size(hmm, 1), size(hmm, 3), size(hmm, 2)
    @argcheck size(LL) == (N, K)

    @inbounds for i in OneTo(K), n in OneTo(N)
        t = n2t[n] # periodic t
        LL[n, i] = logpdf(product_distribution(hmm.B[CartesianIndex.(i, t, 1:D, lag_cat[n, :])]), observations[n, :])
    end
end

####

# # In-place update of the observations distributions.
#
# quikest and most generic version -> to use
function update_B!(B::AbstractArray{T,4} where {T}, γ::AbstractMatrix, observations, estimator, idx_tj::Matrix{Vector{Vector{Int}}})
    @argcheck size(γ, 1) == size(observations, 1)
    @argcheck size(γ, 2) == size(B, 1)
    N = size(γ, 1)
    K = size(B, 1)
    T = size(B, 2)
    D = size(B, 3)
    size_memory = size(B, 4)
    ## For periodicHMM only the n observations corresponding to B(t) are used to update B(t)

    @inbounds for t in OneTo(T)
        for i in OneTo(K)
            for j = 1:D
                for m = 1:size_memory
                    if sum(γ[idx_tj[t, j][m], i]) > 0
                        B[i, t, j, m] = fit_mle(Bernoulli, observations[idx_tj[t, j][m], j], γ[idx_tj[t, j][m], i])
                    else
                        B[i, t, j, m] = Bernoulli(eps())
                    end
                end
            end
        end
    end
end

function fit_mle(hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int}; init = :none, kwargs...)
    hmm = copy(hmm)

    if init == :kmeans
        kmeans_init!(hmm, observations, display = get(kwargs, :display, :none))
    end

    history = fit_mle!(hmm, observations, n2t; kwargs...)
    hmm, history
end

function fit_mle!(
    hmm::HierarchicalPeriodicHMM,
    observations,
    n2t::AbstractArray{Int}
    ;
    display = :none,
    maxiter = 100,
    tol = 1e-3,
    robust = false,
    estimator = fit_mle)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, T, size_memory = size(observations, 1), size(hmm, 1), size(hmm, 3), size(hmm, 4)
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

    # assign category for observation depending in the past observations
    memory = Int(log(size_memory) / log(2))
    lag_cat = conditional_to(observations, memory)
    idx_tj = idx_observation_of_past_cat(lag_cat, n2t, T, K, size_memory)

    loglikelihoods!(LL, hmm, observations, n2t, lag_cat)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, ξ, α, β, LL, n2t)
        update_B!(hmm.B, γ, observations, estimator, idx_tj)

        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely observations.
        robust && (hmm.A .+= eps())

        @check isprobvec(hmm.a)
        @check istransmats(hmm.A)

        # loglikelihoods!(LL, hmm, observations, n2t)
        loglikelihoods!(LL, hmm, observations, n2t, lag_cat)

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

function forward(hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int}; logl = nothing, robust = false,
    past = [0 1 0 1 1 0 1 0 0 0
        1 1 0 1 1 1 1 1 1 1
        1 1 0 1 1 1 0 1 1 1
        1 1 0 1 1 0 0 0 1 0
        1 1 0 1 1 0 0 1 0 1])
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations, n2t; robust = robust, past = past)
    forward(hmm.a, hmm.A, LL, n2t)
end

function backward(hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int}; logl = nothing, robust = false,
    past = [0 1 0 1 1 0 1 0 0 0
        1 1 0 1 1 1 1 1 1 1
        1 1 0 1 1 1 0 1 1 1
        1 1 0 1 1 0 0 0 1 0
        1 1 0 1 1 0 0 1 0 1])
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations; robust = robust, past = past)
    backward(hmm.a, hmm.A, LL, n2t)
end

function posteriors(hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int}; logl = nothing, robust = false,
    past = [0 1 0 1 1 0 1 0 0 0
        1 1 0 1 1 1 1 1 1 1
        1 1 0 1 1 1 0 1 1 1
        1 1 0 1 1 0 0 0 1 0
        1 1 0 1 1 0 0 1 0 1])
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations, n2t; robust = robust, past = past)
    posteriors(hmm.a, hmm.A, LL, n2t)
end

function viterbi(hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int}; logl = nothing, robust = false,
    past = [0 1 0 1 1 0 1 0 0 0
        1 1 0 1 1 1 1 1 1 1
        1 1 0 1 1 1 0 1 1 1
        1 1 0 1 1 0 0 0 1 0
        1 1 0 1 1 0 0 1 0 1])
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations, n2t; robust = robust, past = past)
    viterbi(hmm.a, hmm.A, LL, n2t)
end