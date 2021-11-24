function fit_mle_all_slices(hmm::PeriodicHMM, observations, n2t::AbstractArray{Int};
    robust = false,
    smooth = true, window = -15:15, kernel = :step,
    history = false,
    kwargs...)

    hmm = copy(hmm)
    N = size(observations, 1)
    @argcheck size(observations, 1) == size(n2t, 1)

    K, T = size(hmm, 1), size(hmm, 3)
    α = hcat([vec(sum(hmm.A[:, :, t], dims = 1) / K) for t = 1:T]...)
    n_in_t = [findall(n2t .== t) for t = 1:T] #
    hist = Vector{HMMBase.EMHistory}(undef, T)
    cycle = CyclicArray(1:T, "1D")
    for t = 1:T
        n_in_t_extanded = sort(vcat([n_in_t[tt] for tt in cycle[[t - 12, t - 6, t, t + 6, t + 12]]]...)) # extend dataset
        hist[t] = fit_mle_B_slice!(@view(α[:, t]), @view(hmm.B[:, t]), observations[n_in_t_extanded, :]; kwargs...)
        # hist[t] = fit_mle_B_slice!(@view(α[:,t]), @view(hmm.B[:,t]), observations[n_in_t[t],:]; kwargs...)
    end

    LL = zeros(N, K)

    if smooth == true
        smooth_B = lagou(hmm.B, dims = 2, window = window, kernel = kernel)
        smooth_α = lagou(α, dims = 2, window = window, kernel = kernel)

        # evaluate likelihood for each type k
        loglikelihoods!(LL, smooth_B, observations, n2t)
        [LL[n, k] += log(smooth_α[k, n2t[n]]) for k = 1:K, n = 1:N]
    else
        # evaluate likelihood for each type k
        loglikelihoods!(LL, hmm.B, observations, n2t)
        [LL[n, k] += log(α[k, n2t[n]]) for k = 1:K, n = 1:N]
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    fit_mle_A_from_slice!(hmm.a, hmm.A, LL, n2t, robust = robust)

    return history ? (hmm, hist) : hmm
end

function fit_mle_all_slices(hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int};
    robust = false,
    smooth = true, window = -15:15, kernel = :step,
    history = false,
    kwargs...)

    hmm = copy(hmm)

    N, K, T, size_memory, D = size(observations, 1), size(hmm, 1), size(hmm, 3), size(hmm, 4), size(hmm, 2)
    @argcheck size(observations, 1) == size(n2t, 1)

    # assign category for observation depending in the past observations
    memory = Int(log(size_memory) / log(2))
    lag_cat = conditional_to(observations, memory)

    α = hcat([vec(sum(hmm.A[:, :, t], dims = 1) / K) for t = 1:T]...)
    n_in_t = [findall(n2t .== t) for t = 1:T] #
    hist = Vector{SHHMM.EMHistory}(undef, T)
    cycle = CyclicArray(1:T, "1D")
    for t = 1:T
        n_in_t_extanded = sort(vcat([n_in_t[tt] for tt in cycle[[t - 12, t - 7, t, t + 6, t + 13]]]...)) # extend dataset
        hist[t] = fit_mle_B_slice!(@view(α[:, t]), @view(hmm.B[:, t, :, :]), observations[n_in_t_extanded, :], lag_cat[n_in_t_extanded, :]; kwargs...)
        # hist[t] = fit_mle_B_slice!(α[:,t], @view(hmm.B[:,t,:,:]), observations[n_in_t[t],:], lag_cat[n_in_t[t],:]; kwargs...)
    end

    LL = zeros(N, K)

    if smooth == true
        smooth_B = lagou(hmm.B, dims = 2, window = window, kernel = kernel)
        smooth_α = lagou(α, dims = 2, window = window, kernel = kernel)
        loglikelihoods!(LL, smooth_B, observations, n2t, lag_cat)
        [LL[n, k] += log(smooth_α[k, n2t[n]]) for k = 1:K, n = 1:N]
    else
        loglikelihoods!(LL, hmm.B, observations, n2t, lag_cat)
        [LL[n, k] += log(α[k, n2t[n]]) for k = 1:K, n = 1:N]
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    fit_mle_A_from_slice!(hmm.a, hmm.A, LL, n2t, robust = robust)

    return history ? (hmm, hist) : hmm
end

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,2} where {F<:MultivariateDistribution}, observations, n2t::AbstractArray{Int})
    N, K = size(observations, 1), size(B, 1)
    @argcheck size(LL) == (N, K)

    for n in OneTo(N)
        t = n2t[n]
        for k in OneTo(K)
            LL[n, k] = logpdf(B[k, t], observations[n, :])
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,2} where {F<:UnivariateDistribution}, observations, n2t::AbstractArray{Int})
    N, K = size(observations, 1), size(B, 1)
    @argcheck size(LL) == (N, K)

    for n in OneTo(N)
        t = n2t[n]
        for k in OneTo(K)
            LL[n, k] = logpdf(B[k, t], observations[n])
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, B::AbstractArray{F,4} where {F<:UnivariateDistribution}, observations, n2t::AbstractArray{Int}, lag_cat::Matrix{Int})
    N, K, D = size(observations, 1), size(B, 1), size(observations, 2)
    @argcheck size(LL) == (N, K)

    for n in OneTo(N)
        t = n2t[n]
        for k in OneTo(K)
            LL[n, k] = logpdf(product_distribution(B[CartesianIndex.(k, t, 1:D, lag_cat[n, :])]), observations[n, :])
        end
    end
end

function fit_mle_A_from_slice!(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL, n2t; robust = false)

    N, K = size(LL)
    T = size(A, 3)
    c = zeros(N)
    γ = zeros(N, K)

    # get posterior of each category
    c[:] = logsumexp(LL, dims = 2)
    γ[:, :] = exp.(LL .- c)
    a[:] = γ[1, :] # initial date
    max_aposteriori = argmaxrow(γ)
    max_aposteriori_next = lead(max_aposteriori)

    Q = zeros(K, K, T)
    for n = 1:N
        t = n2t[n]
        if max_aposteriori_next[n] === missing
            continue
        else
            Q[max_aposteriori[n], max_aposteriori_next[n], t] += 1
        end
    end
    robust && (Q .+= eps())
    [A[:, :, t] = Q[:, :, t] ./ sum(Q[:, :, t], dims = 2) for t = 1:T]
end

function fit_mle_B_slice!(α::AbstractVector, B::AbstractVector{F} where {F<:Distribution}, observations;
    rand_ini = true,
    n_random_ini = 10, display_random = false,
    Dirichlet_α = 0.8, Dirichlet_categories = 0.85,
    ref_station = 1, kwargs...)
    if rand_ini == true
        α[:], B[:], h = fit_em_multiD_rand(α, B, observations; n_random_ini = n_random_ini, Dirichlet_α = Dirichlet_α, Dirichlet_categories = Dirichlet_categories, display_random = display_random, kwargs...)
    else
        h = fit_em_multiD!(α, B, observations; kwargs...)
    end
    sort_wrt_ref!(α, B, ref_station)
    h
end

function fit_mle_B_slice!(α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli},
    observations::Matrix{Int64}, lag_cat::Matrix{Int};
    rand_ini = true,
    n_random_ini = 10, display_random = false,
    Dirichlet_α = 0.8, Dirichlet_categories = 0.85,
    ref_station = 1, kwargs...)

    K, size_memory = size(B, 1), size(B, 3)
    idx_j = idx_observation_of_past_cat(lag_cat, K, size_memory)

    if rand_ini == true
        α[:], B[:], h = fit_em_multiD_rand(α, B, observations, lag_cat, idx_j; n_random_ini = n_random_ini, Dirichlet_α = Dirichlet_α, display_random = display_random, kwargs...)
    else
        h = fit_em_multiD!(α, B, observations, lag_cat, idx_j; kwargs...)
    end
    sort_wrt_ref!(α, B, ref_station)
    h
end

##

function sort_wrt_ref!(hmm::HierarchicalPeriodicHMM, ref_station)
    K, T = size(hmm.B, 1), size(hmm.B, 2)
    sorting = [[succprob(hmm.B[k, t, ref_station, 1]) for k = 1:K] for t = 1:T] # 1 is by convention the driest category i.e. Y|d....d
    new_order = sortperm.(sorting, rev = true)
    [hmm.B[:, t, :, :] = hmm.B[new_order[t], t, :, :] for t = 1:T]
    [hmm.A[:, :, t] = hmm.A[new_order[t], new_order[t], t] for t = 1:T]
end

function sort_wrt_ref!(hmm::PeriodicHMM, ref_station)
    # TODO
    K = size(α, 1)
    # sorting = [[succprob(hmm.B[k, t, ref_station, 1]) for k in 1:K] for t in 1:T] # 1 is by convention the driest category i.e. Y|d....d
    # [hmm.B[:,:,:,:] = hmm.B[sortperm(sorting[t], rev = true),t,:,:] for t in 1:T]
    # [hmm.A[:,:,t] = hmm.B[sortperm(sorting[t], rev = true), sortperm(sorting[t], rev = true),t] for t in 1:T]
end

function sort_wrt_ref!(α::AbstractVector, B::AbstractArray{F,3}, ref_station) where {F<:Bernoulli}
    K = size(α, 1)
    sorting = [succprob(B[k, ref_station, 1]) for k = 1:K] # 1 is by convention the driest category i.e. Y|d....d
    B[:, :, :] = B[sortperm(sorting, rev = true), :, :]
    α[:] = α[sortperm(sorting, rev = true)]
end

function sort_wrt_ref!(B::AbstractArray{F,3}, ref_station) where {F<:Bernoulli}
    K = size(B, 1)
    sorting = [succprob(B[k, ref_station, 1]) for k = 1:K] # 1 is by convention the driest category i.e. Y|d....d
    B[:, :, :] = B[sortperm(sorting, rev = true), :, :]
end

function sort_wrt_ref!(α::AbstractVector, B::AbstractVector{Product{Discrete,F,Vector{F}}}, ref_station) where {F<:Bernoulli}
    K = size(α, 1)
    sorting = [succprob(B[k].v[ref_station]) for k = 1:K]
    B[:] = B[sortperm(sorting, rev = true)]
    α[:] = α[sortperm(sorting, rev = true)]
end

function sort_wrt_ref!(B::AbstractVector{Product{Discrete,F,Vector{F}}}, ref_station) where {F<:Bernoulli}
    K = size(α, 1)
    sorting = [succprob(B[k].v[ref_station]) for k = 1:K]
    B[:] = B[sortperm(sorting, rev = true)]
end

function sort_wrt_ref!(α::AbstractVector, B::AbstractVector{Product{Discrete,F,Vector{F}}}, ref_station) where {F<:Categorical}
    K = size(α, 1)
    sorting = [probs(B[k].v[ref_station])[1] - 0.01 * probs(B[k].v[ref_station])[4] for k = 1:K] #add the extra term to remove ties

    B[:] = B[sortperm(sorting)]
    α[:] = α[sortperm(sorting)]
end

function sort_wrt_ref!(B::AbstractVector{Product{Discrete,F,Vector{F}}}, ref_station) where {F<:Categorical}
    K = size(α, 1)
    sorting = [probs(B[k].v[ref_station])[1] - 0.01 * probs(B[k].v[ref_station])[4] for k = 1:K] #add the extra term to remove ties
    B[:] = B[sortperm(sorting)]
end

##

function fit_em_multiD_rand(α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli},
    observations::Matrix{Int}, lag_cat::Matrix{Int}, idx_j::Vector{Vector{Vector{Int}}};
    n_random_ini = 10, Dirichlet_α = 0.8, display_random = false, kwargs...)

    D = size(observations, 2)
    K = size(α, 1)
    size_memory = size(B, 3)

    h = fit_em_multiD!(α, B, observations, lag_cat, idx_j; kwargs...)
    log_max = h.logtots[end]
    α_max, B_max = copy(α), copy(B)
    h_max = h
    (display_random == :iter) && println("random IC 1: logtot = $(h.logtots[end])")
    for i = 1:(n_random_ini-1)
        B[:, :, :] = random_product_Bernoulli(D, K, size_memory)
        α[:] = rand(Dirichlet(K, Dirichlet_α))
        h = fit_em_multiD!(α, B, observations, lag_cat, idx_j; kwargs...)
        (display_random == :iter) && println("random IC $(i+1): logtot = $(h.logtots[end])")
        if h.logtots[end] > log_max
            log_max = h.logtots[end]
            h_max = h
            α_max[:], B_max[:] = copy(α), copy(B)
        end
    end
    return α_max, B_max, h_max
end

function fit_em_multiD_rand(α::AbstractVector, B::AbstractVector{F} where {F<:Distribution},
    observations;
    n_random_ini = 10, Dirichlet_α = 0.8, Dirichlet_categories = 0.85, display_random = false, kwargs...)

    D = size(observations, 2)
    K = size(α, 1)

    h = fit_em_multiD!(α, B, observations; kwargs...)
    log_max = h.logtots[end]
    α_max, B_max = copy(α), copy(B)
    h_max = h
    (display_random == :iter) && println("random IC 1: logtot = $(h.logtots[end])")
    for i = 1:(n_random_ini-1)
        random_product!(B, Dirichlet_categories) # Randomize B
        α[:] = rand(Dirichlet(K, Dirichlet_α)) # Randomize α
        h = fit_em_multiD!(α, B, observations; kwargs...)
        (display_random == :iter) && println("random IC $(i+1): logtot = $(h.logtots[end])")
        if h.logtots[end] > log_max
            log_max = h.logtots[end]
            h_max = h
            α_max[:], B_max[:] = copy(α), copy(B)
        end
    end
    return α_max, B_max, h_max
end

function random_product!(B::AbstractVector{F}, Dirichlet_categories::Real) where {F<:Product{Discrete,U,Vector{U}}} where {U<:Categorical}
    K = size(B, 1)
    D = length(B[1].v)
    B[:] = random_product_Categorical(D, K, Dirichlet_categories)
end

function random_product!(B::AbstractVector{F}, Dirichlet_categories::Real) where {F<:Product{Discrete,U,Vector{U}}} where {U<:Bernoulli}
    K = size(B, 1)
    D = length(B[1].v)
    B[:] = random_product_Bernoulli(D, K)
end

function random_product_Categorical(D::Int, K::Int, Dirichlet_categories::Float64)
    [product_distribution([Categorical(rand(Dirichlet(K, Dirichlet_categories), D)[:, j]) for j = 1:D]) for k = 1:K]
end

function random_product_Bernoulli(D::Int, K::Int)
    [Product(Bernoulli.(rand(D))) for k = 1:K]
end

function random_product_Bernoulli(D::Int, K::Int, size_memory::Int)
    [Bernoulli(rand()) for k = 1:K, j = 1:D, m = 1:size_memory]
end

function idx_observation_of_past_cat(lag_cat, K, size_memory)
    # Matrix(T,D) of vector that give the index of data of same past.
    # ie. size_memory = 1 (no memory) -> every data is in category 1
    # ie size_memory = 2 (memory on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
    D = size(lag_cat, 2)
    idx_j = Vector{Vector{Vector{Int}}}(undef, D)
    for i in OneTo(K)
        for j = 1:D
            idx_j[j] = [findall(lag_cat[:, j] .== m) for m = 1:size_memory]
            ##
        end
    end
    return idx_j
end


function fit_em_multiD!(α::AbstractVector, B::AbstractArray{F,3} where {F<:Bernoulli},
    observations::Matrix{Int}, lag_cat::Matrix{Int}, idx_j::Vector{Vector{Vector{Int}}};
    display = :none, maxiter = 100, tol = 1e-3, robust = false)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, D, size_memory = size(observations, 1), size(B, 1), size(B, 2), size(B, 3)
    history = EMHistory(false, 0, [])

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # Initial parameters already in α, B

    # E-step
    # evaluate likelihood for each type k
    @inbounds for k in OneTo(K), n in OneTo(N)
        LL[n, k] = logpdf(product_distribution(B[CartesianIndex.(k, 1:D, lag_cat[n, :])]), observations[n, :])
    end
    [LL[:, k] .+= log(α[k]) for k = 1:K]
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    c[:] = logsumexp(LL, dims = 2)
    γ[:, :] = exp.(LL .- c)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter

        # M-step
        # with γ in hand, maximize (update) the parameters
        α[:] = mean(γ, dims = 1)
        for k in OneTo(K)
            for j = 1:D
                for m = 1:size_memory
                    if sum(γ[idx_j[j][m], k]) > 0
                        B[k, j, m] = fit_mle(Bernoulli, observations[idx_j[j][m], j], γ[idx_j[j][m], k])
                    else
                        B[k, j, m] = Bernoulli(eps())
                    end
                end
            end
        end

        # E-step
        # evaluate likelihood for each type k
        @inbounds for k in OneTo(K), n in OneTo(N)
            LL[n, k] = logpdf(product_distribution(B[CartesianIndex.(k, 1:D, lag_cat[n, :])]), observations[n, :])
        end
        [LL[:, k] .+= log(α[k]) for k = 1:K]
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
        # get posterior of each category
        c[:] = logsumexp(LL, dims = 2)
        γ[:, :] = exp.(LL .- c)

        # Loglikelihood
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

function fit_em_multiD!(α::AbstractVector, B::AbstractVector{F} where {F<:Distribution},
    observations;
    display = :none, maxiter = 200, tol = 1e-3, robust = false)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = size(observations, 1), size(B, 1)
    history = EMHistory(false, 0, [])

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # Initial parameters already in α, B

    # E-step
    # evaluate likelihood for each type k
    [LL[:, k] = logpdf(B[k], permutedims(observations)) .+ log(α[k]) for k = 1:K]
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    c[:] = logsumexp(LL, dims = 2)
    γ[:, :] = exp.(LL .- c)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    # ensure that for categorical data the correct number of categorie is taken. Else nothing
    if typeof(B[1].v[1]) == Categorical{Float64,Vector{Float64}}
        C = (ncategories(B[1].v[1]),)
    else
        C = ()
    end

    for it = 1:maxiter

        # M-step
        # with γ in hand, maximize (update) the parameters
        α[:] = mean(γ, dims = 1)
        for k in OneTo(K)
            if sum(γ[:, k]) > 0
                B[k] = fit_mle(B[k], C..., permutedims(observations), γ[:, k])
            end
        end

        # E-step
        # evaluate likelihood for each type k
        [LL[:, k] = logpdf(B[k], permutedims(observations)) .+ log(α[k]) for k = 1:K]
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
        # get posterior of each category
        c[:] = logsumexp(LL, dims = 2)
        γ[:, :] = exp.(LL .- c)

        # Loglikelihood
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
