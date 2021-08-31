function update_B_d!(B::AbstractArray{T, 4} where T, θ_Y::AbstractArray{N, 4} where N, γ::AbstractMatrix, observations, model_B::Model, mles; warm_start = true)
    @argcheck size(γ, 1) == size(observations, 1)
    @argcheck size(γ, 2) == size(B, 1)
    N = size(γ, 1)
    K = size(B, 1)
    T = size(B, 2)
    D = size(B, 3)
    size_memory = size(B, 4)
    ## For periodicHMM only the n observations corresponding to B(t) are used to update B(t)
    ## Update the smoothing parameters in the JuMP model
    πk = model_B[:πk]
    for n in 1:N, k in 1:K
        set_value(πk[n,k], γ[n,k]) 
    end    
    
    # TODO In place version (the SharedArray thing does not really work with Slurm on a cluster because threads do not necessarly have shared memory)
    # pmap(tup -> fit_mle_one_B_distributed!(@view(θ_Y[tup...,:]), model_B, mles[tup...]), Iterators.product(1:K, 1:D, 1:size_memory))

    @showprogress pmap(Iterators.product(1:K, 1:D, 1:size_memory)) do tup
        sleep(0.5)
        θ_Y[tup..., :] = fit_mle_one_B_distributed(θ_Y[tup...,:], model_B, mles[tup...], warm_start = warm_start)
    end
    # map(Iterators.product(1:K, 1:D, 1:size_memory)) do tup
    #     sleep(0.5)
    #     @show tup
    #     # if tup == (6,6,1)
    #     #     unset_silent(model_B)
    #     # else
    #     #     set_silent(model_B)
    #     # end
    #     θ_Y[tup..., :] = fit_mle_one_B_distributed(θ_Y[tup...,:], model_B, mles[tup...], warm_start = warm_start)
    # end
    # θ_res = pmap(tup -> fit_mle_one_B_distributed(θ_Y[tup...,:], model_B, mles[tup...]), Iterators.product(1:K, 1:D, 1:size_memory))
    
    # for (k, s, h) in Iterators.product(1:K, 1:D, 1:size_memory)
    #      θ_Y[k,s,h,:] = θ_res[k,s,h] 
    # end

    p = [1/(1+exp(polynomial_trigo(t, θ_Y[k,s,h,:], T = T))) for k in 1:K, t in 1:T, s in 1:D, h in 1:size_memory]
    B[:,:,:,:] = Bernoulli.(p)
end

function fit_mle_one_B_distributed!(θ_Y, model_B, mle; warm_start = true)
    θ_jump = model_B[:θ_jump]
    warm_start && set_start_value.(θ_jump, θ_Y[:])
    # set_silent(model_B)
    set_time_limit_sec(model_B, 60.0)
    set_optimizer_attribute(model_B, "max_iter", 60)
    @NLobjective(
    model_B, Max,
    mle
    )
    optimize!(model_B)
    θ_Y[:] = value.(θ_jump)
end

function fit_mle_one_B_distributed(θ_Y, model_B, mle; warm_start = true)
    θ_jump = model_B[:θ_jump]
    warm_start && set_start_value.(θ_jump, θ_Y[:])
    # set_silent(model_B)
    @NLobjective(
    model_B, Max,
    mle
    )
    optimize!(model_B)
    value.(θ_jump)
end

function fit_mle_one_A_distributed(θ_Q, model, ξ; warm_start = true)
    pklj_jump = model[:pklj_jump]
    πk = model[:πk]
    πkl = model[:πkl]
    N, K = size(ξ)
    ## Update the smoothing parameters in the JuMP model
    for n in 1:N-1
        set_value(πk[n], sum(ξ[n, l] for l in 1:K))
        for l in 1:K-1
            set_value(πkl[n,l], ξ[n, l])
        end
    end
    warm_start && set_start_value.(pklj_jump, θ_Q[:,:])
    # Optimize the updated model
    optimize!(model)
    # Obtained the new parameters
    return value.(pklj_jump)
end

function update_A_d!(
    A::AbstractArray{T,3} where {T},
    θ_Q::AbstractArray{T,3} where T,
    ξ::AbstractArray,
    α::AbstractMatrix,
    β::AbstractMatrix,
    LL::AbstractMatrix,
    n2t::AbstractArray{Int},
    model_A::Model;
    warm_start = true
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
    T = size(A,3)
    @inbounds for n in OneTo(N - 1)
        t = n2t[n] # periodic t
        m = SHHMM.vec_maximum(view(LL, n + 1, :)) #TODO Should be imported from HMMBase maxmouchet 
        c = 0.0

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] = α[n, i] * A[i, j, t] * exp(LL[n+1, j] - m) * β[n+1, j]
            c += ξ[n, i, j]
        end

        for i in OneTo(K), j in OneTo(K)
            ξ[n, i, j] /= c
        end
    end
    # ξ are the filtering probablies

    @showprogress pmap(1:K) do k
        sleep(0.5)
        θ_Q[k,:,:] = fit_mle_one_A_distributed(θ_Q[k, :, :], model_A, ξ[:, k, :]; warm_start = warm_start)
    end

    [A[k,l,t] = exp(polynomial_trigo(t, θ_Q[k,l,:], T = T)) for k in 1:K, l in 1:K-1, t in 1:T]
    [A[k,K,t] = 1 for k in 1:K, t in 1:T] # last colum is 1/normalization (one could do otherwise)
    normalization_polynomial = [1+sum(A[k,l,t] for l in 1:K-1) for k in 1:K, t in 1:T]
    [A[k,l,t] /= normalization_polynomial[k,t]  for k in 1:K, l in 1:K, t in 1:T]
end

function fit_mle_d(hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int},
    θ_Q::AbstractArray{TQ, 3} where TQ,
    θ_Y::AbstractArray{TY, 4} where TY; all_iters = false, kwargs...)
    hmm = copy(hmm)
    θ_Q = copy(θ_Q)
    θ_Y = copy(θ_Y)
    history, all_θ_Q, all_θ_Y = fit_mle_d!(hmm, observations, n2t, θ_Q, θ_Y; kwargs...)
    if all_iters == true
        return hmm, θ_Q, θ_Y, history, all_θ_Q, all_θ_Y
    else
        return hmm, θ_Q, θ_Y, history
    end
end

function fit_mle_d!(
    hmm::HierarchicalPeriodicHMM,
    observations::AbstractArray,
    n2t::AbstractArray{Int},
    θ_Q::AbstractArray{TQ, 3} where TQ,
    θ_Y::AbstractArray{TY, 4} where TY
    ;
    display = :none,
    maxiter = 100,
    tol = 1e-3,
    robust = false,
    silence = true,
    warm_start = true
)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, T, size_memory, D = size(observations, 1), size(hmm, 1), size(hmm, 3), size(hmm,4), size(hmm,2)
    
    #SharedArray
    # TODO In place version (the SharedArray thing does not really work with Slurm on a cluster because threads do not necessarly have shared memory)
    # θ_Y = convert(SharedArray, θ_Y)

    deg_Q = (size(θ_Q, 3)-1)÷2
    deg_Y = (size(θ_Y, 4)-1)÷2

    @argcheck T == size(hmm.B, 2)
    history = EMHistory(false, 0, [])
    
    all_θ_Q = [copy(θ_Q)]
    all_θ_Y = [copy(θ_Y)]
    # Allocate memory for in-place updates
    c = zeros(N)
    α = zeros(N, K)
    β = zeros(N, K)
    γ = zeros(N, K)
    ξ = zeros(N, K, K)
    LL = zeros(N, K)

    # assign category for observation depending in the past observations
    memory = Int(log(size_memory)/log(2))
    lag_cat = conditional_to(observations, memory)
    n_occurence_history = [findall(.&(observations[:,j] .== y, lag_cat[:,j] .== h)) for j in 1:D, h in 1:size_memory, y in 0:1]

    model_A = model_for_A(ξ[:,1,:], n2t, deg_Q, T, silence = silence) # JuMP Model for transition matrix
    model_B = model_for_B(γ, n2t, n_occurence_history, deg_Y, T, D, size_memory, silence = silence) # JuMP Model for transition matrix
    mles = model_B[:mle]

    loglikelihoods!(LL, hmm, observations, n2t, lag_cat)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A_d!(hmm.A, θ_Q, ξ, α, β, LL, n2t, model_A; warm_start = warm_start)
        update_B_d!(hmm.B, θ_Y, γ, observations, model_B, mles; warm_start = warm_start)
        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely observations.
        robust && (hmm.A .+= eps())

        @check isprobvec(hmm.a)
        @check istransmats(hmm.A)

        push!(all_θ_Q, copy(θ_Q))
        push!(all_θ_Y, copy(θ_Y))

        # loglikelihoods!(LL, hmm, observations, n2t)
        loglikelihoods!(LL, hmm, observations, n2t, lag_cat)

        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

        forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
        backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
        posteriors!(γ, α, β)

        logtotp = sum(c)
        (display == :iter) && println(now(), " Iteration $it: logtot = $logtotp") 
        flush(stdout)
        
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

    history, all_θ_Q, all_θ_Y
end