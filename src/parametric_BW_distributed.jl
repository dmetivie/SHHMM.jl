n_per_category(s, h, t, y, n_in_t, n_occurence_history) = (n_in_t[t] ∩ n_occurence_history[s,h,y])

function γ_s!(γ_s, γ, n_all)
  K, D, size_memory, T, rain_cat = size(γ_s)
  for tup in Iterators.product(1:D, 1:size_memory, 1:T, 1:rain_cat)
    for k in 1:K
      γ_s[k, tup...] = sum(γ[n,k] for n in n_all[tup...]; init = 0)
    end
  end
end

function s_ξ!(s_ξ, ξ, n_in_t)
    T, K = size(s_ξ, 1), size(s_ξ, 2)
    for t in 1:T
        for (k, l) in Iterators.product(1:K, 1:K)
            s_ξ[t, k, l] = sum(ξ[n, k, l] for n in n_in_t[t])
        end
    end
    # * We add ξ[N, k, l] but it should be zeros
end

function model_for_B_d(γ_s::AbstractMatrix, d::Int; silence = true)

    T, rain_cat = size(γ_s)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_cpu_time", 60.0)
    set_optimizer_attribute(model, "max_iter", 100)

    silence && set_silent(model)
    f = 2π/T
    cos_nj = [cos(f*j*t) for t in 1:T, j in 1:d]
    sin_nj = [sin(f*j*t) for t in 1:T, j in 1:d]
    trig = [[1; interleave2(cos_nj[t,:], sin_nj[t,:])] for t in 1:T]

    @variable(model, θ_jump[j = 1:(2d+1)])
    # Polynomial P
    @NLexpression(model, Pn[t = 1:T], sum(trig[t][j] * θ_jump[j] for j in 1:length(trig[t])))

    @NLparameter(model, π_s[t = 1:T, y = 1:rain_cat] == γ_s[t, y])

    @NLexpression(model, mle,
    - sum(π_s[t,1]*log1p(exp(-Pn[t])) for t in 1:T) - sum(π_s[t,2]*log1p(exp(+Pn[t])) for t in 1:T) 
    ) # 1 is where it did not rain # 2 where it rained
    @NLobjective(
        model, Max,
        mle
    )
    # I don't know if it is the best but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:π_s] = π_s
    return model
end

function update_B_d!(B::AbstractArray{T, 4} where T, θ_Y::AbstractArray{N, 4} where N, γ::AbstractMatrix, γ_s::AbstractArray, observations, n_all, model_B::Model; warm_start = true)
    @argcheck size(γ, 1) == size(observations, 1)
    @argcheck size(γ, 2) == size(B, 1)
    N = size(γ, 1)
    K = size(B, 1)
    T = size(B, 2)
    D = size(B, 3)
    size_memory = size(B, 4)
    ## For periodicHMM only the n observations corresponding to B(t) are used to update B(t)
    ## Update the smoothing parameters in the JuMP model

    γ_s!(γ_s, γ, n_all) # update coefficient in JuMP model

    θ_res = pmap(tup -> fit_mle_one_B_d(θ_Y[tup...,:], model_B, γ_s[tup...,:,:]; warm_start = warm_start), Iterators.product(1:K, 1:D, 1:size_memory))
    
    for (k, s, h) in Iterators.product(1:K, 1:D, 1:size_memory)
         θ_Y[k,s,h,:] = θ_res[k,s,h] 
    end

    p = [1/(1+exp(polynomial_trigo(t, θ_Y[k,s,h,:], T = T))) for k in 1:K, t in 1:T, s in 1:D, h in 1:size_memory]
    B[:,:,:,:] = Bernoulli.(p)
end

function fit_mle_one_B_d(θ_Y, model_B, γ_s; warm_start = true)
    T, rain_cat = size(γ_s)
    θ_jump = model_B[:θ_jump]
    warm_start && set_start_value.(θ_jump, θ_Y[:])
    π_s = model_B[:π_s]

    for t in 1:T, y in 1:rain_cat
        set_value(π_s[t,y], γ_s[t, y])
    end
    optimize!(model_B)
    value.(θ_jump)
end

# JuMP model use to increase R(θ,θ^i) for the Q(t) matrix
function model_for_A_d(s_ξ::AbstractArray, d::Int; silence = true)
    T, K = size(s_ξ)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 200)
    silence && set_silent(model)
    f = 2π/T
    cos_nj = [cos(f*j*t) for t in 1:T, j in 1:d]
    sin_nj = [sin(f*j*t) for t in 1:T, j in 1:d]

    trig = [[1; interleave2(cos_nj[t,:], sin_nj[t,:])] for t in 1:T]

    @variable(model, pklj_jump[l = 1:(K-1), j = 1:(2d+1)], start = 0.01)
    # Polynomial P_kl
    @NLexpression(model, Pkl[t = 1:T, l = 1:K-1], sum(trig[t][j] * pklj_jump[l,j] for j in 1:length(trig[t])))

    @NLparameter(model, s_πkl[t = 1:T, l = 1:K-1] == s_ξ[t, l])
    #TODO? is it useful to define the extra parameter for the sum?
    @NLparameter(model, s_πk[t = 1:T] == sum(s_ξ[t, l] for l in 1:K))

    @NLobjective(
    model,
    Max,
    sum(sum(s_πkl[t, l]*Pkl[t,l] for l in 1:K-1) - s_πk[t]*log1p(sum(exp(Pkl[t,l]) for l in 1:K-1)) for t in 1:T)
    )
    # To add NL parameters to the model for later use https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:s_πkl] = s_πkl
    model[:s_πk] = s_πk
    return model
end

function update_A_d!(
    A::AbstractArray{T,3} where {T},
    θ_Q::AbstractArray{T,3} where T,
    ξ::AbstractArray,
    s_ξ::AbstractArray,
    α::AbstractMatrix,
    β::AbstractMatrix,
    LL::AbstractMatrix,
    n2t::AbstractArray{Int},
    n_in_t,
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
    ## 
    # ξ are the filtering probablies
    s_ξ!(s_ξ, ξ, n_in_t)
    θ_res = pmap(k -> fit_mle_one_A_d(θ_Q[k, :, :], model_A, s_ξ[:, k, :]; warm_start = warm_start), 1:K)
    
    for k in 1:K
        θ_Q[k,:,:] = θ_res[k][:,:] 
    end

    #TODO change the vectorize form to loop (or something else) to avoid allocation 
    [A[k,l,t] = exp(polynomial_trigo(t, θ_Q[k,l,:], T = T)) for k in 1:K, l in 1:K-1, t in 1:T]
    [A[k,K,t] = 1 for k in 1:K, t in 1:T] # last colum is 1/normalization (one could do otherwise)
    normalization_polynomial = [1+sum(A[k,l,t] for l in 1:K-1) for k in 1:K, t in 1:T]
    [A[k,l,t] /= normalization_polynomial[k,t]  for k in 1:K, l in 1:K, t in 1:T]
end

function fit_mle_one_A_d(θ_Q, model, s_ξ; warm_start = true)
    T, K = size(s_ξ)
    pklj_jump = model[:pklj_jump]
    s_πk = model[:s_πk]
    s_πkl = model[:s_πkl]
    ## Update the smoothing parameters in the JuMP model
    for t in 1:T
        set_value(s_πk[t], sum(s_ξ[t, l] for l in 1:K))
        for l in 1:K-1
            set_value(s_πkl[t,l], s_ξ[t, l])
        end
    end
    warm_start && set_start_value.(pklj_jump, θ_Q[:,:])
    # Optimize the updated model
    optimize!(model)
    # Obtained the new parameters
    return value.(pklj_jump)
end

function fit_mle(hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int},
    θ_Q::AbstractArray{TQ, 3} where TQ,
    θ_Y::AbstractArray{TY, 4} where TY; all_iters = false, kwargs...)
    hmm = copy(hmm)
    θ_Q = copy(θ_Q)
    θ_Y = copy(θ_Y)
    history, all_θ_Q, all_θ_Y = fit_mle!(hmm, observations, n2t, θ_Q, θ_Y; kwargs...)
    if all_iters == true
        return hmm, θ_Q, θ_Y, history, all_θ_Q, all_θ_Y
    else
        return hmm, θ_Q, θ_Y, history
    end
end

function fit_mle!(
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
    rain_cat = 2
    @argcheck T == size(hmm.B, 2)
    history = EMHistory(false, 0, [])
    
    all_θ_Q = [copy(θ_Q)]
    all_θ_Y = [copy(θ_Y)]
    # Allocate memory for in-place updates
    c = zeros(N)
    α = zeros(N, K)
    β = zeros(N, K)
    γ = zeros(N, K) # regular smoothing proba
    γ_s = zeros(K, D, size_memory, T, rain_cat) # summed smoothing proba
    ξ = zeros(N, K, K)
    s_ξ = zeros(T, K, K)
    LL = zeros(N, K)

    # assign category for observation depending in the past observations
    memory = Int(log(size_memory)/log(2))
    lag_cat = conditional_to(observations, memory)

    n_in_t = [findall(n2t .== t) for t in 1:T]
    n_occurence_history = [findall(.&(observations[:,j] .== y, lag_cat[:,j] .== h)) for j in 1:D, h in 1:size_memory, y in 0:1]
    n_all = [n_per_category(tup..., n_in_t, n_occurence_history) for tup in Iterators.product(1:D, 1:size_memory, 1:T, 1:rain_cat)]

    model_A = model_for_A_d(s_ξ[:,1,:], deg_Q, silence = silence) # JuMP Model for transition matrix
    model_B = model_for_B_d(γ_s[1, 1, 1, :, :], deg_Y, silence = silence) # JuMP Model for Emmission distribution
    
    loglikelihoods!(LL, hmm, observations, n2t, lag_cat)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    
    forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
    posteriors!(γ, α, β)
    
    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")
    
    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A_d!(hmm.A, θ_Q, ξ, s_ξ, α, β, LL, n2t, n_in_t, model_A; warm_start = warm_start)
        update_B_d!(hmm.B, θ_Y, γ, γ_s, observations, n_all, model_B; warm_start = warm_start)
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