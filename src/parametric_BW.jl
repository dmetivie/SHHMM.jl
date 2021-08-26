function update_A!(
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
        m = HMMBase.vec_maximum(view(LL, n + 1, :))
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
    pklj_jump = model_A[:pklj_jump]
    πkl = model_A[:πkl]
    πk = model_A[:πk]
    Pkl = model_A[:Pkl]

    for k in 1:K
        ## Update the smoothing parameters in the JuMP model
        [set_value(πkl[n,l], ξ[n, k, l]) for n in 1:N-1, l in 1:K-1]
        [set_value(πk[n], sum(ξ[n, k, l] for l in 1:K)) for n in 1:N-1]
        warm_start && set_start_value.(pklj_jump, θ_Q[k,:,:])
        # Optimize the updated model
        optimize!(model_A)
        # Obtained the new parameters
        θ_Q[k,:,:] = value.(pklj_jump)
    end

    [A[k,l,t] = exp(polynomial_trigo(t, θ_Q[k,l,:], T = T)) for k in 1:K, l in 1:K-1, t in 1:T]
    [A[k,K,t] = 1 for k in 1:K, t in 1:T] # last colum is 1/normalization (one could do otherwise)
    normalization_polynomial = [1+sum(A[k,l,t] for l in 1:K-1) for k in 1:K, t in 1:T]
    [A[k,l,t] /= normalization_polynomial[k,t]  for k in 1:K, l in 1:K, t in 1:T]
end

function fit_mle_B_trig!(θ_Y::AbstractVector, model::Model, γ::AbstractVector; warm_start = true)
    N = size(γ, 1)
    θ_jump = model[:θ_jump]
    πk = model[:πk]
    ## Update the smoothing parameters in the JuMP model
    [set_value(πk[n], γ[n]) for n in 1:N]
    warm_start && set_start_value.(θ_jump, θ_Y[:])
    #set_silent(model)

    optimize!(model)
    θ_Y[:] = value.(θ_jump)
end

function update_B2!(B::AbstractArray{T, 4} where T, θ_Y::AbstractArray{N, 4} where N, γ::AbstractMatrix, observations, model_B; warm_start = true)
    @argcheck size(γ, 1) == size(observations, 1)
    @argcheck size(γ, 2) == size(B, 1)
    K = size(B, 1)
    T = size(B, 2)
    D = size(B, 3)
    size_memory = size(B, 4)
    ## For periodicHMM only the n observations corresponding to B(t) are used to update B(t)
    for k in 1:K, s in 1:D, h in 1:size_memory
        if k > 1 || s > 1
            continue
        else
        println("ARG",k,s,h)
        @time fit_mle_B_trig!(@view(θ_Y[k,s,h,:]), model_B[s,h], γ[:,k]; warm_start = warm_start)
        end
    end
    p = [1/(1+exp(polynomial_trigo(t, θ_Y[k,s,h,:], T = T))) for k in 1:K, t in 1:T, s in 1:D, h in 1:size_memory]
    B[:,:,:,:] = Bernoulli.(p)
end
function update_B!(B::AbstractArray{T, 4} where T, θ_Y::AbstractArray{N, 4} where N, γ::AbstractMatrix, observations, model_B::Model; warm_start = true)
    @argcheck size(γ, 1) == size(observations, 1)
    @argcheck size(γ, 2) == size(B, 1)
    N = size(γ, 1)
    K = size(B, 1)
    T = size(B, 2)
    D = size(B, 3)
    size_memory = size(B, 4)
    ## For periodicHMM only the n observations corresponding to B(t) are used to update B(t)
    θ_jump = model_B[:θ_jump]
    πk = model_B[:πk]
    Pn = model_B[:Pn]
    mle = model_B[:mle]
    ## Update the smoothing parameters in the JuMP model
    [set_value(πk[n,k], γ[n,k]) for n in 1:N, k in 1:K]
    for k in 1:K, s in 1:D, h in 1:size_memory
        if k > 1 || s > 1
            continue
        else
            println("ARG",k,s,h)
            @time begin 
            warm_start && set_start_value.(θ_jump, θ_Y[k,s,h,:])
            # set_silent(model_B)
            @NLobjective(
                model_B, Max,
                mle[k,s,h]
                )
                optimize!(model_B)
                θ_Y[k,s,h,:] = value.(θ_jump)
            end
        end
        end
    p = [1/(1+exp(polynomial_trigo(t, θ_Y[k,s,h,:], T = T))) for k in 1:K, t in 1:T, s in 1:D, h in 1:size_memory]
    B[:,:,:,:] = Bernoulli.(p)
end

# JuMP model use to increase R(θ,θ^i) for the Q(t) matrix
function model_for_A(ξ::AbstractArray, n2t::AbstractArray{Int}, d::Int, T::Int; silence = true)
    K = size(ξ, 2)
    N = length(n2t)
    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)
    f = 2π/T
    cos_nj = [cos(f*j*n2t[n]) for n in 1:(N-1), j in 1:d]
    sin_nj = [sin(f*j*n2t[n]) for n in 1:(N-1), j in 1:d]

    trig_nj = [[1; interleave2(cos_nj[n,:], sin_nj[n,:])] for n in 1:N-1]

    @variable(model, pklj_jump[l = 1:(K-1), j = 1:(2d+1)], start = 0.01)
    # Polynomial P_kl

    @NLexpression(model, Pkl[n = 1:N-1, l = 1:K-1], sum(trig_nj[n][j] * pklj_jump[l,j] for j in 1:length(trig_nj[n])))

    @NLparameter(model, πkl[n = 1:N-1, l = 1:K-1] == ξ[n, l])
    @NLparameter(model, πk[n = 1:N-1] == sum(ξ[n, l] for l in 1:K))

    @NLobjective(
    model,
    Max,
    sum(sum(πkl[n, l]*Pkl[n,l] for l in 1:K-1) - πk[n]*log1p(sum(exp(Pkl[n,l]) for l in 1:K-1)) for n in 1:N-1)
    )
    # To add NL parameters to the model for later use https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:πkl] = πkl
    model[:πk] = πk
    return model
end

function model_for_B3(γ::AbstractVector, n2t, n_occurence_history, d::Int, T::Int, D::Int, size_memory::Int; silence = true)
    N = length(n2t)
    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)
    f = 2π/T
    cos_nj = [cos(f*j*n2t[n]) for n in 1:N, j in 1:d]
    sin_nj = [sin(f*j*n2t[n]) for n in 1:N, j in 1:d]

    trig = [[1; interleave2(cos_nj[n,:], sin_nj[n,:])] for n in 1:N]

    @variable(model, θ_jump[j = 1:(2d+1)])
    # Polynomial P
    @NLexpression(model, Pn[n = 1:N], sum(trig[n][j] * θ_jump[j] for j in 1:length(trig[n])))

    @NLparameter(model, πk[n = 1:N] == γ[n])

    @NLobjective(model, Max,
    - sum(πk[n]*log1p(exp(-Pn[n])) for n in n_occurence_history[1]) - sum(πk[n]*log1p(exp(Pn[n])) for n in n_occurence_history[2])
    ) # 1 is where it did not rain # 2 where it rained

    # I don't know if it is the best to add NL parameters but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:πk] = πk
    return model
end
function model_for_B2(γ::AbstractVector, n2t, n_occurence_history, d::Int, T::Int, D::Int, size_memory::Int; silence = true)
    N = length(n2t)
    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)
    f = 2π/T
    cos_nj = [cos(f*j*n2t[n]) for n in 1:N, j in 1:d]
    sin_nj = [sin(f*j*n2t[n]) for n in 1:N, j in 1:d]

    trig = [[1; interleave2(cos_nj[n,:], sin_nj[n,:])] for n in 1:N]

    @variable(model, θ_jump[j = 1:(2d+1)])
    # Polynomial P
    @NLexpression(model, Pn[n = 1:N], sum(trig[n][j] * θ_jump[j] for j in 1:length(trig[n])))

    @NLparameter(model, πk[n = 1:N] == γ[n])

    @NLexpression(model, mle,
    - sum(πk[n]*log1p(exp(-Pn[n])) for n in n_occurence_history[1]) - sum(πk[n]*log1p(exp(Pn[n])) for n in n_occurence_history[2])
    ) # 1 is where it did not rain # 2 where it rained
    @NLobjective(model, Max,
        mle
    )
    # I don't know if it is the best to add NL parameters but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:πk] = πk
    return model
end
function model_for_B(γ::AbstractMatrix, n2t, n_occurence_history, d::Int, T::Int, D::Int, size_memory::Int; silence = true)
    K = size(γ, 2)
    N = length(n2t)
    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)
    f = 2π/T
    cos_nj = [cos(f*j*n2t[n]) for n in 1:N, j in 1:d]
    sin_nj = [sin(f*j*n2t[n]) for n in 1:N, j in 1:d]

    trig = [[1; interleave2(cos_nj[n,:], sin_nj[n,:])] for n in 1:N]

    @variable(model, θ_jump[j = 1:(2d+1)])
    # Polynomial P
    @NLexpression(model, Pn[n = 1:N], sum(trig[n][j] * θ_jump[j] for j in 1:length(trig[n])))

    @NLparameter(model, πk[n = 1:N, k = 1:K] == γ[n, k])

    @NLexpression(model, mle[k = 1:K, s = 1:D, h = 1:size_memory],
    - sum(πk[n,k]*log1p(exp(-Pn[n])) for n in n_occurence_history[s,h,1]) - sum(πk[n,k]*log1p(exp(Pn[n])) for n in  n_occurence_history[s,h,2])
    ) # 1 is where it did not rain # 2 where it rained

    # I don't know if it is the best but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:πk] = πk
    model[:mle] = mle
    return model
end


function fit_mle2(hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int},
    θ_Q::AbstractArray{TQ, 3} where TQ,
    θ_Y::AbstractArray{TY, 4} where TY; all_iters = false, kwargs...)
    hmm = copy(hmm)
    θ_Q = copy(θ_Q)
    θ_Y = copy(θ_Y)

    history, all_θ_Q, all_θ_Y = fit_mle2!(hmm, observations, n2t, θ_Q, θ_Y; kwargs...)
    if all_iters == true
        return hmm, θ_Q, θ_Y, history, all_θ_Q, all_θ_Y
    else
        return hmm, θ_Q, θ_Y, history
    end
end


function fit_mle3(hmm::HierarchicalPeriodicHMM, observations, n2t::AbstractArray{Int},
    θ_Q::AbstractArray{TQ, 3} where TQ,
    θ_Y::AbstractArray{TY, 4} where TY; all_iters = false, kwargs...)
    hmm = copy(hmm)
    θ_Q = copy(θ_Q)
    θ_Y = copy(θ_Y)

    history, all_θ_Q, all_θ_Y = fit_mle3!(hmm, observations, n2t, θ_Q, θ_Y; kwargs...)
    if all_iters == true
        return hmm, θ_Q, θ_Y, history, all_θ_Q, all_θ_Y
    else
        return hmm, θ_Q, θ_Y, history
    end
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

function fit_mle2!(
    hmm::HierarchicalPeriodicHMM,
    observations,
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
    model_B = [model_for_B2(γ[:,1], n2t, n_occurence_history[j,h,:], deg_Y, T, D, size_memory, silence = silence) for j in 1:D, h in 1:size_memory] # JuMP Model for transition matrix

    loglikelihoods!(LL, hmm, observations, n2t, lag_cat)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, θ_Q, ξ, α, β, LL, n2t, model_A; warm_start = warm_start)
        update_B2!(hmm.B, θ_Y, γ, observations, model_B; warm_start = warm_start)
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

    history, all_θ_Q, all_θ_Y
end


function fit_mle3!(
    hmm::HierarchicalPeriodicHMM,
    observations,
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
    model_B = [model_for_B3(γ[:,1], n2t, n_occurence_history[j,h,:], deg_Y, T, D, size_memory, silence = silence) for j in 1:D, h in 1:size_memory] # JuMP Model for transition matrix

    loglikelihoods!(LL, hmm, observations, n2t, lag_cat)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, θ_Q, ξ, α, β, LL, n2t, model_A; warm_start = warm_start)
        update_B2!(hmm.B, θ_Y, γ, observations, model_B; warm_start = warm_start)
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

    history, all_θ_Q, all_θ_Y
end
function fit_mle!(
    hmm::HierarchicalPeriodicHMM,
    observations,
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

    loglikelihoods!(LL, hmm, observations, n2t, lag_cat)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
    backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, θ_Q, ξ, α, β, LL, n2t, model_A; warm_start = warm_start)
        update_B!(hmm.B, θ_Y, γ, observations, model_B; warm_start = warm_start)
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

    history, all_θ_Q, all_θ_Y
end
