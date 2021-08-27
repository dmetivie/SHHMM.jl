
# function update_B!(B::AbstractArray{T, 4} where T, θ_Y::AbstractArray{N, 4} where N, γ::AbstractMatrix, observations, model_B; warm_start = true)
#     @argcheck size(γ, 1) == size(observations, 1)
#     @argcheck size(γ, 2) == size(B, 1)
#     K = size(B, 1)
#     T = size(B, 2)
#     D = size(B, 3)
#     size_memory = size(B, 4)
#     ## For periodicHMM only the n observations corresponding to B(t) are used to update B(t)

#     θ_res = pmap(tup -> fit_mle_B_trig(θ_Y[tup...,:], model_B[tup[2], tup[3]], γ[:, tup[3]]; warm_start = warm_start), Iterators.product(1:K, 1:D, 1:size_memory))

#     for (k, s, h) in Iterators.product(1:K, 1:D, 1:size_memory)
#          θ_Y[k,s,h,:] = θ_res[k,s,h] 
#     end
#     p = [1/(1+exp(polynomial_trigo(t, θ_Y[k,s,h,:], T = T))) for k in 1:K, t in 1:T, s in 1:D, h in 1:size_memory]
#     B[:,:,:,:] = Bernoulli.(p)
# end

function update_B!(B::AbstractArray{T, 4} where T, θ_Y::AbstractArray{N, 4} where N, γ::AbstractMatrix, observations, model_B::Model, mles; warm_start = true)
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

    pmap(tup -> fit_mle_B_trig_distributed!(view(θ_Y[tup,:]), model_B, mles[tup]), Iterators.product(1:K, 1:D, 1:size_memory))

    # for (k, s, h) in Iterators.product(1:K, 1:D, 1:size_memory)
    #      θ_Y[k,s,h,:] = θ_res[k,s,h] 
    # end
    p = [1/(1+exp(polynomial_trigo(t, θ_Y[k,s,h,:], T = T))) for k in 1:K, t in 1:T, s in 1:D, h in 1:size_memory]
    B[:,:,:,:] = Bernoulli.(p)
end

@everywhere function fit_mle_B_trig_distributed!(θ_Y, model_B, mle; warm_start = true)
    θ_jump = model_B[:θ_jump]
    warm_start && set_start_value.(θ_jump, θ_Y[:])
    # set_silent(model_B)
    @NLobjective(
    model_B, Max,
    mle
    )
    optimize!(model_B)
end

# @everywhere function fit_mle_B_trig!(θ_Y::AbstractVector, model::Model, γ::AbstractVector)
#     N = size(γ, 1)
#     θ_jump = model[:θ_jump]
#     πk = model[:πk]
#     ## Update the smoothing parameters in the JuMP model
#     for n in 1:N
#         set_value(πk[n], γ[n])
#     end
#     set_start_value.(θ_jump, θ_Y[:])
#     #set_silent(model)

#     optimize!(model)
#     θ_Y[:] = value.(θ_jump)
#     value.(θ_jump)
# end


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
    
    #SharedArray
    # model_B = convert(SharedArray, model_B)
    # mles = convert(SharedArray, mles)

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
        update_B!(hmm.B, θ_Y, γ, observations, model_B, mles; warm_start = warm_start)
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

# function fit_mle!(
#     hmm::HierarchicalPeriodicHMM,
#     observations,
#     n2t::AbstractArray{Int},
#     θ_Q::AbstractArray{TQ, 3} where TQ,
#     θ_Y::AbstractArray{TY, 4} where TY
#     ;
#     display = :none,
#     maxiter = 100,
#     tol = 1e-3,
#     robust = false,
#     silence = true,
#     warm_start = true
# )
#     @argcheck display in [:none, :iter, :final]
#     @argcheck maxiter >= 0

#     N, K, T, size_memory, D = size(observations, 1), size(hmm, 1), size(hmm, 3), size(hmm,4), size(hmm,2)

#     #SharedArray
#     θ_Y = convert(SharedArray, θ_Y)


#     deg_Q = (size(θ_Q, 3)-1)÷2
#     deg_Y = (size(θ_Y, 4)-1)÷2

#     @argcheck T == size(hmm.B, 2)
#     history = EMHistory(false, 0, [])
    
#     all_θ_Q = [copy(θ_Q)]
#     all_θ_Y = [copy(θ_Y)]
#     # Allocate memory for in-place updates
#     c = zeros(N)
#     α = zeros(N, K)
#     β = zeros(N, K)
#     γ = zeros(N, K)
#     ξ = zeros(N, K, K)
#     LL = zeros(N, K)

#     #SharedArray
#     γ = convert(SharedArray, γ)

#     # assign category for observation depending in the past observations
#     memory = Int(log(size_memory)/log(2))
#     lag_cat = conditional_to(observations, memory)
#     n_occurence_history = [findall(.&(observations[:,j] .== y, lag_cat[:,j] .== h)) for j in 1:D, h in 1:size_memory, y in 0:1]

#     model_A = model_for_A(ξ[:,1,:], n2t, deg_Q, T, silence = silence) # JuMP Model for transition matrix
#     model_B = [model_for_B2(γ[:,1], n2t, n_occurence_history[j,h,:], deg_Y, T, D, size_memory, silence = silence) for j in 1:D, h in 1:size_memory] # JuMP Model for transition matrix

#     #SharedArray
#     model_B = convert(SharedArray, model_B)

#     loglikelihoods!(LL, hmm, observations, n2t, lag_cat)
#     robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

#     forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
#     backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
#     posteriors!(γ, α, β)

#     logtot = sum(c)
#     (display == :iter) && println("Iteration 0: logtot = $logtot")

#     for it = 1:maxiter
#         update_a!(hmm.a, α, β)
#         update_A!(hmm.A, θ_Q, ξ, α, β, LL, n2t, model_A; warm_start = warm_start)
#         update_B2!(hmm.B, θ_Y, γ, observations, model_B; warm_start = warm_start)
#         # Ensure the "connected-ness" of the states,
#         # this prevents case where there is no transitions
#         # between two extremely likely observations.
#         robust && (hmm.A .+= eps())

#         @check isprobvec(hmm.a)
#         @check istransmats(hmm.A)

#         push!(all_θ_Q, copy(θ_Q))
#         push!(all_θ_Y, copy(θ_Y))

#         # loglikelihoods!(LL, hmm, observations, n2t)
#         loglikelihoods!(LL, hmm, observations, n2t, lag_cat)

#         robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

#         forwardlog!(α, c, hmm.a, hmm.A, LL, n2t)
#         backwardlog!(β, c, hmm.a, hmm.A, LL, n2t)
#         posteriors!(γ, α, β)

#         logtotp = sum(c)
#         (display == :iter) && println("Iteration $it: logtot = $logtotp")

#         push!(history.logtots, logtotp)
#         history.iterations += 1

#         if abs(logtotp - logtot) < tol
#             (display in [:iter, :final]) &&
#                 println("EM converged in $it iterations, logtot = $logtotp")
#             history.converged = true
#             break
#         end

#         logtot = logtotp
#     end

#     if !history.converged
#         if display in [:iter, :final]
#             println("EM has not converged after $(history.iterations) iterations, logtot = $logtot")
#         end
#     end

#     history, all_θ_Q, all_θ_Y
# end