
function MixtureModel(hmm::HMM)
    sdists = hmm.A[1,:]
    MixtureModel([hmm.B...], sdists)
end

function HMM(m::MixtureModel)
    K = ncomponents(m)
    a = probs(m)
    A = repeat(permutedims(m.prior.p), K, 1)
    B = m.components
    HMM(a, A, B)
end
#
# function PeriodicHMM(vec_mix::Vector{MixtureModel})
#     T = length(vec_mix)
#     K = ncomponents(vec_mix[1])
#     a = probs(vec_mix[1])
#     A = repeat.(permutedims.(probs.(vec_mix)), K, 1)
#     B = components.(vec_mix)
#     PeriodicHMM(a, [A[t][k,l] for k in 1:K, l in 1:K, t in 1:T], [B[t][k] for k in 1:K, t in 1:T])
# end

function copy(mix::MixtureModel)
	MixtureModel(components(mix), probs(mix))
end

function fit_mle(mix::MixtureModel, observations; init = :none, kwargs...)
    mix = copy(mix)

    if init == :kmeans
        kmeans_init!(mix, observations, display = get(kwargs, :display, :none))
    end
	fit_em(mix, observations; kwargs...)
end

function update_a!(
    a::AbstractVector,
	A::AbstractMatrix
)
	a[:] = A[1,:]
end

function update_A!(
    A::AbstractMatrix,
    γ::AbstractMatrix
)
	K = size(A,1)
	A[:,:] = repeat(mean(γ, dims=1), K, 1)
end

function posteriors!(
    γ::AbstractMatrix,
	c::AbstractVector,
	A::AbstractMatrix,
	LL::AbstractMatrix
)
	K = size(A,1)
	a = similar(LL)
	α = log.(A[1,:])
	[a[:,k] = LL[:,k] .+ α[k] for k in 1:K]
	c[:] = logsumexp(a, dims = 2)
	γ[:,:] =  exp.(a .- c)
end

function fit_em(mix::MixtureModel, y; display = :none, maxiter = 100, tol = 1e-3, robust = false)

	@argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

	N, K = length(y), ncomponents(mix)
	history = EMHistory(false, 0, [])

	# Allocate memory for in-place updates

    LL = zeros(N,K)
    γ = similar(LL)
	c = zeros(N)
	types = typeof.(components(mix))

	# Initial parameters
	α = copy(probs(mix))
	dists = copy(components(mix))

	# E-step
	# evaluate likelihood for each type k
	[LL[:,k] = log(α[k]).+logpdf.(dists[k], y) for k in 1:K]
	robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
	# get posterior of each category
	c[:] = logsumexp(LL, dims = 2)
	γ[:,:] =  exp.(LL .- c)

	# Loglikelihood
	logtot = sum(c)
	(display == :iter) && println("Iteration 0: logtot = $logtot")

	for it = 1:maxiter

		# M-step
		# with γ in hand, maximize (update) the parameters
		α[:] = mean(γ, dims=1)
		dists[:] = [fit_mle(types[k], y, γ[:,k]) for k in 1:K]

		# E-step
		# evaluate likelihood for each type k
		[LL[:,k] = log(α[k]).+logpdf.(dists[k], y) for k in 1:K]
		robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
		# get posterior of each category
		c[:] = logsumexp(LL, dims = 2)
		γ[:,:] =  exp.(LL .- c)

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

    return MixtureModel(dists, α), history
end


# function fit_em_GE(mix::MixtureModel, y, M1; display = :none, maxiter = 100, tol = 1e-3, robust = false)
#
# 	@argcheck display in [:none, :iter, :final]
#     @argcheck maxiter >= 0
#
# 	N, K = length(y), ncomponents(mix)
# 	history = EMHistory(false, 0, [])
#
# 	# Allocate memory for in-place updates
#
#     LL = zeros(N,K)
#     γ = similar(LL)
# 	c = zeros(N)
# 	types = typeof.(components(mix))
# 	α = copy(probs(mix))
# 	dists = copy(components(mix))
#
# 	# evaluate likelihood for each type k
# 	[LL[:,k] = log(α[k]).+logpdf.(dists[k], y) for k in 1:K]
# 	robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
#
# 	# get posterior of each type
# 	c[:] = logsumexp(LL, dims = 2)
# 	γ[:,:] =  exp.(LL .- c)
#
# 	logtot = sum(c)
# 	(display == :iter) && println("Iteration 0: logtot = $logtot")
#
# 	for it = 1:maxiter
#
# 		# with p in hand, update
# 		α[:] = mean(γ, dims=1)
# 		θ = [mean(y, Weights(γ[:,k])) for k in 1:K]
# 		alpha = (M1 - θ[2]α[2])/(α[1]θ[1])
# 		dists[1] = Gamma(alpha, θ[1])
# 		dists[2] = Exponential(θ[2])
#
#         # evaluate likelihood for each type k
# 		[LL[:,k] = log(α[k]) .+ logpdf.(dists[k], y) for k in 1:K]
# 		robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
#
# 		# get posterior of each type
#         c[:] = logsumexp(LL, dims = 2)
#         γ[:,:] =  exp.(LL .- c)
#
#         # Log likelihood
# 		logtotp = sum(c)
# 		(display == :iter) && println("Iteration $it: logtot = $logtotp")
#
# 		push!(history.logtots, logtotp)
#         history.iterations += 1
#
# 		if abs(logtotp - logtot) < tol
#             (display in [:iter, :final]) &&
#                 println("EM converged in $it iterations, logtot = $logtotp")
#             history.converged = true
#             break
#         end
#
#         logtot = logtotp
#     end
#
#     return MixtureModel(dists, α), history
# end

# same as fit_em but a tad slower (howeve)
function fit_mle_mixture!(
    hmm::HMM,
    observations,
    ;
    display = :none,
    maxiter = 100,
    tol = 1e-3,
    robust = false,
    estimator = fit_mle,
)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = size(observations, 1), size(hmm, 1)
    history = EMHistory(false, 0, [])

    # Allocate memory for in-place updates

    c = zeros(N)
    γ = zeros(N, K)
    LL = zeros(N, K)

    loglikelihoods!(LL, hmm, observations)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    posteriors!(γ, c, hmm.A, LL)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter

		update_A!(hmm.A, γ)
        update_B!(hmm.B, γ, observations, estimator)

		# Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely observations.
        robust && (hmm.A .+= eps())

		@check istransmat(hmm.A)

        loglikelihoods!(LL, hmm, observations)
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

        posteriors!(γ, c, hmm.A, LL)

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
	update_a!(hmm.a, hmm.A)

    if !history.converged
        if display in [:iter, :final]
            println("EM has not converged after $(history.iterations) iterations, logtot = $logtot")
        end
    end
    history
end

## * Parametric model: Trigonometric function

function model_for_dist(dist::Exponential, θ_Y::AbstractArray, γ::AbstractVector, y, n2t, T::Int; silence = true)
	d = (size(θ_Y, 1)-1)÷2
	@argcheck length(y) == length(n2t)
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
    @NLexpression(model, Pθ[n = 1:N], sum(trig[n][j] * θ_jump[j] for j in 1:(2d+1)))

    @NLparameter(model, πk[n = 1:N] == γ[n])
	@NLparameter(model, R[n = 1:N] == y[n])

	@NLobjective(
	model, Max,
	-sum(πk[n]*(Pθ[n]+R[n]*exp(-Pθ[n])) for n in 1:N)
	)

    # I don't know if it is the best but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:πk] = πk
	model[:R] = R

    return model
end

function model_for_dist(dist::Gamma, θ_Y::AbstractArray, γ::AbstractVector, y, n2t, T::Int; silence = true)
	d = (size(θ_Y, 2)-1)÷2
    K = size(γ, 2)
    N = length(n2t)
    model = Model(Ipopt.Optimizer)
	set_optimizer_attributes(model, "max_iter" => 20)
    silence && set_silent(model)
    f = 2π/T
    cos_nj = [cos(f*j*n2t[n]) for n in 1:N, j in 1:d]
    sin_nj = [sin(f*j*n2t[n]) for n in 1:N, j in 1:d]

    trig = [[1; interleave2(cos_nj[n,:], sin_nj[n,:])] for n in 1:N]

    @variable(model, θ_jump[j = 1:(2d+1)])
	@variable(model, a_jump[j = 1:(2d+1)])
    # set_lower_bound(θ_jump[1], 0)
    # set_upper_bound.(θ_jump, 6)    
    # set_lower_bound(a_jump[1], 0)
    # set_upper_bound(a_jump[1], 3)    

    # Polynomial P
    @NLexpression(model, Pθ[n = 1:N], sum(trig[n][j] * θ_jump[j] for j in 1:length(trig[n])))
	@NLexpression(model, Pa[n = 1:N], sum(trig[n][j] * a_jump[j] for j in 1:length(trig[n])))

    @NLparameter(model, πk[n = 1:N] == γ[n])
	@NLparameter(model, R[n = 1:N] == y[n])
	register(model, :loggamma, 1, loggamma; autodiff = true) # cf the JuMP warning that prevent loggamma with more than one argument
	@NLobjective(
	model, Max,
	# -sum(πk[n]*(Pθ[n]+R[n]*exp(-Pθ[n])) for n in 1:N)
	# -sum(πk[n]*(exp(Pa[n])*Pθ[n]+R[n]*exp(-Pθ[n])+log(gamma(exp(Pa[n])))-expm1(Pa[n])*log(R[n])) for n in 1:N)
	-sum(πk[n]*(exp(Pa[n])*Pθ[n]+R[n]*exp(-Pθ[n])+loggamma(exp(Pa[n]))-expm1(Pa[n])*log(R[n])) for n in 1:N)
	)
	# @NLexpression(model, mle,
	# sum(πk[n]*(-exp(Pa[n])*Pθ[n]-R[n]*exp(-Pθ[n])-loggamma(exp(Pa[n]))+expm1(Pa[n])*log(R[n])) for n in N)
	# )
    # I don't know if it is the best but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:πk] = πk
	model[:R] = R
    return model
end

function model_for_α(θ_α::AbstractMatrix, γ::AbstractMatrix, n2t, T::Int; silence = true)
    N, K = size(γ)
	d = (size(θ_α, 2)-1)÷2
    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)
    f = 2π/T
    cos_nj = [cos(f*j*n2t[n]) for n in 1:N, j in 1:d]
    sin_nj = [sin(f*j*n2t[n]) for n in 1:N, j in 1:d]

    trig = [[1; interleave2(cos_nj[n,:], sin_nj[n,:])] for n in 1:N]
    @variable(model, θ_jump[k = 1:K-1, j = 1:(2d+1)])
    # Polynomial P
    @NLexpression(model, Pn[n = 1:N, k = 1:K-1], sum(trig[n][j] * θ_jump[k,j] for j in 1:length(trig[n])))

    @NLparameter(model, πk[n = 1:N, k = 1:K] == γ[n, k])
	@NLobjective(
    model,
    Max,
    sum(sum(πk[n,k]*Pn[n,k] for k in 1:K-1) - log1p(sum(exp(Pn[n,k]) for k in 1:K-1)) for n in 1:N) #Σ_k πk[n,k] = 1
    )
    # I don't know if it is the best but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:πk] = πk
    return model
end

function update_α!(
    α::AbstractArray where {T},
    θ::AbstractArray where T,
    γ::AbstractMatrix,
    model::Model;
    warm_start = true
)
    N, K = size(γ)
    T = size(α,2)

    # ξ are the filtering probablies
    θ_jump = model[:θ_jump]
    πk = model[:πk]
    ## Update the smoothing parameters in the JuMP model
    [set_value(πk[n,k], γ[n,k]) for n in 1:N, k in 1:K]
    warm_start && set_start_value.(θ_jump, θ)
    # Optimize the updated model
	set_silent(model)

    optimize!(model)
    # Obtained the new parameters
    θ[:,:] = value.(θ_jump)
    [α[k,t] = exp(polynomial_trigo(t, θ[k,:], T = T)) for k in 1:K-1, t in 1:T]
    [α[K,t] = 1 for t in 1:T] # last colum is 1/normalization (one could do otherwise)
    normalization_polynomial = [1+sum(α[k,t] for k in 1:K-1) for t in 1:T]
    [α[k,t] /= normalization_polynomial[t] for k in 1:K, t in 1:T]
end

function update_dist(dist::AbstractVector{F} where F<:Exponential, θ_Y::AbstractArray, γ::AbstractVector, observations, model; warm_start = true)
    @argcheck size(γ, 1) == size(observations, 1)
	dist = copy(dist)
	θ_Y = copy(θ_Y)

    N = size(γ, 1)
	T = size(dist, 1)
    θ_jump = model[:θ_jump]
    πk = model[:πk]
    R = model[:R]

    ## Update the smoothing parameters in the JuMP model
    for n in 1:N
        set_value(πk[n], γ[n]) 
	    set_value(R[n], observations[n])
    end

    warm_start && set_start_value.(θ_jump, θ_Y[:])
    set_silent(model)

    optimize!(model)
    θ_Y[:] = value.(θ_jump)

    p = [exp(polynomial_trigo(t, θ_Y[:], T = T)) for t in 1:T]
    dist[:] = Exponential.(p)
	return dist, θ_Y
end

function update_dist(dist::AbstractVector{F} where F<:Gamma, θ_Y::AbstractArray, γ::AbstractVector, observations, model; warm_start = true)
    @argcheck size(γ, 1) == size(observations, 1)
	dist = copy(dist)
	θ_Y = copy(θ_Y)

    N = size(γ, 1)
	T = size(dist, 1)
    θ_jump = model[:θ_jump]
	a_jump = model[:a_jump]
    πk = model[:πk]
    R = model[:R]

    ## Update the smoothing parameters in the JuMP model
    for n in 1:N
        set_value(πk[n], γ[n]) 
	    set_value(R[n], observations[n])
    end

    warm_start && set_start_value.(a_jump, θ_Y[1,:]) # a parameter of Gamma
	warm_start && set_start_value.(θ_jump, θ_Y[2,:]) # θ parameter of Gamma

    optimize!(model)
	θ_Y[1,:] = value.(a_jump)
    θ_Y[2,:] = value.(θ_jump)

    a = [exp(polynomial_trigo(t, θ_Y[1,:], T = T)) for t in 1:T]
	p = [exp(polynomial_trigo(t, θ_Y[2,:], T = T)) for t in 1:T]
    dist[:] = [Gamma(a[t],p[t]) for t in 1:T]
	return dist, θ_Y

end


# * Improved and distributed compatible version

function model_for_dist_d(dist::Exponential, θ_Y::AbstractArray, T::Int; silence = true)
	d = (size(θ_Y, 1)-1)÷2

    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)
    f = 2π/T

    cos_nj = [cos(f*j*t) for t in 1:T, j in 1:d]
    sin_nj = [sin(f*j*t) for t in 1:T, j in 1:d]
    trig = [[1; interleave2(cos_nj[t,:], sin_nj[t,:])] for t in 1:T]

    @variable(model, θ_jump[j = 1:(2d+1)])
    # Polynomial P
    @NLexpression(model, Pθ[t = 1:T], sum(trig[t][j] * θ_jump[j] for j in 1:(2d+1)))

    @NLparameter(model, s_π[t = 1:T] == 1)
	@NLparameter(model, s_π_R[t = 1:T] == 1) # same initialization but will be changed later

    """
        -sum(πk[n]*(Pθ[n]+R[n]*exp(-Pθ[n])) for n in 1:N)
    """
	@NLobjective(
	model, Max,
	- sum(s_π[t]*Pθ[t] + s_π_R[t]*exp(-Pθ[t]) for t in 1:T)
	)

    # I don't know if it is the best but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:s_π] = s_π
	model[:s_π_R] = s_π_R

    return model
end

function update_dist_d(dist::AbstractVector{F} where F<:Exponential, θ_Y::AbstractArray, γ::AbstractVector, observations::AbstractArray, n_in_t, model::Model; warm_start = true)
    @argcheck size(γ, 1) == size(observations, 1)
	dist = copy(dist)
	θ_Y = copy(θ_Y)

    N = size(γ, 1)
	T = size(dist, 1)
    θ_jump = model[:θ_jump]
    s_π = model[:s_π]
    s_π_R = model[:s_π_R]

    ## Update the smoothing parameters in the JuMP model
    for t in 1:T
        set_value(s_π[t], sum(γ[n] for n in n_in_t[t]; init = 0))
	    set_value(s_π_R[t], sum(γ[n]*observations[n] for n in n_in_t[t]; init = 0))
    end

    warm_start && set_start_value.(θ_jump, θ_Y[:])

    optimize!(model)
    θ_Y[:] = value.(θ_jump)

    p = [exp(polynomial_trigo(t, θ_Y[:], T = T)) for t in 1:T]
    dist[:] = Exponential.(p)
	return dist, θ_Y
end

function model_for_dist_d(dist::Gamma, θ_Y::AbstractArray, T::Int; silence = true)
	d = (size(θ_Y, 2)-1)÷2
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_cpu_time", 60.0)
    set_optimizer_attribute(model, "max_iter", 100)    
    silence && set_silent(model)
    f = 2π/T
    cos_nj = [cos(f*j*t) for t in 1:T, j in 1:d]
    sin_nj = [sin(f*j*t) for t in 1:T, j in 1:d]
    trig = [[1; interleave2(cos_nj[t,:], sin_nj[t,:])] for t in 1:T]

    @variable(model, θ_jump[j = 1:(2d+1)])
	@variable(model, a_jump[j = 1:(2d+1)])
    # set_lower_bound(θ_jump[1], 0)
    # set_upper_bound.(θ_jump, 6)    
    # set_lower_bound(a_jump[1], 0)
    # set_upper_bound(a_jump[1], 3)    

    # Polynomial P
    @NLexpression(model, Pθ[t = 1:T], sum(trig[t][j] * θ_jump[j] for j in 1:length(trig[t])))
	@NLexpression(model, Pa[t = 1:T], sum(trig[t][j] * a_jump[j] for j in 1:length(trig[t])))

    @NLparameter(model, s_π[t = 1:T] == 1)
	@NLparameter(model, s_π_R[t = 1:T] == 1)
   	@NLparameter(model, s_π_logR[t = 1:T] == 1)

	register(model, :loggamma, 1, loggamma; autodiff = true) # cf the JuMP warning that prevent loggamma with more than one argument
    """
        - sum(πk[n]*(exp(Pa[n])*Pθ[n]+R[n]*exp(-Pθ[n])+loggamma(exp(Pa[n]))-expm1(Pa[n])*log(R[n])) for n in 1:N)
    """
	@NLobjective(
	model, Max,
		- sum(s_π[t]*exp(Pa[t])*Pθ[t] + s_π_R[t]*exp(-Pθ[t]) 
        + s_π[t]*loggamma(exp(Pa[t])) - s_π_logR[t]*expm1(Pa[t]) for t in 1:T)
	)

    model[:s_π] = s_π
	model[:s_π_R] = s_π_R
	model[:s_π_logR] = s_π_logR

    return model
end


function update_dist_d(dist::AbstractVector{F} where F<:Gamma, θ_Y::AbstractArray, γ::AbstractVector, observations::AbstractArray, n_in_t, model::Model; warm_start = true)
    @argcheck size(γ, 1) == size(observations, 1)
	dist = copy(dist)
	θ_Y = copy(θ_Y)

    N = size(γ, 1)
	T = size(dist, 1)
    θ_jump = model[:θ_jump]
	a_jump = model[:a_jump]
    s_π = model[:s_π]
	s_π_R = model[:s_π_R]
	s_π_logR = model[:s_π_logR]

    ## Update the smoothing parameters in the JuMP model
    for t in 1:T
        set_value(s_π[t], sum(γ[n] for n in n_in_t[t]; init = 0))
	    set_value(s_π_R[t], sum(γ[n]*observations[n] for n in n_in_t[t]; init = 0))
	    set_value(s_π_logR[t], sum(γ[n]*log(observations[n]) for n in n_in_t[t]; init = 0))
    end

    warm_start && set_start_value.(a_jump, θ_Y[1,:]) # a parameter of Gamma
	warm_start && set_start_value.(θ_jump, θ_Y[2,:]) # θ parameter of Gamma

    optimize!(model)
	θ_Y[1,:] = value.(a_jump)
    θ_Y[2,:] = value.(θ_jump)

    a = [exp(polynomial_trigo(t, θ_Y[1,:], T = T)) for t in 1:T]
	p = [exp(polynomial_trigo(t, θ_Y[2,:], T = T)) for t in 1:T]
    dist[:] = [Gamma(a[t],p[t]) for t in 1:T]
	return dist, θ_Y

end

function s_γα!(s_ξ, ξ, n_in_t)
    T, K = size(s_ξ, 1), size(s_ξ, 2)
    for t in 1:T
        for k in 1:K
            s_ξ[t, k] = sum(ξ[n, k] for n in n_in_t[t]; init = 0)
        end
    end
end

function model_for_α_d(θ_α::AbstractMatrix, s_γ::AbstractMatrix; silence = true)
    T, K = size(s_γ)
	d = (size(θ_α, 2)-1)÷2
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_cpu_time", 60.0)
    set_optimizer_attribute(model, "max_iter", 100)
    silence && set_silent(model)
    f = 2π/T
    cos_nj = [cos(f*j*t) for t in 1:T, j in 1:d]
    sin_nj = [sin(f*j*t) for t in 1:T, j in 1:d]
    trig = [[1; interleave2(cos_nj[t,:], sin_nj[t,:])] for t in 1:T]

    @variable(model, θ_jump[k = 1:K-1, j = 1:(2d+1)])
    # Polynomial P
    @NLexpression(model, P[t = 1:T, k = 1:K-1], sum(trig[t][j] * θ_jump[k,j] for j in 1:(2d+1)))

    @NLparameter(model, s_π[t = 1:T, k = 1:K] == s_γ[t, k])
	@NLobjective(
    model,
    Max,
    sum(sum(s_π[t, k]*P[t,k] for k in 1:K-1) - sum(s_π[t, k] for k in 1:K)*log1p(sum(exp(P[t,k]) for k in 1:K-1)) for t in 1:T) #Σ_k πk[n,k] = 1
    )
    # I don't know if it is the best but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:s_π] = s_π
    return model
end

function update_α_d!(
    α::AbstractArray where {T},
    θ::AbstractArray where T,
    s_γ::AbstractMatrix,
    model::Model;
    warm_start = true
)
    T, K = size(s_γ)

    # ξ are the filtering probablies
    θ_jump = model[:θ_jump]
    s_π = model[:s_π]
    ## Update the smoothing parameters in the JuMP model
    for t in 1:T, k in 1:K
        set_value(s_π[t,k], s_γ[t,k])
    end
    warm_start && set_start_value.(θ_jump, θ)
    # Optimize the updated model
    optimize!(model)
    # Obtained the new parameters
    θ[:,:] = value.(θ_jump)
    [α[k,t] = exp(polynomial_trigo(t, θ[k,:], T = T)) for k in 1:K-1, t in 1:T]
    [α[K,t] = 1 for t in 1:T] # last colum is 1/normalization (one could do otherwise)
    normalization_polynomial = [1+sum(α[k,t] for k in 1:K-1) for t in 1:T]
    [α[k,t] /= normalization_polynomial[t] for k in 1:K, t in 1:T]
end

# TODO use the in place version (but it was bugy)

function fit_mle(mix::Vector{F} where F<:MixtureModel, θ_α::AbstractArray, θ_Y::AbstractArray, y, n2t;
	display = :none, maxiter = 100, tol = 1e-3, robust = false, silence = true, warm_start = true)
	@argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

	N, K, T = length(y), ncomponents(mix[1]), size(mix,1)
	# TODO deal with K=1 case
	@argcheck K>1
	d = (length(θ_α[1,:])-1)÷2 #
	history = EMHistory(false, 0, [])
	# Allocate memory for in-place updates

    LL = zeros(N,K)
    γ = similar(LL)
    s_γ = zeros(T,K)

	c = zeros(N)

	# Initial parameters
	θ_α = copy(θ_α)
	θ_Y = copy(θ_Y)
    n_in_t = [findall(n2t .== t) for t in 1:T]

	α = copy(hcat(probs.(mix)...))
	dists = copy(hcat(components.(mix)...))
	types = typeof.(dists[:,1])
	dists = [convert(Vector{types[k,1]}, dists[k,:]) for k in 1:K] # I have to separate the array to preserve types

	model_α = model_for_α_d(θ_α, s_γ; silence = silence)
	model_dist = [model_for_dist_d(dists[k][1], θ_Y[k], T; silence = silence) for k in 1:K]

	# E-step
	# evaluate likelihood for each type k
    for k in 1:K, n in 1:N
        LL[n,k] = log(α[k, n2t[n]]) + logpdf(dists[k][n2t[n]], y[n])
    end
	robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
	# get posterior of each category
	c[:] = logsumexp(LL, dims = 2)
	γ[:,:] = exp.(LL .- c)

	# Loglikelihood
	logtot = sum(c)
	(display == :iter) && println("Iteration 0: logtot = $logtot")

	for it = 1:maxiter

		# M-step
		# with γ in hand, maximize (update) the parameters
        s_γα!(s_γ, γ, n_in_t)
		update_α_d!(α, θ_α, s_γ, model_α; warm_start = warm_start)
        for k in 1:K
			dists[k][:], θ_Y[k] = update_dist_d(dists[k], θ_Y[k], γ[:,k], y, n_in_t, model_dist[k], warm_start = warm_start)
		end
		# E-step
		# evaluate likelihood for each type k
        for k in 1:K, n in 1:N
		    LL[n,k] = log(α[k,n2t[n]]) + logpdf(dists[k][n2t[n]], y[n])
        end
		robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
		# get posterior of each category
		c[:] = logsumexp(LL, dims = 2)
		γ[:,:] =  exp.(LL .- c)

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

	return [MixtureModel( [dists[k][t] for k in 1:K], [α[k,t] for k in 1:K] ) for t in 1:T], θ_α, θ_Y, history
end


function fit_em_GE(mix::Vector{F} where F<:MixtureModel, θ_α::AbstractArray, θ_Y::AbstractArray, y, n2t;
	display = :none, maxiter = 100, tol = 1e-3, robust = false, silence = true, warm_start = true)
	@argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

	N, K, T = length(y), ncomponents(mix[1]), size(mix,1)
	# TODO deal with K=1 case
	@argcheck K>1
	d = (length(θ_α[1,:])-1)÷2 #
	history = EMHistory(false, 0, [])
	# Allocate memory for in-place updates

    LL = zeros(N,K)
    γ = similar(LL)
	c = zeros(N)

	# Initial parameters
	θ_α = copy(θ_α)
	θ_Y = copy(θ_Y)

	α = copy(hcat(probs.(mix)...))
	dists = copy(hcat(components.(mix)...))
	types = typeof.(dists[:,1])
	dists = [convert(Vector{types[k,1]},dists[k,:]) for k in 1:K] # I have to separate the array to preserve types

	model_α = model_for_α(θ_α, γ, n2t, T; silence = silence)
	model_dist = [model_for_dist(dists[k][1], θ_Y[k], γ[:,k], y, n2t, T; silence = silence) for k in 1:K]

	# E-step
	# evaluate likelihood for each type k
    for k in 1:K, n in 1:N
        LL[n,k] = log(α[k, n2t[n]]) + logpdf(dists[k][n2t[n]], y[n])
    end
	robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
	# get posterior of each category
	c[:] = logsumexp(LL, dims = 2)
	γ[:,:] =  exp.(LL .- c)

	# Loglikelihood
	logtot = sum(c)
	(display == :iter) && println("Iteration 0: logtot = $logtot")

	for it = 1:maxiter

		# M-step
		# with γ in hand, maximize (update) the parameters
		update_α!(α, θ_α, γ, model_α; warm_start = warm_start)
		for k in 1:K
			dists[k][:], θ_Y[k] = update_dist(dists[k], θ_Y[k], γ[:,k], y, model_dist[k], warm_start = warm_start)
			# update_dist!(dists[k], θ_Y[k], γ[:,k], y, model_dist[k], warm_start = warm_start)
		end
		# E-step
		# evaluate likelihood for each type k
        for k in 1:K, n in 1:N
		    LL[n,k] = log(α[k,n2t[n]]) + logpdf(dists[k][n2t[n]], y[n])
        end
		robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
		# get posterior of each category
		c[:] = logsumexp(LL, dims = 2)
		γ[:,:] =  exp.(LL .- c)

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

	return [MixtureModel( [dists[k][t] for k in 1:K], [α[k,t] for k in 1:K] ) for t in 1:T], θ_α, θ_Y, history
end



