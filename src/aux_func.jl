argmaxrow(A::AbstractArray{T}) where T<:Real = [argmax(A[i,:]) for i in 1:size(A,1)]

function lagou(a::AbstractArray; dims = 1, kwargs...)
    T = size(a, dims)
    mapslices(x->lagou(x,T; kwargs...), a, dims = dims)
end

function lagou(hmm::PeriodicHMM; kwargs...)
    T = size(hmm.A, 3)
    lagou(hmm, T; kwargs...)
end

function lagou(mix::Array{MixtureModel{Univariate, Continuous, Distribution{Univariate, Continuous}, Float64}, 3}; kwargs...)
    T = size(mix, 2)
    lagou(mix, T; kwargs...)
end

function lagou(hmm::HierarchicalPeriodicHMM; kwargs...)
    T = size(hmm.A, 3)
    lagou(hmm, T; kwargs...)
end

function lagou(a::AbstractVector, T::Int; window = -15:15, kernel = :step)
    cycle = CyclicArray(1:T, "1D")
    if collect(window) == [0]
        return a
    elseif kernel == :step
        return [mean(a[cycle[t+tt]] for tt in window) for t in 1:T]
    elseif kernel == :step_fast # faster but produces for small positive values small negative numbers
        m = similar(a)
        N = length(window)
        m[1] = mean(a[tt] for tt in cycle[(window) .+ 1])
        for t in eachindex(a)[2:end]
            m[t] = (N*m[t-1] + a[cycle[t+window.stop]] - a[cycle[t-1+window.start]])/N
        end
        return m
    elseif kernel == :Epanechnikov
        ker(u) = 3/4*(1-abs(u)^2)
        s = [sum(ker(tt/window.stop) for tt in window) for t in 1:T]
        return [sum(a[cycle[t+tt]] * ker(tt/window.stop) for tt in window) for t in 1:T]./s
    end
end

function lagou(B::Vector{F}, T::Int; kwargs...) where F<:Bernoulli
    K = size(B, 1)
    return Bernoulli.(lagou(succprob.(B), T; kwargs...))
end

function lagou(B::Vector{Product{Discrete, F, Vector{F}}}, T::Int; kwargs...) where F<:Categorical
    K = size(B, 1)
    D = length(B[1])
    smooth_param = lagou([probs.(B[t].v) for t in 1:T], T; kwargs...)
    d = [product_distribution([Categorical(abs.(smooth_param[t][j])) for j in 1:D]) for t in 1:T]
    return d
end

function lagou(B::Vector{Product{Discrete, F, Vector{F}}}, T::Int; kwargs...) where F<:Bernoulli
    K = size(B, 1)
    D = length(B[1])
    smooth_param = lagou([succprob.(B[t].v) for t in 1:T], T; kwargs...)
    d = [product_distribution([Bernoulli(abs.(smooth_param[t][j])) for j in 1:D]) for t in 1:T]
    return d
end

function lagou(hmm::PeriodicHMM{Multivariate}, T::Int; kwargs...)
    K = size(hmm.A, 1)
    # abs to ensure positivness (sometime means of super small nuber produce negative number small in abs value)
    Q = lagou(hmm.A, dims = 3; kwargs...)

    d = lagou(hmm.B, dims = 2; kwargs...)

    return PeriodicHMM(Q,d)
end

function lagou(B::AbstractVector{F} where F<:Bernoulli, T; kwargs...)
    param = succprob.(B)
    lparam = lagou(param, T; kwargs...)
    Bernoulli.(abs.(lparam[t]))
end

function lagou(hmm::HierarchicalPeriodicHMM, T::Int; kwargs...)
    cycle = CyclicArray(1:T, "1D")
    K = size(hmm.A, 1)

    Q = lagou(hmm.A, dims = 3; kwargs...)

    d = lagou(hmm.B, dims = 2; kwargs...)

    return HierarchicalPeriodicHMM(Q,d)
end

function lagou(mix::Vector{F}, T::Int; kwargs...) where F<:MixtureModel{Univariate}
    data_type = typeof.(Distributions.components(mix[1]))
    C = length(data_type)
    mix_s = similar(mix)
    p = params.(mix)
    smooth_α = lagou(getindex.(p,2); kwargs...)
    [p[t][2][:] = smooth_α[t] for t in 1:T]
    for c in 1:C
        len = length(p[1][1][c])
        smooth_c = zeros(T,len)
        for l in 1:len
            smooth_c[:,l] = lagou([p[t][1][c][l] for t in 1:T]; kwargs...)
        end
        [p[t][1][c] = tuple([smooth_c[t,l] for l in 1:len]...) for t in 1:T]
    end
    for t in 1:T
        mix_s[t] = MixtureModel([data_type[c](p[t][1][c]...) for c in 1:C], p[t][2])
    end
    return mix_s
end

# function lagou(mix::Array{MixtureModel{Univariate, Continuous, Distribution{Univariate, Continuous}, Float64}, 3}, T::Int; kwargs...)
#     data_type = typeof.(Distributions.components(mix[1]))
#     C = length(data_type)
#     K, D = size(mix, 1), size(mix, 3)
#     mix_s = similar(mix)
#     for k in 1:K, j in 1:D
#         p = params.(mix[k,:,j])
#         ps = lagou(p; kwargs...)
#         for t in 1:T
#             mix_s[k,t,j] = MixtureModel([data_type[c](ps[t][1][c]...) for c in 1:C], ps[t][2])
#         end
#     end
#     return mix_s
# end
#
# index_cum(c, s) = c == 1 ? 1 : s[c-1]+1
#
# function lagou(p::Vector{Tuple{Vector{Tuple{Float64, Vararg{Float64, N} where N}}, Vector{Float64}}}, T::Int; kwargs...)
#     component = length(p[1][1])
#     size_component = [length(p[1][1][c]) for c in 1:component]
#     ca = unfold.(p)
#
#     cas = lagou(ca, T, kwargs...)
#     [tuple([Tuple(cas[t][index_cum(c, size_component):cumsum(size_component)[c]]) for c in 1:component], cas[t][(sum(size_component)+1):end]) for t in 1:T]
# end
##


# x[first] = previous day -> n-1
# x[last] = farest day -> n-> c
bin2digit(x) = sum(x[length(x) - i + 1]*2^(i-1) for i in 1:length(x)) + 1
bin2digit(x::Tuple) = bin2digit([x...])

function dayx(lag_obs)
    memory = length(lag_obs)
    t = tuple.([lag_obs[m] for m in 1:memory]...)
    bin2digit.(t)
end

function conditional_to(observations, memory::Int)
    if memory == 0
        return ones(Int, size(observations))
    elseif memory<5
        past =  [0  1  0  1  1  0  1  0  0  0;
                 1  1  0  1  1  1  1  1  1  1;
                 1  1  0  1  1  1  0  1  1  1;
                 1  1  0  1  1  0  0  0  1  0;
                 1  1  0  1  1  0  0  1  0  1]
        lag_obs = [copy(lag(observations, m)) for m in 1:memory]  # transform dry=0 into 1 and wet=1 into 2 for array indexing
        [lag_obs[m][1:m,:] .= reverse(past[1:m,:], dims=1) for m in 1:memory] # avoid the missing first row

        return dayx(lag_obs)
    else
        println("Not implemented need to add initial conditions (or fake ones)")
    end
end


function conditional_to(observations, memory::Float64)
    if memory == 0
        return ones(Int, size(observations))
    elseif memory<5
        past =  [0  1  0  1  1  0  1  0  0  0;
                 1  1  0  1  1  1  1  1  1  1;
                 1  1  0  1  1  1  0  1  1  1;
                 1  1  0  1  1  0  0  0  1  0;
                 1  1  0  1  1  0  0  1  0  1]
        lag_obs = [copy(lag(observations, m)) for m in 1:memory]  # transform dry=0 into 1 and wet=1 into 2 for array indexing
        [lag_obs[m][1:m,:] .= reverse(past[1:m,:], dims=1) for m in 1:memory] # avoid the missing first row

        return dayx(lag_obs)
    else
        println("Not implemented need to add initial conditions (or fake ones)")
    end
end

function idx_observation_of_past_cat(lag_cat, n2t, T, K, size_memory)
    # Matrix(T,D) of vector that give the index of data of same past.
    # ie. size_memory = 1 (no memory) -> every data is in category 1
    # ie size_memory = 2 (memory on previous day) -> idx_tj[t,j][1] = vector of index of data where previous day was dry, idx_tj[t,j][2] = index of data where previous day was wet
    D = size(lag_cat, 2)
    idx_tj = Matrix{Vector{Vector{Int}}}(undef, T, D)
    n_in_t = [findall(n2t .== t) for t in 1:T] # could probably be speeded up e.g. recusivly suppressing already classified label with like sortperm
    @inbounds for t in OneTo(T)
        n_t = n_in_t[t]
        for i in OneTo(K)
            for j in 1:D
                # apparently the two following takes quite long ~29ms for all the loops
                # TODO improve time!
                n_tm = [findall(lag_cat[n_t, j] .== m) for m in 1:size_memory]
                idx_tj[t, j] = [n_t[n_tm[m]] for m in 1:size_memory]
                ##
            end
        end
    end
    return idx_tj
end

## Trigo part

function polynomial_trigo(t::Number,β; T = 366)
    d = (length(β)-1)÷2
    if d == 0
        return β[1]
    else
        f = 2π/T
        # everything is shifted from 1 from usual notation due to array starting at 1
        return β[1] + sum(β[2*l]*cos(f*l*t) + β[2*l+1]*sin(f*l*t) for l in 1:d)
    end
end

function polynomial_trigo(t::AbstractArray,β; T = 366)
    d = (length(β)-1)÷2
    if d == 0
        return β[1]
    else
        f = 2π/T
        # everything is shifted from 1 from usual notation due to array starting at 1
        return β[1] .+ sum(β[2*l]*cos.(f*l*t) + β[2*l+1]*sin.(f*l*t) for l in 1:d)
    end
end

interleave2(args...) = collect(Iterators.flatten(zip(args...))) # merge two vector with alternate elements

function fit_Q!(p::AbstractArray, A::AbstractArray{N,2} where N; silence = true)
    T, K = size(A,2), size(A,1)
    @assert K-1==size(p,1)
    d = (size(p,2)-1)÷2
    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)
    f = 2π/T
    cos_nj = [cos(f*j*t) for t in 1:T, j in 1:d]
    sin_nj = [sin(f*j*t) for t in 1:T, j in 1:d]

    trig = [[1; interleave2(cos_nj[t,:], sin_nj[t,:])] for t in 1:T]

    @variable(model, p_jump[k = 1:(K-1), j = 1:(2d+1)])
    set_start_value.(p_jump, p)
    # Polynomial P_kl

    @NLexpression(model, Pol[t = 1:T, k = 1:K-1], sum(trig[t][j] * p_jump[k,j] for j in 1:length(trig[t])))

    @NLobjective(
    model,
    Min,
    sum( (A[k,t] - exp(Pol[t,k])/(1+sum(exp(Pol[t,l]) for l in 1:K-1)))^2 for k in 1:K-1, t in 1:T)
        +sum( (A[K,t] - 1/(1+sum(exp(Pol[t,l]) for l in 1:K-1)))^2 for t in 1:T)
    )
    optimize!(model)
    p[:,:] = value.(p_jump)
end

m_Bernoulli(t, p; T = 366) = 1 ./(1 .+ exp.(polynomial_trigo(t,p; T = T)))
function fit_Y!(p::AbstractVector, B::AbstractVector)
    T = size(B,1)
    p[:] = curve_fit((t,p)->m_Bernoulli(t,p,T=T), collect(1:T), B, convert(Vector,p)).param
end
# slower
# function fit_Y2!(p::AbstractVector, B::AbstractVector; silence = true)
#     T = size(B,1)
#     d = (size(p,1)-1)÷2
#     model = Model(Ipopt.Optimizer)
#     silence && set_silent(model)
#     f = 2π/T
#     cos_nj = [cos(f*j*t) for t in 1:T, j in 1:d]
#     sin_nj = [sin(f*j*t) for t in 1:T, j in 1:d]

#     trig = [[1; interleave2(cos_nj[t,:], sin_nj[t,:])] for t in 1:T]

#     @variable(model, p_jump[j = 1:(2d+1)])
#     set_start_value.(p_jump, p)
#     # Polynomial P_kl

#     @NLexpression(model, Pol[t = 1:T], sum(trig[t][j] * p_jump[j] for j in 1:length(trig[t])))

#     @NLobjective(
#     model,
#     Min,
#         sum( (B[t] - 1/(1+exp(Pol[t])))^2 for t in 1:T)
#     )
#     optimize!(model)
#     p[:] = value.(p_jump)
# end

# function fit_Y2(p::AbstractVector, B::AbstractVector; silence = true)
#     T = size(B,1)
#     d = (size(p,1)-1)÷2
#     model = Model(Ipopt.Optimizer)
#     silence && set_silent(model)
#     f = 2π/T
#     cos_nj = [cos(f*j*t) for t in 1:T, j in 1:d]
#     sin_nj = [sin(f*j*t) for t in 1:T, j in 1:d]

#     trig = [[1; interleave2(cos_nj[t,:], sin_nj[t,:])] for t in 1:T]

#     @variable(model, p_jump[j = 1:(2d+1)])
#     set_start_value.(p_jump, p)
#     # Polynomial P_kl

#     @NLexpression(model, Pol[t = 1:T], sum(trig[t][j] * p_jump[j] for j in 1:length(trig[t])))

#     @NLobjective(
#     model,
#     Min,
#         sum( (B[t] - 1/(1+exp(Pol[t])))^2 for t in 1:T)
#     )
#     return model
# end
