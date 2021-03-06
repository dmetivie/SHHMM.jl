"""
Seasonal Hierarchical Hidden Markov Models for Julia.

Extansion of the [HMMBase](https://maxmouchet.github.io/HMMBase.jl/stable/) package.
"""
module SHHMM

# TODO!  Package SHHMM does not have CyclicArrays in its dependencies
using Distributed
using ArgCheck
using Distributions
using LinearAlgebra
using JuMP
using Ipopt
using LsqFit
using SpecialFunctions
using Base: OneTo
using Random: AbstractRNG, GLOBAL_RNG
using StatsFuns: logsumexp
using StatsBase: Weights
using ShiftedArrays: lag, lead
using CyclicArrays: CyclicArray
using Dates: now

# Extended functions
import Base: ==, copy, rand, size
import Distributions: MixtureModel, fit_mle, loglikelihood

export
    # shhmm.jl
    AbstractHMM,
    HMM,
    copy,
    rand,
    size,
    nparams,
    permute,
    statdists,
    istransmat,
    istransmats,
    likelihoods,
    # messages.jl
    forward,
    backward,
    posteriors,
    loglikelihood,
    # likelihoods.jl
    loglikelihoods,
    # mle.jl
    fit_mle,
    # viterbi.jl
    viterbi,
    # utilities.jl,
    gettransmat,
    randtransmat,
    remapseq,
    # PeriodicHMM
    PeriodicHMM,
    # # HierarchicalPeriodicHMM
    HierarchicalPeriodicHMM,
    bin2digit,
    dayx,
    conditional_to,
    # Slice fit for initialization
    sort_wrt_ref!,
    fit_mle_all_slices,
    # Trig
    polynomial_trigo,
    fit_Y!,
    fit_Q!,
    HierarchicalPeriodicHMM_trig


include("hmm.jl")
include("mle.jl")
include("mle_init.jl")
include("messages.jl")
include("viterbi.jl")
include("likelihoods.jl")
include("utilities.jl")
include("experimental.jl")
include("periodichmm.jl")
include("periodichmm_leap.jl")
include("periodichmm_leap_hierarchical.jl")
include("fit_mle_slice.jl")
# include("parametric_BW.jl") 
include("parametric_BW_distributed.jl")

include("aux_func.jl")
include("mixture.jl")


end # module
