"""
Seasonal Hierarchical Hidden Markov Models for Julia.

Extansion of the [HMMBase](https://maxmouchet.github.io/HMMBase.jl/stable/) package.
"""
module SHHMM

# TODO In place version (the SharedArray thing does not really work with Slurm on a cluster because threads do not necessarly have shared memory)
using Distributed #, SharedArrays

using JuMP, Ipopt

using ArgCheck
using Clustering
using Distributions
# using Hungarian
using LinearAlgebra
using CyclicArrays # Should be removed in future versions to avoid extra dependency?

using CyclicArrays: CyclicArray
using Base: OneTo
using Random: AbstractRNG, GLOBAL_RNG
using ShiftedArrays: lag, lead
using StatsFuns: logsumexp
using StatsBase: Weights
using LsqFit
using SpecialFunctions
# Extended functions
import Base: ==, copy, rand, size
import Distributions: MixtureModel, fit_mle, loglikelihood

export
    # hmm.jl
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
include("parametric_BW.jl") 
include("parametric_BW_distributed.jl") 

include("aux_func.jl")
include("mixture.jl")


end # module
