using Flux, ForwardDiff, Zygote, StaticArrays
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent, ZeroTangent
using Flux: @functor

using ACE
using ACE: O3, State, SymmetricBasis, evaluate, LinearACEModel, set_params!, grad_params, grad_config, grad_params_config

#functions for converting SVectors and matrices
function svector2matrix(sv)
   M = zeros(length(sv[1]), length(sv))
   for i in 1:length(sv)
      M[:,i] = sv[i]
   end
   return M
end

function matrix2svector(M)
   sv = [SVector{size(M)[1]}(M[:,i]) for i in 1:size(M)[2]]
   return sv
end

getprops(x) = getproperty.(x, :val)

# # ------------------------------------------------------------------------
# #    ACE linear layer
# # ------------------------------------------------------------------------

"""
`mutable struct Linear_ACE{TW, TM}` : 
A layer to calculate site energies for multiple properties.
The forward pass will calculate site energies for Nproperties, the derivative
w.r.t configurations or atoms will return the Forces. And the derivative of
Forces w.r.t parameters is done through a second rrule. All other mixed derivatives
are not computed, and dF_params is computed through adjoints. The adjoint implementation
and all other derivatives live in ACE.jl, this is simply a wrapper. 
"""

mutable struct Linear_ACE{TW, TM}
   weight::TW
   m::TM 
end

function Linear_ACE(maxdeg, ord, Nprop)
    #building the basis
    Bsel = SimpleSparseBasis(ord, maxdeg)
    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
    φ = ACE.Invariant()
    bsis = SymmetricBasis(φ, B1p, O3(), Bsel)

   #create a multiple property model
   W = rand(Nprop, length(bsis))
   LM = LinearACEModel(bsis, matrix2svector(W), evaluator = :standard) 
   return Linear_ACE(W,LM)
end

@functor Linear_ACE #so that flux can find the parameters
 
#forward pass
(y::Linear_ACE)(cfg) = _eval_linear_ACE(y.weight, y.m, cfg)

#energy evaluation
function _eval_linear_ACE(W, M, cfg)
   set_params!(M, matrix2svector(W))
   E = getproperty.(evaluate(M ,cfg), :val)
   return E
end

function adj_evaluate(dp, W, M::ACE.LinearACEModel, cfg)
   set_params!(M, matrix2svector(W)) #TODO is it necesary?
   gp_ = ACE.grad_params(M, cfg)
   gp = [ ACE.val.(a .* dp) for a in gp_ ]
   g_cfg = ACE._rrule_evaluate(dp, M, cfg) # rrule for cfg only...
   return NoTangent(), svector2matrix(gp), NoTangent(), g_cfg
end

function ChainRules.rrule(::typeof(_eval_linear_ACE), W, M, cfg)
   E = _eval_linear_ACE(W, M, cfg)
   return E, dp -> adj_evaluate(dp, W, M, cfg)
end

function ChainRules.rrule(::typeof(adj_evaluate), dp, W, M, cfg)
   function secondAdj(dq_)
      @assert dq_[1] == dq_[2] == dq_[3] == ZeroTangent()
      @assert dq_[4] isa AbstractVector{<: SVector}
      @assert length(dq_[4]) == length(cfg)
      dq = dq_[4]  # Vector of SVector
      dq_ace = [ ACE.DState(rr = dqi) for dqi in dq ]
     
      grad = ACE.adjoint_EVAL_D1(M, M.evaluator, cfg, dq_ace)

      # gradient w.r.t parameters: 
      sdp = SVector(dp...)
      grad_params = grad .* Ref(sdp)

      # gradient w.r.t. dp    # TODO: remove the |> Vector? 
      grad_dp = sum( M.c[k] * grad[k] for k = 1:length(grad) )  |> Vector 

      return(NoTangent(), grad_dp, svector2matrix(grad_params), NoTangent(), NoTangent())
   end
   return(adj_evaluate(dp, W, M, cfg), secondAdj)
end
