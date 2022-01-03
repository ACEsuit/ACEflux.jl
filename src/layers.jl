using Flux, ForwardDiff, Zygote, StaticArrays
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent, ZeroTangent
using Flux: @functor
using ChainRules: @thunk

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

mutable struct Linear_ACE{TW,TM}
   weight::TW
   m::TM 
end

function Linear_ACE(ord::Int, maxdeg::Int, Nprop)
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

function diag2vect(A)
   n = Int(sqrt(length(A[1])))
   tmp = []
   for i in 1:n
      for j in 1:n
         if(i==j)
            push!(tmp, A[i,j])
         end
      end
   end
   return tmp
end

function adj_evaluate(dp, W, M::ACE.LinearACEModel, cfg)
   
   set_params!(M, matrix2svector(W)) #TODO is it necesary?
   
   function g_par()
      gp_ = ACE.grad_params(M, cfg)
      gp = [ ACE.val.(diag2vect(a) .* dp) for a in gp_ ]
      return svector2matrix(gp)
   end
   
   g_cfg = () -> ACE._rrule_evaluate(dp, M, cfg) # rrule for cfg only...
   return NoTangent(), g_par(), NoTangent(), g_cfg()
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
     
      grad = ACE.adjoint_EVAL_D(M, M.evaluator, cfg, dq_ace)

      # gradient w.r.t parameters: 
      #sdp = SVector(dp...)
      #grad_params = grad .* Ref(sdp)
      #svector2matrix(grad_params)
      grad_params = zeros(length(dp), length(grad))
      for i in 1:length(dp)
         grad_params[i,:] = dp[i] .* ACE.val.(grad)
      end

      # gradient w.r.t. dp    # TODO: remove the |> Vector? 
      grad_dp = sum( M.c[k] * grad[k] for k = 1:length(grad) )  |> Vector 
      return(NoTangent(), ACE.val.(grad_dp), grad_params, NoTangent(), NoTangent())
   end
   return(adj_evaluate(dp, W, M, cfg), secondAdj)
end


# # ------------------------------------------------------------------------
# #    Generic non linearity layer
# # ------------------------------------------------------------------------

"""
A layer to wrap an ACE layer in a non-linearity.
"""

struct GenLayer{TF}
   F::TF
end 

(L::GenLayer)(x) = L.F(x)

function rrule(L::GenLayer, x)
   f = L.F(x)
   return f, dp -> _rrule_GenLayer(L, dp, x)
end

function _rrule_GenLayer(L, dp, x)
   _, pb = Zygote.pullback(L.F, x)
   return NoTangent(), first(pb(dp))
end

function rrule(::typeof(_rrule_GenLayer), L, dp, x)
   val = _rrule_GenLayer(L, dp, x)
   function pb(dq) 
      gx = Zygote.hessian(L.F, x) * dq[2]
      gdp = sum(first(Zygote.gradient(L.F, x)) .* dq[2]) #TODO is this needed?
      return NoTangent(), NoTangent(), NoTangent(), gx
   end 
   return val, pb 
end 
