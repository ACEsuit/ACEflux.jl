using Flux, ForwardDiff, Zygote, StaticArrays
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent
using Flux: @functor

using ACE
using ACE: State, NaiveTotalDegree, SymmetricBasis, evaluate, LinearACEModel, set_params!, grad_params, grad_config, grad_params_config

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
   B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = NaiveTotalDegree())
   pibasis = PIBasis(B1p, ord, maxdeg; property = ACE.Invariant())
   basis = SymmetricBasis(pibasis, ACE.Invariant());   

   #create a multiple property model
   W = rand(Nprop, length(basis))
   LM = LinearACEModel(basis, matrix2svector(W), evaluator = :standard) 
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

#adj is outside chainrule so we can define a chainrule for it
function adj(dp, W, M, cfg)

   set_params!(M, matrix2svector(W)) 

   g_p = getprops.(grad_params(M ,cfg))
   for i = 1:length(g_p) 
      g_p[i] = g_p[i] .* dp
   end

   g_cfg = grad_config(M,cfg)
   for i = 1:size(g_cfg,1) #loops over number of configs
      for j = 1:length(dp) #loops over properties
         g_cfg[i,j] *= dp[j] 
      end
   end

   return (NoTangent(), g_p, NoTangent(), g_cfg)
end

function ChainRules.rrule(::typeof(_eval_linear_ACE), Wt, M, cfg)
   E = _eval_linear_ACE(Wt, M, cfg)
   return E, dp -> adj(dp, Wt, M, cfg)
end



#currently not working
function ChainRules.rrule(::typeof(adj), dp, W, M, cfg)
   function secondAdj(dq)   
      #TODO wait for AR branch to merge into main of ACE, then call adjoint

      #temp_grad = ACE.adjoint_EVAL_D(M, cfg, dq[4])
      temp_grad = rand(SVector{length(M.c[1]),Float64}, length(M.c))

      #we convert our SVector into a matrix
      grad_force_params = zeros(length(temp_grad[1]), length(temp_grad))
      for i in 1:length(temp_grad)
         grad_force_params[:,i] = temp_grad[i]
      end

      return(NoTangent(), NoTangent(), grad_force_params, NoTangent(), NoTangent())
   end
   return(adj(dp, W, M, cfg), secondAdj)
end
