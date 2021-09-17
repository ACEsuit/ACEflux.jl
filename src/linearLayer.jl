using Flux, ForwardDiff, Zygote, StaticArrays
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent
using Flux: @functor

using ACE
using ACE: State, NaiveTotalDegree, SymmetricBasis, evaluate, LinearACEModel, set_params!, grad_params, grad_config, grad_params_config


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
   W = rand(Nprop, length(basis))  #matrix for Flux 
   Wsv =  [SVector{size(W)[1]}(W[:,i]) for i in 1:size(W)[2]]  #SVector for ACE
   LM = LinearACEModel(basis, Wsv, evaluator = :standard) 
   return Linear_ACE(W,LM)
end

@functor Linear_ACE #so that flux can find the parameters
 
#forward pass
(y::Linear_ACE)(cfg) = _eval_linear_ACE(y.weight, y.m, cfg)

#energy evaluation
function _eval_linear_ACE(Wt, M, cfg)
   W = [SVector{size(Wt)[1]}(Wt[:,i]) for i in 1:size(Wt)[2]] #SVector conversion
   set_params!(M, W)
   E = getproperty.(evaluate(M ,cfg), :val)
   return E
end

#adj is outside chainrule so we can define a chainrule for it
function adj(dp, Wt, M, cfg)

   W = [SVector{size(Wt)[1]}(Wt[:,i]) for i in 1:size(Wt)[2]]  #SVector conversion
   set_params!(M, W) 

   #Params derivative
   gparams = grad_params(M ,cfg)  #gradient is returned as SVector for now
   #make them numbers rather than invariants
   grad = zeros(SVector{length(gparams[1])},length(gparams))
   for i = 1:length(gparams)
      grad[i] = getproperty.(gparams[i], :val)
   end
   #multiply by dp
   temp_grad = [dp .* grad[i] for i in 1:length(grad)]
   #we convert our SVector into a matrix
   gradMatrix = zeros(length(temp_grad[1]), length(temp_grad))
   for i in 1:length(temp_grad)
      gradMatrix[:,i] = temp_grad[i]
   end

   #Forces
   #TODO do we want to multiply dp?
   gconfig = grad_config(M,cfg)
  
   return (NoTangent(), gradMatrix, NoTangent(), gconfig)
end

function ChainRules.rrule(::typeof(_eval_linear_ACE), Wt, M, cfg)
   E = _eval_linear_ACE(Wt, M, cfg)
   return E, dp -> adj(dp, Wt, M, cfg)
end

#it actually returns all the gradient combinations. So der of parameters twice, of model
#twice of model and parameters, etc. However, we can just pass NoTangent() when we don't need them
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
