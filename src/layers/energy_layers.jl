
using ACE: State, NaiveTotalDegree, SymmetricBasis, evaluate, LinearACEModel, set_params!, grad_params
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent 
using Flux
using Flux: @functor
using ACE
using StaticArrays

#TODO make this cutoff flexible
using JuLIP: EMT
using JuLIP

# # ------------------------------------------------------------------------
# #    ACE Energy Layer with non-linearity
# # ------------------------------------------------------------------------

"""
`mutable struct NL_ENERGY_ACE{TW, TM, Tσ}` : 
A simple ACE layer for multiple properties all of them with the same user set
parameters, but different weights. wrapping a non-linearity as an activation 
function one can model Finnis Sinclair or other outer nonlinear ACE model.
If the goal is to compose ACE models, the recomended layer is ENERGY_ACE
which returns all properties.
"""
mutable struct NL_ENERGY_ACE{TW, TM, Tσ}
   weight::TW
   m::TM 
   σ::Tσ
end

function NL_ENERGY_ACE(maxdeg, ord, Nprop, Tσ)
   #building the basis
   B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = NaiveTotalDegree())
   pibasis = PIBasis(B1p, ord, maxdeg; property = ACE.Invariant())
   basis = SymmetricBasis(pibasis, ACE.Invariant());   

   #create a multiple property model
   W = rand(SVector{Nprop,Float64}, length(basis))
   LM = LinearACEModel(basis, W, evaluator = :standard) 
   return NL_ENERGY_ACE(W,LM,Tσ)
end

@functor NL_ENERGY_ACE

function (y::NL_ENERGY_ACE)(at)
   Rs = []
   nlist = neighbourlist(at, cutoff(EMT()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      tmpRs=ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      append!(Rs,[tmpRs])
   end 
   set_params!(y.m, y.weight)
   E = y.σ(sum([getproperty.(evaluate(y.m ,r), :val) for r in Rs]))
   return E
end

function ChainRules.rrule(y::NL_ENERGY_ACE, at)
   Rs = []
   nlist = neighbourlist(at, cutoff(EMT()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      tmpRs=ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      append!(Rs,[tmpRs])
   end 
   set_params!(y.m, y.weight)
   E = sum([getproperty.(evaluate(y.m ,r), :val) for r in Rs])

   function adj(dp)
      gσt = Flux.gradient(y.σ, E)[1]
      gσ = SVector{length(gσt)}(gσt)
      gparams = sum([grad_params(y.m ,r) for r in Rs])
      grad = zeros(SVector{length(gparams[1])},length(gparams))
      for i = 1:length(gparams)
         grad[i] = getproperty.(gparams[i], :val) .* gσ
      end
      return (dp * grad, NoTangent()) 
   end
   return y.σ(E), adj
end

# # ------------------------------------------------------------------------
# #    ACE Energy Layer without non-linearity
# # ------------------------------------------------------------------------

"""
`mutable struct ENERGY_ACE{TW, TM, Tσ}` : 
A simple ACE layer for multiple properties all of them with the same user set
parameters, but different weights. This layer will return the calculated properties
(energies in this case) as an array that can be inmeadiatley fed into other layers.
For composition of ACE models use another layer that takes and returns ACE states.
"""
mutable struct ENERGY_ACE{TW, TM}
   weight::TW
   m::TM 
end

function ENERGY_ACE(maxdeg, ord, Nprop)
   #building the basis
   B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = NaiveTotalDegree())
   pibasis = PIBasis(B1p, ord, maxdeg; property = ACE.Invariant())
   basis = SymmetricBasis(pibasis, ACE.Invariant());   

   #create a multiple property model
   W = rand(SVector{Nprop,Float64}, length(basis))
   LM = LinearACEModel(basis, W, evaluator = :standard) 
   return ENERGY_ACE(W,LM)
end

@functor ENERGY_ACE

function (y::ENERGY_ACE)(at)
   Rs = []
   nlist = neighbourlist(at, cutoff(EMT()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      tmpRs=ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      append!(Rs,[tmpRs])
   end 
   set_params!(y.m, y.weight)
   E = sum([getproperty.(evaluate(y.m ,r), :val) for r in Rs])
   return E
end

function ChainRules.rrule(y::ENERGY_ACE, at)
   Rs = []
   nlist = neighbourlist(at, cutoff(EMT()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      tmpRs=ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      append!(Rs,[tmpRs])
   end 
   set_params!(y.m, y.weight)
   E = sum([getproperty.(evaluate(y.m ,r), :val) for r in Rs])

   function adj(dp)
      gparams = sum([grad_params(y.m ,r) for r in Rs])
      grad = zeros(SVector{length(gparams[1])},length(gparams))
      for i = 1:length(gparams)
         grad[i] = getproperty.(gparams[i], :val)
      end
      return (dp * grad, NoTangent()) 
   end
   return E, adj
end

#########################################################################

mutable struct LLayer{TW, TM}
   weight::TW
   m::TM 
end

function LLayer(maxdeg, ord, k)
   #building the basis
   B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = NaiveTotalDegree())
   pibasis = PIBasis(B1p, ord, maxdeg; property = ACE.Invariant())
   basis = SymmetricBasis(pibasis, ACE.Invariant());   

   #create a multiple property model
   W = rand(length(basis))
   LM = LinearACEModel(basis, W, evaluator = :standard) 
   return LLayer(W,LM)
end

@functor LLayer

function (y::LLayer)(at)
   Rs = []
   nlist = neighbourlist(at, cutoff(EMT()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      tmpRs=ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      append!(Rs,[tmpRs])
   end 
   set_params!(y.m, y.weight)
   E = sum([getproperty(evaluate(y.m ,r), :val) for r in Rs])
   return [E]
end

function ChainRules.rrule(y::LLayer, at)
   Rs = []
   nlist = neighbourlist(at, cutoff(EMT()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      tmpRs=ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      append!(Rs,[tmpRs])
   end 
   set_params!(y.m, y.weight)
   E = sum([getproperty(evaluate(y.m ,r), :val) for r in Rs])

   function adj(dp)
      gparams = sum([grad_params(y.m ,r) for r in Rs])
      @show dp .* getproperty.(gparams, :val)
      return (dp .* getproperty.(gparams, :val), NoTangent()) 
   end
   return [E], adj
end