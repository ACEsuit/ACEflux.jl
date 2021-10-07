using Flux, ForwardDiff, Zygote, StaticArrays
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent
using Flux: @functor
using JuLIP

using ACE
using ACE: O3, val, State, SymmetricBasis, evaluate, LinearACEModel, set_params!, grad_params, grad_config, grad_params_config

#neighboor list finder, It's here so we can indicate that it should not be differentiated
function neighbourfinder(at)
   Rs = []
   nlist = neighbourlist(at, cutoff(EMT()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      tmpRs=ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      append!(Rs,[tmpRs])
   end 
   return Rs
end

#Random copper fcc atoms and EMT() energies
function genData(Ntrain)
   atoms = []
   tmpEnergy = []
   tmpForces = []
   for i in 1:Ntrain
      at = bulk(:Cu, cubic=true) * 3
      rattle!(at,0.6) 
      #push!(atoms,at)
      push!(atoms, neighbourfinder(at)[1])
      push!(tmpEnergy, energy(EMT(),at))
      push!(tmpForces, forces(EMT(),at))
   end
   prop = [ (E = E, F = F) for (E, F) in zip(tmpEnergy, tmpForces) ]
   return(atoms,prop)
end

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

mutable struct linear_ACE{TW, TM}
   weight::TW
   m::TM 
end

function linear_ACE(maxdeg, ord, Nprop)
   #building the basis
   Bsel = SimpleSparseBasis(ord, maxdeg)
   B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg)
   φ = ACE.Invariant()
   basis = SymmetricBasis(φ, B1p, O3(), Bsel)

   #create a multiple property model
   W = rand(Nprop, length(basis))
   LM = LinearACEModel(basis, matrix2svector(W), evaluator = :standard) 
   return linear_ACE(W, LM)
end

@functor linear_ACE #so that flux can find the parameters
 
#forward pass
(y::linear_ACE)(cfg) = _eval_linear_ACE(y.weight, y.m, cfg)


#energy evaluation
function _eval_linear_ACE(W, M, cfg)
   set_params!(M, matrix2svector(W))
   E = getproperty.(evaluate(M ,cfg), :val)
   return E
end

function _adj(dp, W, M, cfg)
   set_params!(M, matrix2svector(W))
   _, dW = Zygote.pullback(M_ -> evaluate(M_ ,cfg), M)
   _, dcgf = Zygote.pullback(X_ -> evaluate(M ,X_), cfg)
   return(NoTangent(), svector2matrix(dW(dp)[1]), NoTangent(), dcgf(dp)[1])
end

function ChainRules.rrule(::typeof(_eval_linear_ACE), W, M, cfg)
   E = _eval_linear_ACE(W, M, cfg)
   return E, dp -> _adj(dp)
end

function ChainRules.rrule(::typeof(_adj), dp, W, M, cfg)
   function _second_adj(dq)
      Zygote.pullback(_adj(dp, W, M, cfg), M)

model = Chain(linear_ACE(6, 4, 2), Dense(2, 3, σ), Dense(3, 1), sum)
energ(x) = model(x)
force(x) = Zygote.gradient( x -> energ(x), x )

loss(x,y) = abs2(energ(x) - y.E) + sum(sum([force(x)[1][i].rr - y.F[i] for i in 1:5]))

# optimize
opt = Descent()
Xtrain, Ytrain = genData(2)
data = zip(Xtrain, Ytrain)
total_loss = () -> sum(loss(x, y) for (x, y) in data)
@show total_loss()
p = params(model)

_, dW = Zygote.pullback(M_ -> evaluate(M_ ,cfg), M)
Zygote.gradient(()->dW([1,2]), )

Zygote.gradient(()->loss(Xtrain[1], Ytrain[1]), p)

Flux.train!(loss, p, data, opt)
@show total_loss()