using Flux
using JuLIP
using ACE
using StaticArrays
using Statistics
using JuLIP: EMT
using Flux: @functor, @epochs
using ACE: State, NaiveTotalDegree, SymmetricBasis, evaluate, LinearACEModel, set_params!, grad_params
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent 
using LinearAlgebra: length
using Plots


mutable struct ACELayer{TW}
   weights::TW
end

@functor ACELayer #for Flux to be able to update the weights

#evaluation of the ACE layer / Forward pass
function (l::ACELayer)(at)
   Rs = []
   nlist = neighbourlist(at, cutoff(EMT()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      tmpRs=ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      append!(Rs,[tmpRs])
   end 

   set_params!(LM, l.weights)
   E = sum([getproperty.(evaluate(LM ,r), :val) for r in Rs])
   u = σ(E)
   u
end

function ChainRules.rrule(l::ACELayer, at)
   Rs = []
   nlist = neighbourlist(at, cutoff(EMT()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      tmpRs=ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      append!(Rs,[tmpRs])
   end 

   set_params!(LM, l.weights)
   E = sum([getproperty.(evaluate(LM ,r), :val) for r in Rs])

   function adj(dp)
      gσt = Flux.gradient(σ, E)[1]
      gσ = SVector{length(gσt)}(gσt)
      gparams = sum([grad_params(LM ,r) for r in Rs])
      grad = zeros(SVector{length(gparams[1])},length(gparams))
      for i = 1:length(gparams)
         grad[i] = getproperty.(gparams[i], :val) .* gσ
      end
      return (dp * grad , NoTangent()) 
   end
   return σ(E), adj
end

function data(Ntrain)
   train = []
   Etrain = []
   for i in 1:Ntrain
      at = bulk(:Cu, cubic=true) * 3
      rattle!(at,0.6)
      push!(train,at)
      push!(Etrain, energy(EMT(),at))
   end
   return(train,Etrain)
end

X_train,y_train = data(10)
X_test,y_test = data(2)
train_data = zip(X_train,y_train)

Nprop = 2
σ = ϕ -> ϕ[1] + sqrt((1/10)^2 + abs(ϕ[2])) - 1/10

maxdeg = 6
ord = 4

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = NaiveTotalDegree())
pibasis = PIBasis(B1p, ord, maxdeg; property = ACE.Invariant())
basis = SymmetricBasis(pibasis, ACE.Invariant());   

initW = rand(SVector{Nprop,Float64}, length(basis))

LM = LinearACEModel(basis, initW, evaluator = :standard) 

model = ACELayer(initW)

model(X_train[1])

loss(x, y) = Flux.Losses.mse(model(x), y)

opt = ADAM(0.001)

# and a callback to see training progress
evalcb() = mean(loss.(X_test, y_test))

testL = []
push!(testL, evalcb())
for i in 1:100
   g = Flux.gradient(model -> model(X_train[1]), model)[1]
   Flux.update!(opt,model.weights,g)
   push!(testL, evalcb())
end

plot(testL)