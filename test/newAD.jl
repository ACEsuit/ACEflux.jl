using Flux, ForwardDiff, Zygote, JuLIP, StaticArrays
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent
using Flux: @functor

using ACE
using ACE: State, NaiveTotalDegree, SymmetricBasis, evaluate, LinearACEModel, set_params!, grad_params, grad_config, grad_params_config

# # ------------------------------------------------------------------------
# #    Sample EMT data energies
# # ------------------------------------------------------------------------

#Random copper fcc atoms and EMT() energies
function genData(Ntrain)
   atoms = []
   tmpEnergy = []
   tmpForces = []
   for i in 1:Ntrain
      at = bulk(:Cu, cubic=true) * 3
      rattle!(at,0.6)
      push!(atoms,at)
      push!(tmpEnergy, energy(EMT(),at))
      push!(tmpForces, forces(EMT(),at))
   end
   prop = [ (E = E, F = F) for (E, F) in zip(tmpEnergy, tmpForces) ]
   return(atoms,prop)
end

Xtrain,Ytrain = genData(3)

# # ------------------------------------------------------------------------
# #    Layer definition
# # ------------------------------------------------------------------------

mutable struct Energy_ACE{TW, TM}
   weight::TW
   m::TM 
end

function Energy_ACE(maxdeg, ord, Nprop)
   #building the basis
   B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = NaiveTotalDegree())
   pibasis = PIBasis(B1p, ord, maxdeg; property = ACE.Invariant())
   basis = SymmetricBasis(pibasis, ACE.Invariant());   

   #create a multiple property model
   W = rand(Nprop, length(basis))  #matrix for Flux 
   Wsv =  [SVector{size(W)[1]}(W[:,i]) for i in 1:size(W)[2]]  #SVector for ACE
   LM = LinearACEModel(basis, Wsv, evaluator = :standard) 
   return Energy_ACE(W,LM)
end

@functor Energy_ACE #so that flux can find the parameters
 
#forward pass
(y::Energy_ACE)(at) = _eval_energy_ACE(y.weight, y.m, at)

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

#a hack to avoid derviating neighbourfinder, this will be done differently 
function ChainRules.rrule(::typeof(neighbourfinder), at)
   return neighbourfinder(at), dn -> dn
end

#energy evaluation
function _eval_energy_ACE(Wt, M, at)
   W = [SVector{size(Wt)[1]}(Wt[:,i]) for i in 1:size(Wt)[2]] #SVector conversion
   set_params!(M, W)
   Rs = neighbourfinder(at)
   E = sum([getproperty.(evaluate(M ,r), :val) for r in Rs])
   return E
end

#adj is outside chainrule so we can define a chainrule for it
function adj(dp, Wt, M, Rs)

   W = [SVector{size(Wt)[1]}(Wt[:,i]) for i in 1:size(Wt)[2]]  #SVector conversion
   set_params!(M, W) 

   #Params derivative
   gparams = sum([grad_params(M ,r) for r in Rs]) #gradient is returned as SVector for now
   #make them numbers rather than invariants, this should be different
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
   gconfig = [sum(grad_config(M,r)) for r in Rs]
  
   return (NoTangent(), gradMatrix, NoTangent(), gconfig)
end

function ChainRules.rrule(::typeof(_eval_energy_ACE), Wt, M, at)
   Rs = neighbourfinder(at)
   E = _eval_energy_ACE(Wt, M, at)
   return E, dp -> adj(dp, Wt, M, Rs)
end

#it actually returns all the gradient combinations. So der of parameters twice, of model
#twice of model and parameters, etc. However, we can just pass NoTangent() when we don't need them
function ChainRules.rrule(::typeof(adj), dp, W, M, Rs)

   #this is not fast, but we can change this function to return or calculate
   #exactly what we want, so theoretically could be optimized.
   #all the sum() work because the loss only sums, need to think
   #exactly about how to pull the matrix together, but ideally we wouldn't even compute the matrix.
   function secondAdj(dq)
      #sum over all r in Rs
      cumsum = zeros(length(M.c[1]), length(M.c))
      for (ri,r) in enumerate(Rs)
         Temp = grad_params_config(M, r)
         #sum over config length
         Temp = [sum([Temp[prop][:,i] for i in 1:length(Temp[prop][1,:])]) for prop in 1:length(M.c[1])]
         for prop in 1:length(M.c[1])
            #multiply by dq and sum over x,y,z
            cumsum[prop, :] += [sum(dq[4][ri].rr .* Temp[prop][i].rr) for i in 1:length(Temp[prop])]
         end
      end

      return(NoTangent(), NoTangent(), cumsum, NoTangent(), NoTangent())
   end
   return(adj(dp, W, M, Rs), secondAdj)
end
# # ------------------------------------------------------------------------
# #    create a flux model with a few layers
# # ------------------------------------------------------------------------


model = Chain(Energy_ACE(6, 4, 2), Dense(2, 3, σ), Dense(3, 1), sum)
EVAL(model, x) = model(x)
GRAD(model, x) = Zygote.gradient( x -> EVAL(model, x), x )[1]

## define the loss
#done this way to substract SVector and state objects
function loss(x,y)
   Ftemp = GRAD(model, x)
   return(abs2(EVAL(model, x) - y.E) + sum([sum(Ftemp[i].rr - y.F[i]) for i in 1:length(y.F)]))
end

## optimize
opt = Descent()
data = zip(Xtrain, Ytrain)
total_loss = () -> sum(loss(x, y) for (x, y) in data)
@show total_loss()
p = params(model)

Flux.train!(loss, p, data, opt)
@show total_loss()

