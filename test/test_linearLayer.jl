using ACEgnns, JuLIP, ACE, Flux, Zygote
using ACE: State
using LinearAlgebra
using Printf
using Test, ACE.Testing

#forward difference test, takes a function(model, x), the model, the derivative
# the input x and the parameters as given by Flux. Only checks the parameters
#of the first layer, and will probably only work for matrices.
function fdtest(ffunc,fmodel, dE, x, p; verbose=true)
   errors = Float64[]
   E = ffunc(fmodel,x)
   verbose && @printf("---------|----------- \n")
   verbose && @printf("    h    | error \n")
   verbose && @printf("---------|----------- \n")
   for k = 2:11
      h = 0.1^k
      dEh = copy(dE[p[1]])
      for n in 1:length(dE[p[1]])
         fmodel.model[1].weight[n] += h
         dEh[n] = (ffunc(fmodel,x) - E) / h
         fmodel.model[1].weight[n] -= h
      end
      push!(errors, norm(dE[p[1]] - dEh, Inf))
      verbose && @printf(" %1.1e | %4.2e  \n", h, errors[end])
   end
   verbose && @printf("---------|----------- \n")
   if minimum(errors) <= 1e-3 * maximum(errors)
      verbose && println("passed")
      return true
   else
      verbose && println("finite-difference failed")
      return false
   end
end

model = Chain(Linear_ACE(6, 4, 2), Dense(2, 3, σ), Dense(3, 1), sum)

FluxModel = FluxPotential(model, 5.0) #model, cutoff

# we only check the derivatives of the parameters in the linear ace layer
# we do finite difference on the whole function, but only compare ACE parameters
@info "dEnergy, dE/dP"

at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6) 

for ntest in 1:5
   s = size(FluxModel.model[1].weight)
   FluxModel.model[1].weight = rand(s[1],s[2])
   fluxmodel(x) = FluxEnergy(FluxModel, x)
   p = params(model)
   dE = Zygote.gradient(()->FluxEnergy(FluxModel, at), p)
   print_tf(@test fdtest(FluxEnergy , FluxModel, dE, at, p, verbose=false))
end

#we initialize with whatever random weights where left from above
@info "Forces, dE/dX"

#Way more complicated than I thought initially. We need to do finite difference
#on configurations, which means altering by small ammounts the R for every at.
#I don't know how to do this other than a small rattle! 

# for ntest in 1:5
#    at = bulk(:Cu, cubic=true) * 3
#    rattle!(at,0.6) 
#    fluxmodel(x) = FluxEnergy(FluxModel, x)
#    dE = FluxForces(FluxModel, at)
#    print_tf(@test fdtest(FluxEnergy , FluxModel, dE, at, p, verbose=false))
# end

@info "dForces, dE^2/dXdP"




# #Random copper fcc atoms and EMT() energies
# function genData(Ntrain)
#    atoms = []
#    tmpEnergy = []
#    tmpForces = []
#    for i in 1:Ntrain
#       at = bulk(:Cu, cubic=true) * 3
#       rattle!(at,0.6) 
#       #push!(atoms,at)
#       push!(atoms, neighbourfinder(at)[1])
#       push!(tmpEnergy, energy(EMT(),at))
#       push!(tmpForces, forces(EMT(),at))
#    end
#    prop = [ (E = E, F = F) for (E, F) in zip(tmpEnergy, tmpForces) ]
#    return(atoms,prop)
# end

# Xtrain,Ytrain = genData(3)
# data = zip(Xtrain, Ytrain)

# function fdtest()

# en(x) = FluxEnergy(FluxModel, x)
# for x in Xtrain

#    Zygote.gradient(()->en(x), p)

# loss(x,y) = abs2(FluxEnergy(FluxModel, x) - y.E)# + sum(sum(forces(FluxModel, x) - y.F))
# lossTest(x,y) = abs2(energy(FluxModel, x) - y.E)# + sum(sum(forces(FluxModel, x) - y.F))

# opt = Descent()

# total_loss = () -> sum(loss(x, y) for (x, y) in data)
# total_loss_test = () -> sum(lossTest(x, y) for (x, y) in data)

# @show total_loss()
# @show total_loss_test()

# #TODO error seems to be in += for zygote in forces
# Flux.train!(loss, p, data, opt)

# @show total_loss()