using ACEgnns, JuLIP, ACE, Flux, Zygote

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
data = zip(Xtrain, Ytrain)


model = Chain(Linear_ACE(6, 4, 2), Dense(2, 3, σ), Dense(3, 1), sum)

FluxModel = FluxPotential(model, 5.0) #model, cutoff

loss(x,y) = abs2(energy(FluxModel, x) - y.E)# + sum(sum(forces(FluxModel, x) - y.F))

opt = Descent()

p = params(model)

total_loss = () -> sum(loss(x, y) for (x, y) in data)
@show total_loss()

#TODO error seems to be in += for zygote in forces
Flux.train!(loss, p, data, opt)

@show total_loss()