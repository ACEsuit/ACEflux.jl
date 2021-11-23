using IPFitting, ACE, ACEgnns, Flux, Zygote, Statistics, StatsBase
using ThreadTools

all_Si = IPFitting.Data.read_xyz("G:/My Drive/documents/UBC/Julia Codes/silicon/Si.xyz", energy_key="dft_energy", force_key="dft_force")
#all_Si = IPFitting.Data.read_xyz("/Users/ortner/Dropbox/PIBmat/Si/silicon_database_gp_iter6_sparse9k.xml.xyz", energy_key="dft_energy")
#all_Si = IPFitting.Data.read_xyz("/zfs/users/aross88/aross88/silicon/Si.xyz", energy_key="dft_energy", force_key="dft_force")
data = filter(at -> configtype(at) == "dia", all_Si) #get diamon configurations

#simply pull out all the configurations
Y = []  
X = []
for i in 1:length(data)
   push!(Y, [ data[i].D["E"][1], ACEgnns.matrix2svector(reshape(data[i].D["F"], (3, Int(length(data[i].D["F"])/3)))) ])
   push!(X, data[i].at)
end

FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10

fs_model = Chain(Linear_ACE(3, 4, 2), GenLayer(FS), sum)
#gen_model = Chain(Linear_ACE(3, 4, 2), Dense(2, 6), Dense(6, 10), Dense(10, 1), sum)
fs_pot = FluxPotential(fs_model, 6.0) #model, cutoff
#gen_pot = FluxPotential(gen_model, 6.0) 

sqr(x) = x.^2
fs_loss(at, EF) =  Flux.Losses.mse(FluxEnergy(fs_pot, at), EF[1]) + sum(sum(sqr.(FluxForces(fs_pot, at) - EF[2]))) #could maybe use MSE for forces as well
#gen_loss(at, EF) =  Flux.Losses.mse(FluxEnergy(gen_pot, at), EF[1]) + sum(sum(sqr.(FluxForces(gen_pot, at) - EF[2]))) #could maybe use MSE for forces as well

# Zygote.gradient(()->loss(X[1], Y[1]), params(model))

opt = ADAM(0.1)

(θ_fs, loss_fs, grd_fs) = opt_Flux(fs_loss, params(fs_model), X, Y, opt, 10, 1, "test_si", b=1, lograte=2)
#(θ_gen, loss_gen, grd_gen) = opt_Flux(gen_loss, params(gen_model), X, Y, opt, 10, b=5)

  #batching per configuration type
  #two particle basis
  #inner and outter cutoffs (also p and p0)


# using BSON: @save
  
# @save "mymodel.bson" model