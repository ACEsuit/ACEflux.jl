using IPFitting, ACE, ACEgnns, Flux, Zygote, Statistics, StatsBase

#data = IPFitting.Data.read_xyz("G:/My Drive/documents/UBC/Julia Codes/silicon/Si.xyz", energy_key="dft_energy")
#data = IPFitting.Data.read_xyz("/Users/ortner/Dropbox/PIBmat/Si/silicon_database_gp_iter6_sparse9k.xml.xyz", energy_key="dft_energy")
all_Si = IPFitting.Data.read_xyz("/zfs/users/aross88/aross88/silicon/Si.xyz", energy_key="dft_energy", force_key="dft_force")
data = filter(at -> configtype(at) == "dia", all_Si) #get diamon configurations

#simply pull out all the configurations
Y = [] 
X = []
for i in 1:length(data)
   push!(Y, [ data[i].D["E"][1], ACEgnns.matrix2svector(reshape(data[i].D["F"], (3, Int(length(data[i].D["F"])/3)))) ])
   push!(X, data[i].at)
end

model = Chain(Linear_ACE(4, 10, 2), Dense(2, 6), Dense(6, 10), Dense(10, 1), sum)
FluxModel = FluxPotential(model, 6.0) #model, cutoff
sqr(x) = x.^2
loss(at, EF) =  Flux.Losses.mse(FluxEnergy(FluxModel, at), EF[1]) + sum(sum(sqr.(FluxForces(FluxModel, at) - EF[2]))) #could maybe use MSE for forces as well

verbose = true
opt = ADAM(0.001)

train_loss = []

function evalcb_verbose()
   push!(train_loss,mean(loss.(X, Y)))
   @show(last(train_loss))
end
evalcb_quiet() = return nothing
evalcb = verbose ? evalcb_verbose : evalcb_quiet
evalcb()

epochs = 100
b = 1 #batch size
for e in 1:epochs
   @show e
   indx = zeros(Int,b) 
   StatsBase.sample!(1:length(X), indx; replace=false)
   batch_data = zip(X[indx], Y[indx])  

   t = @elapsed Flux.train!(loss,params(model),batch_data,opt,cb = Flux.throttle(evalcb, 5),)
   @show (t*(epochs-e))/60
end


  #normalize the loss
  #batching per configuration type
  #threading
  #two particle basis
  #inner and outter cutoffs (also p and p0)

