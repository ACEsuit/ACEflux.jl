using IPFitting, ACE, ACEgnns, Flux, Zygote, Statistics
using Flux: @epochs

# data = IPFitting.Data.read_xyz("G:/My Drive/documents/UBC/Julia Codes/silicon/Si.xyz", energy_key="dft_energy")
data = IPFitting.Data.read_xyz("/Users/ortner/Dropbox/PIBmat/Si/silicon_database_gp_iter6_sparse9k.xml.xyz", energy_key="dft_energy")

model = Chain(Linear_ACE(3, 2, 2), Dense(2, 3, σ), Dense(3, 1), sum)
FluxModel = FluxPotential(model, 5.19) #model, cutoff

#simply pull out all the configurations
Y = zeros(2475) #hardcoded for this example
X = []
for i in 1:2475
   Y[i] = data[i].D["E"][1]
   push!(X, data[i].at)
end
train_data = zip(X, Y) 

loss(x, y) = Flux.Losses.mse(FluxEnergy(FluxModel, x), y)

verbose = true
opt = ADAM(0.001)

evalcb_verbose() = @show(mean(loss.(X, Y)))
evalcb_quiet() = return nothing
evalcb = verbose ? evalcb_verbose : evalcb_quiet
evalcb()

num_epochs = 5

@epochs num_epochs Flux.train!(
   loss,
   params(model),
   train_data,
   opt,
   cb = Flux.throttle(evalcb, 5),
  )

