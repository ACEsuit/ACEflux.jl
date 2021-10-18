using IPFitting, ACE, ACEgnns, Flux, Zygote

data = IPFitting.Data.read_xyz("G:/My Drive/documents/UBC/Julia Codes/titanium/Cas_Ti_ZopeTiAl2003.xyz", energy_key="energy", force_key="force");

@show 1

Ti_1 = filter(at -> configtype(at) == "FLD_Ti_bcc", data)
Ti_2 = filter(at -> configtype(at) == "FLD_Ti_hcp", data)
Ti_3 = filter(at -> configtype(at) == "PH_bcc", data)
Ti_4 = filter(at -> configtype(at) == "PH_hcp", data)
Ti_5 = filter(at -> configtype(at) == "Ti_T3700", data)
Ti_6 = filter(at -> configtype(at) == "Ti_T4200", data)
Ti_data = vcat(Ti_1, Ti_2, Ti_3, Ti_4, Ti_5, Ti_6)

# targets = ACEgnns._alloc_EF_vector(Ti_data)

# while i < length(targets)
#    Ti_data[i].Dvec_obs_flux


@show 2

model = Chain(Linear_ACE(3, 2, 2), Dense(2, 3, σ), Dense(3, 1), sum)
FluxModel = FluxPotential(model, 5.19) #model, cutoff

@show 3

dB(FluxModel, Ti_data) = sum(LsqDBflux("", FluxModel, Ti_data).Ψ) #-dB.Y
p = params(model)
Zygote.gradient(()->dB(FluxModel, Ti_data), p)



loss = sum(dB.Ψ - dB.Y)/length(dB.Ψ)

loss(x, y) = Flux.Losses.mse(model(x), y)

evalcb_verbose() = @show(mean(loss.(test_input, test_output)))
evalcb_quiet() = return nothing
evalcb = verbose ? evalcb_verbose : evalcb_quiet
evalcb()

@epochs num_epochs Flux.train!(
   loss,
   params(model),
   train_data,
   opt,
   cb = Flux.throttle(evalcb, 5),
  )

# params(dB)

# Zygote.gradient(()-loss())

# #Ti_data[indx].D["E"][1]
