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

dB = LsqDBflux("", FluxModel, Ti_data)

