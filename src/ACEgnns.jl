module ACEgnns

export opt_Flux, GenLayer, Linear_ACE, FluxPotential, FluxEnergy, FluxForces, neighbourfinder, svector2matrix
include("layers.jl")
include("calculator.jl")
include("training.jl")

end
