module ACEgnns

export opt_Flux, GenLayer, Linear_ACE, FluxPotential, gsum, FluxEnergy, FluxForces, neighbourfinder, svector2matrix

include("JuLIP_wrapper.jl")
include("layers.jl")
include("calculator.jl")
include("training.jl")

end
