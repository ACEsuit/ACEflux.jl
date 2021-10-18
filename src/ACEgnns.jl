module ACEgnns

export LsqDBflux, Linear_ACE, FluxPotential, FluxEnergy, FluxForces, neighbourfinder, svector2matrix
include("linearLayer.jl")
include("JuLIP_wrapper.jl")
include("calculator.jl")
include("IPfitting_functions.jl")

end
