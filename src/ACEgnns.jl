module ACEgnns

export Linear_ACE, FluxPotential, FluxEnergy, FluxForces, neighbourfinder, svector2matrix
include("linearLayer.jl")
include("JuLIP_wrapper.jl")
include("calculator.jl")

end
