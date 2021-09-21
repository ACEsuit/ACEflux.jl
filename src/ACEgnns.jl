module ACEgnns

export Linear_ACE, FluxPotential, FluxEnergy, FluxForces
include("linearLayer.jl")
include("JuLIP_wrapper.jl")
include("calculator.jl")

end
