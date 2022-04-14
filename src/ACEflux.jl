module ACEflux
export read_pymatgen, opt_Flux, GenLayer, Linear_ACE, FluxPotential, gsum, svector2matrix, matrix2svector, neighbours_R, make_fg_multi

include("calculator.jl")
include("layers.jl")
include("utils.jl")

end
