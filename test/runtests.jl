using ACEflux
using Test

@testset "ACEflux.jl" begin
    include(test_fsModel_der.jl)
    include(test_genModel_der.jl)
end
