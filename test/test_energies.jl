using ACEgnns, JuLIP, ACE, Flux, Zygote
using ACE: State
using LinearAlgebra
using Printf
using Test, ACE.Testing
using ACEbase

model = Chain(Linear_ACE(6, 4, 2), Dense(2, 3, σ), Dense(3, 1), sum)

FluxModel = FluxPotential(model, 5.0) #model, cutoff

# we only check the derivatives of the parameters in the linear ace layer
# we do finite difference on the whole function, but only compare ACE parameters
@info "dEnergy, dE/dP"

at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6) 

s = size(FluxModel.model[1].weight)

# function F(c)
#    FluxModel.model[1].weight = reshape(c, s[1], s[2])
#    return FluxEnergy(FluxModel, at)
# end

# function dF(c)
#    FluxModel.model[1].weight = reshape(c, s[1], s[2])
#    p = params(model)
#    dE = Zygote.gradient(()->FluxEnergy(FluxModel, at), p)
#    return(svector2matrix(dE[p[1]]))
# end

# for _ in 1:5
#    c = rand(s[1]*s[2])
#    println(@test ACEbase.Testing.fdtest(F, dF, c, verbose=true))
# end
# println()


# @info("Check the AD Forces for an FS-like model")
# Us = randn(SVector{3, Float64}, length(cfg))
# dF = t -> sum( dot(u, g.rr) for (u,g) in zip(Us, Zygote.gradient(F, cfg)[1]) )
# dF(0.0)

# ACEbase.Testing.fdtest(F, dF, 0.0, verbose=true)



r = neighbourfinder(at)[1]

# frcs(model, r) = sum(sum(Zygote.gradient(model, r)[1]).rr)

# a,b = Zygote.pullback(()->frcs(model,r), params(model))

# a,b = Zygote.gradient(()->FluxEnergy(FluxModel,at), params(model))


ffrcs(FluxModel, at) = 0.77 .* sum(0.7 .* sum(0.77 .* FluxForces(FluxModel, at)))^2

Zygote.gradient(()->ffrcs(FluxModel,at), params(model))




@info "dloss, d{E+sum(F)}/dP"

loss(FluxModel, at) = FluxEnergy(FluxModel, at) + sum(sum(FluxForces(FluxModel, at)))

function F2(c)
   FluxModel.model[1].weight = reshape(c, s[1], s[2])
   return loss(FluxModel, at)
end

function dF2(c)
   FluxModel.model[1].weight = reshape(c, s[1], s[2])
   p = params(model)
   dE = Zygote.gradient(()->loss(FluxModel, at), p)
   return(svector2matrix(dE[p[1]]))
end

for _ in 1:5
   c = rand(s[1]*s[2])
   println(@test ACEbase.Testing.fdtest(F2, dF2, c, verbose=true))
end
println()