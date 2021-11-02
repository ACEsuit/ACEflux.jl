using ACEgnns, JuLIP, ACE, Flux, Zygote
using ACE: State
using LinearAlgebra
using Printf
using Test, ACE.Testing
using ACEbase

FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10

model = Chain(Linear_ACE(3, 4, 2), GenLayer(FS), sum)
FluxModel = FluxPotential(model, 6.0) 

##

# # we only check the derivatives of the parameters in the linear ace layer
# # we do finite difference on the whole function, but only compare ACE parameters

@info "dEnergy, dE/dP"

at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6) 

s = size(FluxModel.model[1].weight)

function F(c)
   FluxModel.model[1].weight = reshape(c, s[1], s[2])
   return FluxEnergy(FluxModel, at)
end

function dF(c)
   FluxModel.model[1].weight = reshape(c, s[1], s[2])
   p = params(model)
   dE = Zygote.gradient(()->FluxEnergy(FluxModel, at), p)
   return(dE[p[1]])
end

for _ in 1:5
   c = rand(s[1]*s[2])
   println(@test ACEbase.Testing.fdtest(F, dF, c, verbose=true))
end
println()

##

@info "dForces, d{sum(F)}/dP"

sqr(x) = x.^2
ffrcs(FluxModel, at) = sum(sum(sqr.(FluxForces(FluxModel, at))))

function F(c)
   FluxModel.model[1].weight = reshape(c, s[1], s[2])
   return ffrcs(FluxModel, at)
end

function dF(c)
   FluxModel.model[1].weight = reshape(c, s[1], s[2])
   p = params(model)
   dF = Zygote.gradient(() -> ffrcs(FluxModel, at), p)
   return(dF[p[1]])
end

for _ in 1:5
   c = rand(s[1]*s[2])
   println(@test ACEbase.Testing.fdtest(F, dF, c, verbose=true))
end
println()

##

@info "dloss, d{E+sum(F)}/dP"

loss(FluxModel, at) = FluxEnergy(FluxModel, at) + sum(sum(sqr.(FluxForces(FluxModel, at))))

function F2(c)
   FluxModel.model[1].weight = reshape(c, s[1], s[2])
   return loss(FluxModel, at)
end

function dF2(c)
   FluxModel.model[1].weight = reshape(c, s[1], s[2])
   p = params(model)
   dL = Zygote.gradient(()->loss(FluxModel, at), p)
   return(dL[p[1]])
end

for _ in 1:5
   c = rand(s[1]*s[2])
   println(@test ACEbase.Testing.fdtest(F2, dF2, c, verbose=true))
end
println()


