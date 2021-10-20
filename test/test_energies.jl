using ACEgnns, JuLIP, ACE, Flux, Zygote
using ACE: State
using LinearAlgebra
using Printf
using Test, ACE.Testing
using ACEbase

model = Chain(Linear_ACE(3, 2, 2), Dense(2, 3, σ), Dense(3, 1), sum)

FluxModel = FluxPotential(model, 5.0) #model, cutoff

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






# RRs = ACEgnns.Flux_neighbours_R(FluxModel, at)
# Ri = RRs[5]
# J = ACEgnns.Flux_neighbours_J(FluxModel, at)
# Ji = J[5]

# p = params(FluxModel.model)
# g = Flux.gradient(() -> FluxModel(Ri), p)

# length(Ri) == length(Ji)
# Ji

# ##

# f = FluxForces(FluxModel, at)
# U = randn(eltype(f), length(f))

# L = () -> sum( dot(u, f) for (u, f) in zip(U, FluxForces(FluxModel, at)) )
# p = params(FluxModel.model)
# g = Flux.gradient( L, p )

# ##


# # @info("Check the AD Forces for an FS-like model")
# # Us = randn(SVector{3, Float64}, length(cfg))
# # dF = t -> sum( dot(u, g.rr) for (u,g) in zip(Us, Zygote.gradient(F, cfg)[1]) )
# # dF(0.0)

# # ACEbase.Testing.fdtest(F, dF, 0.0, verbose=true)



# r = neighbourfinder(at)[1]

# #frcs(model, r) = sum(sum(Zygote.gradient(model, r)[1]).rr)

# # a,b = Zygote.pullback(()->frcs(model,r), params(model))

# #a,b = Zygote.gradient(()->FluxEnergy(FluxModel,at), params(model))


# # dptmp = [ACE.DState(rr=zeros(3)) for _ in 1:length(r)]

# # a,b = Zygote.pullback(()->Zygote.gradient(FluxModel, r)[1], params(model))
# #g = Zygote.gradient(c_ -> Zygote.gradient(c_, r)[1], FluxModel)

# # @show b(dptmp)

# #ffrcs(FluxModel, at) = 0.77 .* sum(0.7 .* sum(0.77 .* FluxForces(FluxModel, at))^2)
# sqr(x) = x.^2
# ffrcs(FluxModel, at) = sum(sum(sqr.(FluxForces(FluxModel, at))))

# # g = Zygote.gradient(()->ffrcs(at), params(model))
# #g = Zygote.gradient(x -> ffrcs(x, at), FluxModel)




# @info "dForces, d{sum(F)}/dP"

# function F(c)
#    FluxModel.model[1].weight = reshape(c, s[1], s[2])
#    return ffrcs(FluxModel, at)
# end

# function dF(c)
#    FluxModel.model[1].weight = reshape(c, s[1], s[2])
#    p = params(model)
#    dE = Zygote.gradient(x -> ffrcs(x, at), FluxModel)[1]
#    return(dE[p[1]])
# end

# for _ in 1:5
#    c = rand(s[1]*s[2])
#    println(@test ACEbase.Testing.fdtest(F, dF, c, verbose=true))
# end
# println()


# @info "dloss, d{E+sum(F)}/dP"

# loss(FluxModel, at) = FluxEnergy(FluxModel, at) + sum(sum(sqr.(FluxForces(FluxModel, at))))

# function F2(c)
#    FluxModel.model[1].weight = reshape(c, s[1], s[2])
#    return loss(FluxModel, at)
# end

# function dF2(c)
#    FluxModel.model[1].weight = reshape(c, s[1], s[2])
#    p = params(model)
#    #dE = Zygote.gradient(()->loss(FluxModel, at), p)
#    #dE = Zygote.gradient(x -> ffrcs(x, at), FluxModel)[1]
#    return(dE[p[1]])
# end

# for _ in 1:5
#    c = rand(s[1]*s[2])
#    println(@test ACEbase.Testing.fdtest(F2, dF2, c, verbose=true))
# end
# println()

