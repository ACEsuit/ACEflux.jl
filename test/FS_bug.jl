using IPFitting, ACE, ACEgnns, Flux, Zygote, JuLIP

at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6) 

gen_model = Chain(Linear_ACE(3, 4, 2), Dense(2, 3, σ), Dense(3, 1), sum)

FluxModel = FluxPotential(gen_model, 5.19) #model, cutoff
g = Zygote.gradient(()->sum(sum(FluxForces(FluxModel,at))), params(gen_model))
g[params(gen_model)[1]]


ϵ = 1/10
FS(ϕ) = sum(ϕ)#ϕ[1] + sqrt(ϕ[2] + ϵ^2) - ϵ
FS_model = Chain(Linear_ACE(3, 4, 2, σ=FS))

FluxModel = FluxPotential(FS_model, 5.19) #model, cutoff
g = Zygote.gradient(()->sum(sum(FluxForces(FluxModel,at))), params(FS_model))
g[params(FS_model)[1]]


ϵ = 1/10
FS(ϕ) = ϕ[1] + sqrt(ϕ[2] + ϵ^2) - ϵ
FS_model = Chain(Linear_ACE(3, 4, 2, σ=FS), Dense(1, 1), sum)

FluxModel = FluxPotential(FS_model, 5.19) #model, cutoff
g = Zygote.gradient(()->sum(sum(FluxForces(FluxModel,at))), params(FS_model))
g[params(FS_model)[1]]

ϵ = 1/10
FS(ϕ) = ϕ[1] + sqrt(ϕ[2] + ϵ^2) - ϵ
FS_model = Chain(Linear_ACE(3, 4, 2), Dense(2, 2), FS)

FluxModel = FluxPotential(FS_model, 5.19) #model, cutoff
g = Zygote.gradient(()->sum(sum(FluxForces(FluxModel,at))), params(FS_model))
g[params(FS_model)[1]]