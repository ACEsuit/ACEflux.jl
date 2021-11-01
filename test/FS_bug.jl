using IPFitting, ACE, ACEgnns, Flux, Zygote, JuLIP
using Flux: @functor

at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6) 

struct FS end

(::FS)(ϕ) = [ϕ[1] + sqrt(ϕ[2]^2 + 1/100) - 1/10] #3
@functor FS

fs(x) = x[1] + sqrt(x[2]^2 + 1/100) - 1/10

FS_model = Chain(Linear_ACE(3, 4, 2), Dense(2,1), sigmoid) 
#FS_model = Chain(Linear_ACE(3, 4, 2), Dense(2,2), fs, sum) 


FS_model(neighbourfinder(at)[1])
FluxModel = FluxPotential(FS_model, 5.19)
g = Zygote.gradient(()->sum(sum(FluxForces(FluxModel,at))), params(FS_model))











FluxModel = FluxPotential(FS_model, 5.19) #model, cutoff
g[params(FS_model)[1]]




gen_model = Chain(Linear_ACE(3, 4, 2), Dense(2, 3, σ), Dense(3, 1), sum)

FluxModel = FluxPotential(gen_model, 5.19) #model, cutoff
g = Zygote.gradient(()->sum(sum(FluxForces(FluxModel,at))), params(gen_model))
g[params(gen_model)[1]]



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


FS(x) = x[1] + sqrt(abs(x[2]) + 1/100) - 1/10
model = Linear_ACE(3, 4, 2)
function FSmodel(θ, X) #theta is the ace params and X is a configuration
   model.weight = θ
   return FS(model(X))
end

c = rand(2,28)
FSmodel(c, a[1])

FluxModel = FluxPotential(FSmodel, 5.19) #model, cutoff
g = Zygote.gradient(()->sum(sum(FluxForces(FluxModel,at))), params(gen_model))
g[params(gen_model)[1]]

