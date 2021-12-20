using JuLIP, ACEgnns, Zygote, Flux

at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6) 
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10
model = Chain(Linear_ACE(2, 3, 2), GenLayer(FS), sum);
pot = FluxPotential(model, 6.0) ;

p = Flux.params(model)
g = Zygote.gradient(()->energy(pot, at), p)

q = deepcopy(Flux.params(model))
g = Zygote.gradient(()->energy(pot, at), q)

p = Flux.params(model)
g1 = Zygote.gradient(()->sum(sum(forces(pot, at))), p)

q = deepcopy(Flux.params(model))
g2 = Zygote.gradient(()->sum(sum(forces(pot, at))), q)