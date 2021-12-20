using Distributed

Distributed.addprocs(1)

@everywhere using Pkg
@everywhere Pkg.activate(pwd())
@everywhere Pkg.instantiate()

@everywhere using JuLIP, ACEgnns, Zygote, Flux, ACE, StaticArrays

@show nprocs()

@everywhere at = bulk(:Cu, cubic=true) * 3
@everywhere rattle!(at,0.6) 

@everywhere FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10

@everywhere model = Chain(Linear_ACE(2, 3, 2), GenLayer(FS), sum);
@everywhere pot = FluxPotential(model, 6.0) ;


@everywhere p = Flux.params(model)
f = @spawnat 2 Zygote.gradient(()->sum(sum(forces(pot, at))), p)
@show fetch(f)

f = @spawnat 2 Zygote.gradient(()->sum(sum(energy(pot, at))), Flux.params(model))
@show fetch(f)

@everywhere p = deepcopy(Flux.params(model))
f = @spawnat 2 Zygote.gradient(()->sum(sum(energy(pot, at))), p)
@show fetch(f)

@everywhere function test()
   at = bulk(:Cu, cubic=true) * 3
   rattle!(at,0.6) 
   FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10
   model = Chain(Linear_ACE(2, 3, 2), GenLayer(FS), sum);
   pot = FluxPotential(model, 6.0) ;


   p = Flux.params(model)
   g1 = Zygote.gradient(()->sum(sum(forces(pot, at))), p)


   g2 = Zygote.gradient(()->sum(sum(energy(pot, at))), Flux.params(model))

   q = deepcopy(Flux.params(model))
   g3 = Zygote.gradient(()->sum(sum(energy(pot, at))), q)

   return([g1,g2,g3])
end

f = @spawnat 2 test()
gs = fetch(f)

gs = test()

########################################
@everywhere p = Flux.params(model)
f = @spawnat 2 Zygote.gradient(()->energy(pot, at), p)
@show fetch(f)


f = @spawnat 2 Zygote.gradient(()->energy(pot, at), Flux.params(model))
@show fetch(f)


@everywhere p = deepcopy(Flux.params(model))
f = @spawnat 2 Zygote.gradient(()->energy(pot, at), p)
@show fetch(f)


#are we using the adjoints from ACE o naive evaluate
#proccs
#pair basis 
#regularization and weights on the loss
#size of basis