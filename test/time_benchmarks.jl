using ACEgnns, ACE, Flux, JuLIP, StaticArrays, BenchmarkTools
using Zygote: gradient

ord = 3
maxdeg = 10
Nprop =  2

@info("ord = 3, maxdeg = 10, Nprop = 2")
@info("configuration of length 100")

R_i = ACE.ACEConfig([ACE.State(rr=rand(SVector{3, Float64})) for _ = 1:100]);

@info("linear ACE build")
ϕ = @btime Linear_ACE(maxdeg, ord, Nprop)

@info("linear ACE eval")
@btime ϕ(R_i)

@info("2 dense and FS eval")
FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10
E_i = Chain(ϕ, Dense(2,7), Dense(7,2), GenLayer(FS), sum);
E_i(R_i)
@btime E_i(R_i)

@info("get params")
p = @btime Flux.params(E_i)

@info("grad params model")
@btime gradient(()->E_i(R_i), p);

@info("grad config model")
@btime gradient(E_i, R_i)

@info("build a potential")
pot = @btime FluxPotential(E_i, 6.0); #model, cutoff

@info("sanity check, eval pot")
@btime pot(R_i)


#######################
# Atoms
#######################

at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6);

@info("at of length " * string(length(at)))

@info("energy eval")
@btime energy(pot, at)

@info("forces eval")
@btime forces(pot, at)

@info("energy der")
@btime gradient(()->energy(pot, at), p)

@info("sum forces der")
@btime gradient(()->sum(sum(forces(pot, at))), p)

#######################
# Loss 
#######################

sqr(x) = x.^2 #to iterate twice
loss(at, EF) = Flux.Losses.mse(energy(pot, at), EF[1]) + sum(sum(sqr.(forces(pot, at) - EF[2])));
lE(at, EF) = Flux.Losses.mse(energy(pot, at), EF[1])


SW = StillingerWeber()

Y = [energy(SW,at), forces(SW,at)]

@info("next evaluations are for 1 data point")
@info("full loss evaluation")
@btime loss(at,Y)

@info("only energy loss evaluation")
@btime lE(at,Y)

@info("grad full loss")
@btime gradient(()->loss(at,Y), p)

@info("grad only energy loss")
@btime gradient(()->lE(at,Y), p)
