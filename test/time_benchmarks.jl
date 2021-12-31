using ACEflux, ACE, Flux, JuLIP, StaticArrays, BenchmarkTools
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
@btime gradient(()->lE(at,Y), p);


# [ Info: ord = 3, maxdeg = 10, Nprop = 2
# [ Info: configuration of length 100
# [ Info: linear ACE build
#   219.254 ms (3018481 allocations: 132.97 MiB)
# [ Info: linear ACE eval
#   861.200 μs (5629 allocations: 428.88 KiB)
# [ Info: 2 dense and FS eval
#   859.300 μs (5635 allocations: 429.41 KiB)
# [ Info: get params
#   2.322 μs (51 allocations: 2.39 KiB)
# [ Info: grad params model
#   74.169 ms (409631 allocations: 13.81 MiB)
# [ Info: grad config model
#   74.096 ms (409672 allocations: 13.80 MiB)
# [ Info: build a potential
#   82.803 ns (1 allocation: 64 bytes)
# [ Info: sanity check, eval pot
#   854.000 μs (5633 allocations: 429.33 KiB)
# [ Info: at of length 108
# [ Info: energy eval
#   87.344 ms (594941 allocations: 45.04 MiB)
# [ Info: forces eval
#   7.231 s (38767986 allocations: 1.23 GiB)
# [ Info: energy der
#   7.177 s (38757150 allocations: 1.23 GiB)
# [ Info: sum forces der
#   24.378 s (114890076 allocations: 3.96 GiB)
# [ Info: next evaluations are for 1 data point
# [ Info: full loss evaluation
#   6.755 s (39362931 allocations: 1.28 GiB)
# [ Info: only energy loss evaluation
#   97.300 ms (594942 allocations: 45.04 MiB)
# [ Info: grad full loss
#   72.671 s (153647541 allocations: 5.19 GiB)
# [ Info: grad only energy loss
#   13.089 s (38757331 allocations: 1.23 GiB)
