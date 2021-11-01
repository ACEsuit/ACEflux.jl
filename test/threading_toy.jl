
using Base.Threads
using ThreadTools
using Zygote
import ChainRules
import ChainRules: rrule, NoTangent
using ForwardDiff

using ThreadTools
using Zygote
using Flux
using Flux: @functor

mutable struct Model{T}
   p::T
end

@functor Model

(y::Model)(x) = (x - y.p[1]) * y.p[2] 

m = Model(rand(2))

x = rand(100)
m(x[1])

ourloss(x) = sum(abs2, tmap(m, x))
ourloss(x)
g = Zygote.gradient(()->ourloss(x), params(m))
g[params(m)[1]]


##

# { configs } -> [ ri ] -> norm( [w_i r_i] )^2

function fun_serial(x)
   return [ 1 / (x[i] + 1)^2 for i = 1:length(x) ]
end

loss_serial(x) = sum(abs2, fun_serial(x))

##

x = rand(100)
loss_serial(x)
Zygote.gradient(loss_serial, x)[1]

##

ourmap(f, x) = tmap(f, x)

function rrule(::typeof(ourmap), f, x)
   val = ourmap(f, x)

   function _pb(dp)
      @assert length(dp) == length(val) == length(x)
      df(dpi, xi) = dpi * ForwardDiff.derivative(f, xi)
      g = tmap(df, dp, x)
      return NoTangent(), NoTangent(), g  
   end

   return val, _pb
end   

ourloss(x) = sum(abs2, ourmap(xi -> 1/(xi+1)^2, x))

ourloss(x)

Zygote.gradient(ourloss, x)[1] ≈ Zygote.gradient(loss_serial, x)[1]

##

using Flux
using Flux: @functor

mutable struct Model{T}
   p::T
end

@functor Model

(y::Model)(x) = (x - y.p[1]) * y.p[2] 


m = Model(rand(2))

m(x[1])

ourloss(x) = sum(abs2, tmap(m, x))
ourloss(x)
g = Zygote.gradient(()->ourloss(x), params(m))
g[params(m)[1]]
















using IPFitting, ACE, ACEgnns, Flux, Zygote, JuLIP

at = bulk(:Cu, cubic=true) * 3
rattle!(at,0.6) 

model = Chain(Linear_ACE(3, 4, 2), Dense(2, 1), sum)#, Dense(6, 10), Dense(10, 1), sum)
FluxModel = FluxPotential(model, 6.0) #model, cutoff
sqr(x) = x.^2

FluxEnergy(FluxModel, at)
FluxEnergyOld(FluxModel, at)

loss(at, EF) =  Flux.Losses.mse(FluxEnergy(FluxModel, at), EF[1])# + sum(sum(sqr.(FluxForces(FluxModel, at) - EF[2]))) #could maybe use MSE for forces as well
loss(at, [5.0])

g = Zygote.gradient(()->loss(at,[5.0]), params(model))

@show g[params(model)[1]]






struct pot{M, U}
   model::M
   useless::U
end

(y::pot)(x) = y.model(x)