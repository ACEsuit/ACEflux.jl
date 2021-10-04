using Zygote
using ChainRules
import ChainRulesCore: rrule, NoTangent

#sample function
function f(x,y)
   return sin.(x) .* exp.(-7 * y)
end

#the derivative of the function, outside of chainrule so we can
#define an rrule for it
function adj(dp, x, y)
   fx = cos.(x) .* exp.(-7 .* y) .* dp
   fy = -sin.(x) .* 7 .* exp.(-7 .* y) .* dp
   return(NoTangent(), fx, fy)
end

function ChainRules.rrule(::typeof(f), x, y)
   return f(x,y), dp -> adj(dp, x, y)
end

#second derivatice, we only care about df/dydx
#Here is where the problem is, we can multiply by dq which includes
#information of the outermost function, but we can't get the hessian of 
#NL (defined below). dp is already evaluated for the first derivative.
function ChainRules.rrule(::typeof(adj), dp, x, y)
   function secondAdj(dq)
      fyx = -cos.(x) .* 7 .* exp.(-7 .* y) .* dq[3]
      return (NoTangent(), NoTangent(), fyx, NoTangent())
   end
   return (adj(dp, x, y), secondAdj)
end

#random parameters
x = rand(10)
y = rand(10)

#using the functions
@info("sum(f)")
display(sum(abs2, f(x,y)))
@info("df/dx")
display(Zygote.gradient(z -> sum(abs2, f(z,y)), x)[1])
@info("df/dy")
display(Zygote.gradient(z -> sum(abs2, f(x,z)), y)[1])
@info("df/dydx")
df_dy = v -> sum(Zygote.gradient(z -> sum(abs2, f(v,z)), y)[1]) #extra sum to get gradient not jacobian
display(Zygote.gradient(df_dy, x))


#testing
using ACE, ACEbase, Test, ACE.Testing

NL(z) = [ 0.77^n * (1 + z[n]^2)^(1/n) for n = 1:length(z) ]

@info("df/dx test")
F = θ -> sum(abs2, NL(f(θ,y)))
dF = θ -> Zygote.gradient(F, θ)[1]
ACEbase.Testing.fdtest(F, dF, x; verbose=true)

@info("df/dy test")
F = θ -> sum(abs2, NL(f(x,θ)))
dF = θ -> Zygote.gradient(F, θ)[1]
ACEbase.Testing.fdtest(F, dF, y; verbose=true)

@info("df/dydx test")
df_dy = θ -> sum(Zygote.gradient(z -> sum(abs2, f(θ,z)), y)[1])
dF = θ -> Zygote.gradient(df_dy, θ)[1]
ACEbase.Testing.fdtest(df_dy, dF, x; verbose=true)



using Zygote
using ChainRules
import ChainRulesCore: rrule, NoTangent
using ACEbase.Testing: fdtest

## sample function
const N = 10
v(y) = [ exp(-n * y[1] - y[2]) for n = 1:N ]
Dv(y) = - [ (1:N) .* v(y) v(y) ]
f(x,y) = sin.(x) .* v(y)
f1(x,y) = sin.(x) .* v(y)

# the derivative of the function, outside of chainrule so we can define an rrule for it
function adj(dp, x, y)
   fx = cos.(x) .* v(y) .* dp
   fy = (Dv(y)') * (sin.(x) .* dp)
   return(NoTangent(), fx, fy)
end

function ChainRules.rrule(::typeof(f), x, y)
   return f(x,y), dp -> adj(dp, x, y)
end

## compose some functions and AD
g(f) = sum(abs2, f)
g1(f) = sum(abs2, f)
h = g ∘ f
h1 = g1 ∘ f1

x = rand(N)
y = rand(2)

all( Zygote.gradient(h, x, y) .≈ Zygote.gradient(h1, x, y) )

##
# now define a new function that evaluates D_x h only.
Dx_h = (x, y) -> (Zygote.gradient(x_ -> h(x_, y), x))[1]
# and compute it again with g (a nomincal outer nonlienarlty -> the loss)
k = g ∘ Dx_h
# and now differentiate k via Zyote.
Dy_k = (x, y) -> Zygote.gradient(y_ -> k(x, y_), y)[1]

##
# .... This fails unless we provide the rrule
function ChainRules.rrule(::typeof(adj), dp, x, y)
   #  think of adj = (a1, a2, a3) then
   #  we want D (dq1 * a1 + dq2 *   a2 + dq3 * a3) / D (dp, x, y)
   # but a1 is a NoTangent(), a2 = fx, a3 = fy from above. so a3 is not needed and this is
   # reflexted in dq3 = NoTangent()
   # finally a2 = cos.(x) .* v(y) .* dp, and secondAdj just needs to compute the
   # derivatives of   <dq2, a2> w.r.t.all three parameters (dp, x, y)
   #     (note adj has 3 input parameters!!!)
   function secondAdj(dq)
      Dqa2_Ddp = dq[2] .* cos.(x) .* v(y)
      Dqa2_Dx = NoTangent()   # we ignore this one, but technically it has a value
      Dqa2_Dy = Dv(y)' * (dq[2] .* dp .* cos.(x))
      return NoTangent(), Dqa2_Ddp, Dqa2_Dx, Dqa2_Dy
   end
   return adj(dp, x, y), secondAdj
end

Dy_k(x, y)
##

fdtest( y_ -> k(x, y_), y_ -> Dy_k(x, y_), y )