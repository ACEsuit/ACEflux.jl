using Zygote
using ChainRules
using ChainRules: NoTangent


f(x,y) = x.^2 + y.^2

function adj(dxy, x, y)
   gx = 2 .* x .* dxy
   gy = 2 .* y .* dxy
   return(NoTangent(), gx, gy)
end

function ChainRules.rrule(::typeof(f), x,y)
   return(f(x,y), dxy -> adj(dxy, x, y))
end

function ChainRules.rrule(::typeof(adj),dxy,x,y)
   function secondDer(dt)
      @show 2 .* dt
      return(NoTangent(), 1, 2 .* dt, 2 .* dt)
   end
   return(adj(dxy,x,y), secondDer)
end

gx = Zygote.gradient(x->f(x,5),3)[1]
gy = Zygote.gradient(y->f(3,y),5)[1]

display(gx)
display(gy)



gxf = x -> Zygote.gradient(x->f(x,5),x)[1]
gyf = y -> Zygote.gradient(y->f(3,y),y)[1]


gxx = Zygote.gradient(gxf, 7)[1]
gyy = Zygote.gradient(gyf, 7)[1]


display(gyx)
display(gxx)
display(gyy)