using Zygote
using StatsBase
using ThreadTools
using Folds
using BSON: @save

function gsum(grads)
   gr = first(grads)
   for g in grads[2:length(grads)]
      gr = gr .+ g
   end
   return(gr)
end

#+(a::Zygote.Grads, b::Zygote.Grads) = a .+ b 


#loss function, parameters in the form of params(model), X input samples, Y targets
#the optimizer (flux object), the size of the batch, number of epochs,
#multi threading or single thread (boolean)
function opt_Flux(loss, θ, X, Y, opt, epochs, nthrds, fname; b=length(X), lograte = 50)
   trn_loss = Float64[]
   gradN = Float64[]
   append!(trn_loss, mean(loss.(X,Y)))
   n = 0 #a hack so that n exists outside the loop

   #our batch size, saved as a mutable array of length b
   indx = zeros(Int64,b)
   tic = time()
   for e in 1:epochs

      #we sample our training data and get the gradient
      StatsBase.sample!(1:length(X), indx; replace=false)

      if(e%lograte==0)

         g = gsum(tmap((x,y) -> Zygote.gradient(()->loss(x,y), θ), nthrds, X[indx], Y[indx]))
         
         Flux.Optimise.update!(opt, θ, g)
      
         append!(trn_loss, sum(tmap((x,y) -> loss(x,y), nthrds, X,Y)) / length(X))
         for gp in g #L1 norm, (g is a Flux gradient object)
            n += sum(abs.(gp))
         end
         append!(gradN,n)

         @show "#############################"
         @show (epochs - e)*(time() - tic) / (lograte*3600)
         @show last(trn_loss)

         @save string(fname, ".bson") θ

         tic = time()

      else
         #both have similar performance
         g = gsum(tmap((x,y) -> Zygote.gradient(()->loss(x,y), θ), nthrds, X[indx], Y[indx]))
         #g = Folds.mapreduce((x,y) -> Zygote.gradient(()->loss(x,y), θ), +, X[indx], Y[indx])

         Flux.Optimise.update!(opt, θ, g)
      
         append!(trn_loss, sum(tmap((x,y) -> loss(x,y), nthrds, X,Y)) / length(X))
         for gp in g #L1 norm, (g is a Flux gradient object)
            n += sum(abs.(gp))
         end
         append!(gradN,n)
      end

      
   end
   append!(gradN, n) #so that gradN has same length as iterations
   return (θ, trn_loss, gradN)
end