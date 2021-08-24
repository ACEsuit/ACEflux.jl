
using ACE.Testing
using JuLIP
using Statistics
using StatsBase
using Flux
using StaticArrays
using Printf, Test
using Plots
using LinearAlgebra

using ACE
using ACE: State, NaiveTotalDegree, SymmetricBasis, evaluate, LinearACEModel, set_params!, grad_params
using ACEgnns

#some extra functions to handle SVectors
import Base: ^, sqrt, +, /
^(φ::SVector, x::Number) = φ .^ x
sqrt(φ::SVector) = sqrt.(φ)
+(φ::SVector, x::Number) = φ .+ x
/(φ::SVector, x::SVector) = φ ./ x

# # ------------------------------------------------------------------------
# #    Model definition
# # ------------------------------------------------------------------------

#this non-linearity should be flexible (make sure σ and Nprop match)
Nprop = 2

#activation function or non-linearity for the ACE layer
σ = ϕ -> ϕ[1] + sqrt((1/10)^2 + abs(ϕ[2])) - 1/10

#size of our ACE basis
maxdeg = 6
ord = 4

ace_E = ENERGY_ACE(maxdeg, ord, Nprop)
model = x -> Chain(ace_E, Dense(Nprop, 1))(x)[1]

nl_ace_E = NL_ENERGY_ACE(maxdeg, ord, Nprop, σ)

# # ------------------------------------------------------------------------
# #    Sample EMT data energies
# # ------------------------------------------------------------------------

#Random copper fcc atoms and EMT() energies
function data(Ntrain)
   train = []
   Etrain = []
   for i in 1:Ntrain
      at = bulk(:Cu, cubic=true) * 3
      rattle!(at,0.6)
      push!(train,at)
      push!(Etrain, energy(EMT(),at))
   end
   return(train,Etrain)
end

X_test,y_test = data(3)

@show model.(X_test)
@show nl_ace_E.(X_test)

m = Dense(Nprop, 1)
@show Flux.gradient(m -> m([1,2])[1], m)[1]
@show Flux.gradient(model -> sum(model.(X_test))[1], model)[1]

@show Flux.gradient(model -> sum(model.(X_test))[1], model)[1]

@show Flux.gradient(nl_ace_E -> sum(nl_ace_E.(X_test)), nl_ace_E)[1]






# # # ------------------------------------------------------------------------
# # #    Finite difference test for SVectors
# # # ------------------------------------------------------------------------

function fdtestSVector(F, dF, x; h0 = 1.0, verbose=true)
   errors = Float64[]
   E = F(x)
   dE = dF(x)
   # loop through finite-difference step-lengths
   verbose && @printf("---------|----------- \n")
   verbose && @printf("    h    | error \n")
   verbose && @printf("---------|----------- \n")
   for p = 2:11
      h = 0.1^p
      dEh = copy(dE)
      for n = 1:length(dE)
         x[n] = x[n] + SVector{2}([h,0])
         t1 = (F(x) - E) / h
         x[n] = x[n] - SVector{2}([h,0])
         x[n] = x[n] + SVector{2}([0,h])
         t2 = (F(x) - E) / h
         x[n] = x[n] - SVector{2}([0,h])

         dEh[n] = SVector{2}([t1,t2])
      end
      push!(errors, norm(dE - dEh, Inf))
      verbose && @printf(" %1.1e | %4.2e  \n", h, errors[end])
   end
   verbose && @printf("---------|----------- \n")
   if minimum(errors) <= 1e-3 * maximum(errors)
      verbose && println("passed")
      return true
   else
      @warn("""It seems the finite-difference test has failed, which indicates
      that there is an inconsistency between the function and gradient
      evaluation. Please double-check this manually / visually. (It is
      also possible that the function being tested is poorly scaled.)""")
      return false
   end
end

@info("AD test layers")

B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = NaiveTotalDegree())
pibasis = PIBasis(B1p, ord, maxdeg; property = ACE.Invariant())
basis = SymmetricBasis(pibasis, ACE.Invariant())  

# @info("ENERGY_ACE")
# #derivative is only computed for the ACE layer, the dense layer is assumed to
# #work since it's ported from Flux. 
# for ntest = 1:30
#     c_tst = rand(SVector{Nprop,Float64}, length(basis))
#     function F(t)
#       model.layers[1].weight = c_tst + t
#       return sum(model.(X_test))#L_test(X_test,y_test)
#     end
#     function dF(t)
#       model.layers[1].weight = c_tst + t
#       return Flux.gradient(model -> sum(model.(X_test))[1], model)[1]
#       #return Flux.gradient(L_test -> L_test(X_test,y_test), L_test)[1]
#     end
#     print_tf(@test fdtestSVector(F, dF, zeros(SVector{Nprop,Float64}, length(basis)), verbose=false))
# end

@info("NL_ENERGY_ACE")

for ntest = 1:30
   c_tst = rand(SVector{Nprop,Float64}, length(basis))
   function F(t)
      nl_ace_E.weight = c_tst + t
     return sum(nl_ace_E.(X_test))#L_test(X_test,y_test)
   end
   function dF(t)
      nl_ace_E.weight = c_tst + t
     return Flux.gradient(nl_ace_E -> sum(nl_ace_E.(X_test)), nl_ace_E)[1]
     #return Flux.gradient(L_test -> L_test(X_test,y_test), L_test)[1]
   end
   print_tf(@test fdtestSVector(F, dF, zeros(SVector{Nprop,Float64}, length(basis)), verbose=false))
end