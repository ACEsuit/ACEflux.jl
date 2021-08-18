using Flux
using JuLIP
using ACE
using ACE.Testing
using StaticArrays
using Statistics
using StatsBase
using JuLIP: EMT
using ACE: State, NaiveTotalDegree, SymmetricBasis, evaluate, LinearACEModel, set_params!, grad_params
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent 
using Printf, Test
using Plots
using LinearAlgebra
import Base: ^, sqrt, +, /

#some extra functions to handle SVectors
^(φ::SVector, x::Number) = φ .^ x
sqrt(φ::SVector) = sqrt.(φ)
+(φ::SVector, x::Number) = φ .+ x
/(φ::SVector, x::SVector) = φ ./ x


# # ------------------------------------------------------------------------
# #    Energy calculator functions
# # ------------------------------------------------------------------------

struct ENERGY_FS{TM, TRS, Tσ}
    m::TM 
    Rs::TRS
    σ::Tσ
end

function ENERGY_FS(m, at, vref, Tσ)
    Rs = []
    nlist = neighbourlist(at, cutoff(vref))
    for i = 1:length(at)
        Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
        tmpRs=ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
        append!(Rs,[tmpRs])
    end 
    return ENERGY_FS(m,Rs,Tσ)
end


function (y::ENERGY_FS)(θ)
   #evaluate gets site energies
   #getpropety makes them numbers rather than invariants
   #sum adds over site energies to get the energy
   #σ is the FS model
   set_params!(y.m, θ)
   E = y.σ(sum([getproperty.(evaluate(y.m ,r), :val) for r in y.Rs]))
   return E
end

function ChainRules.rrule(y::ENERGY_FS, θ)
   set_params!(y.m, θ)
   E = sum([getproperty.(evaluate(y.m ,r), :val) for r in y.Rs])

   function adj(dp)
      gσt = Flux.gradient(y.σ, E)[1]
      gσ = SVector{length(gσt)}(gσt)
      gparams = sum([grad_params(LM ,r) for r in y.Rs])
      grad = zeros(SVector{length(gparams[1])},length(gparams))
      for i = 1:length(gparams)
         grad[i] = getproperty.(gparams[i], :val) .* gσ
      end
      return (NoTangent(), dp * grad) 
   end
   return σ(E), adj
end

# # ------------------------------------------------------------------------
# #    ACE parameters
# # ------------------------------------------------------------------------

#this non-linearity should be flexible (make sure σ and Nprop match)
Nprop = 2
σ = ϕ -> ϕ[1] + sqrt((1/10)^2 + abs(ϕ[2])) - 1/10

#size of our ACE basis
maxdeg = 6
ord = 4

#building the basis
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = NaiveTotalDegree())
pibasis = PIBasis(B1p, ord, maxdeg; property = ACE.Invariant())
basis = SymmetricBasis(pibasis, ACE.Invariant());   

#create a multiple property model
W = rand(SVector{Nprop,Float64}, length(basis))
LM = LinearACEModel(basis, W, evaluator = :standard) 

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
      #we don't save the atom as training data, but rather a structure
      #to calculate the energy, this saves time when calculating the 
      #neighbourlist
      push!(train, ENERGY_FS(LM, at, EMT(), σ))
      push!(Etrain, energy(EMT(),at))
   end
   return(train,Etrain)
end

X_train,y_train = data(100)
X_test,y_test = data(2)

# # ------------------------------------------------------------------------
# #    Losses and optimizers
# # ------------------------------------------------------------------------

L_train(θ,indx) = sum([sum(abs2,X_train[i](θ) - y_train[i]) for i in indx])/length(indx)

L_test(θ) = sum([sum(abs2,X_test[i](θ) - y_test[i]) for i in 1:length(y_test)])/length(y_test)

function opt_Flux(p, opt, b, iter)
   #saving information
   trn_loss = []
   tst_loss = []
   gradN = []
   #first append
   append!(trn_loss, L_train(p,1:length(X_train)))
   append!(tst_loss, L_test(p))
   #our batch size, saved as a mutable array of length b
   indx = zeros(Int64,b)
   n=0 #n is here to double append the last grad, a small hack
   for _ in 1:iter
       #we sample our training data and get the gradient
       StatsBase.sample!(1:length(X_train), indx; replace=false)
       g = Flux.gradient(θ -> L_train(θ,indx),p)[1]
       
       #here is where the heavy lifting is done
       Flux.Optimise.update!(opt, p, g)

       #data to save
       append!(trn_loss, L_train(p,1:length(X_train)))
       n=norm(g,1)
       append!(gradN,n)
       append!(tst_loss, L_test(p))

   end
   append!(gradN, n)
   return (p, trn_loss, gradN, tst_loss)
end

# # ------------------------------------------------------------------------
# #    Running the code
# # ------------------------------------------------------------------------

@info "optimize"

opt = ADAM(0.01, (0.9, 0.8)) #https://fluxml.ai/Flux.jl/stable/training/optimisers/
b = 100 #batch
iter = 50 #epochs
(p, trn_loss, gradN, tst_loss) = opt_Flux(W, opt, b, iter)

plot(trn_loss, yaxis=:log, label="SGD", xlabel = "iterations", ylabel = "test Loss")


















# # # ------------------------------------------------------------------------
# # #    Finite difference test
# # # ------------------------------------------------------------------------

# function fdtestSVector(F, dF, x; h0 = 1.0, verbose=true)
#    errors = Float64[]
#    E = F(x)
#    dE = dF(x)
#    # loop through finite-difference step-lengths
#    verbose && @printf("---------|----------- \n")
#    verbose && @printf("    h    | error \n")
#    verbose && @printf("---------|----------- \n")
#    for p = 2:11
#       h = 0.1^p
#       dEh = copy(dE)
#       for n = 1:length(dE)
#          x[n] = x[n] + SVector{2}([h,0])
#          t1 = (F(x) - E) / h
#          x[n] = x[n] - SVector{2}([h,0])
#          x[n] = x[n] + SVector{2}([0,h])
#          t2 = (F(x) - E) / h
#          x[n] = x[n] - SVector{2}([0,h])

#          dEh[n] = SVector{2}([t1,t2])
#       end
#       push!(errors, norm(dE - dEh, Inf))
#       verbose && @printf(" %1.1e | %4.2e  \n", h, errors[end])
#    end
#    verbose && @printf("---------|----------- \n")
#    if minimum(errors) <= 1e-3 * maximum(errors)
#       verbose && println("passed")
#       return true
#    else
#       @warn("""It seems the finite-difference test has failed, which indicates
#       that there is an inconsistency between the function and gradient
#       evaluation. Please double-check this manually / visually. (It is
#       also possible that the function being tested is poorly scaled.)""")
#       return false
#    end
# end

# for ntest = 1:30
#     c_tst = rand(SVector{Nprop,Float64}, length(basis))
#     F = t ->  L_test(c_tst + t)
#     dF = t -> Flux.gradient(L_test, c_tst)[1]
#     print_tf(@test fdtestSVector(F, dF, zeros(SVector{Nprop,Float64}, length(basis)), verbose=false))
# end