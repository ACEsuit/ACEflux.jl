using ACEgnns, JuLIP, StaticArrays
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent
import ACE: PositionState

#functions we don't want to differentiate
#TODO change this to a nograd, or Zygote.ignore or something similar.
function Flux_neighbours_R(calc::FluxPotential, at::Atoms)
   tmp = JuLIP.Potentials.alloc_temp(calc, at)
   domain=1:length(at)
   nlist = neighbourlist(at, cutoff(calc))

   TX = PositionState{Float64}
   TCFG = ACEConfig{TX}
   domain_R = TCFG[]
   for i in domain
      j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
      # cfg = ACEConfig(TX[ TX(rr = R[j]) for j in 1:length(R) ])
      cfg = ACEConfig( TX[ TX(rr = rr) for rr in R ] )
      push!(domain_R, cfg)
   end
   return domain_R
end

function ChainRules.rrule(::typeof(Flux_neighbours_R), calc::FluxPotential, at::Atoms)
   return Flux_neighbours_R(calc, at), dp -> (NoTangent(), dp, NoTangent())
end

function Flux_neighbours_J(calc::FluxPotential, at::Atoms)
   tmp = JuLIP.Potentials.alloc_temp(calc, at)
   domain=1:length(at)
   nlist = neighbourlist(at, cutoff(calc))
   J = []
   for i in domain
      j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
      push!(J, j)
   end
   return J
end

function ChainRules.rrule(::typeof(Flux_neighbours_J), calc::FluxPotential, at::Atoms)
   return Flux_neighbours_J(calc, at), dp -> (NoTangent(), dp, NoTangent())
end


##
#Energy and forces calculators

function FluxEnergy(calc::FluxPotential, at::Atoms)
   domain_R = Flux_neighbours_R(calc, at)
   return sum([calc(r) for r in domain_R])
end



function FluxForces(calc::FluxPotential, at::Atoms)
   domain_R = Flux_neighbours_R(calc, at)
   J = Flux_neighbours_J(calc, at)
   return _eval_forces(calc, at, domain_R, J)
end


# function rrule(::typeof(FluxForces), calc::FluxPotential, at::Atoms)

#    # compute forces 
#    domain_R = Flux_neighbours_R(calc, at)
#    J = Flux_neighbours_J(calc, at)
#    frc = _eval_forces(calc, at, domain_R, J)

#    function _pullback(dp)

#       return NoTangent(), dparams, NoTangent()
#    end
# end


function _eval_forces(calc::FluxPotential, at::Atoms, domain_R, J)
   frc = zeros(SVector{3, Float64}, length(at))
   for (i,r) in enumerate(domain_R)
      # [1] local forces
      tmpfrc = Zygote.gradient(calc, r)[1]
      # [2] loc to glob
      frc += loc_to_glob(tmpfrc, J[i], length(at), i)
   end
   return frc
end

function loc_to_glob(Gi, Ji, Nat, i)
   frc = zeros(SVector{3, Float64}, Nat)
   for a = 1:length(Ji)
      frc[Ji[a]] -= Gi[a].rr
      frc[i] += Gi[a].rr
   end
   return frc
end

function rrule(::typeof(loc_to_glob), Gi, Ji, Nat, i)
   frc = loc_to_glob(Gi, Ji, Nat, i)
   
   function _pullback(dP)  #dp is the global dp 
      TDX = eltype(frc) # typeof( DState( rr = zero(SVector{3, Float64}) ) )
      dPi = zeros( TDX, length(Ji) )
      for a = 1:length(Ji)
         dPi[a] -= dP[Ji[a]]
         dPi[a] += dP[i]
      end
      return dPi
   end

   return frc, dP -> (NoTangent(), _pullback(dP), NoTangent(), NoTangent(), NoTangent())
end



# _eval_local_forces(calc::FluxPotential, Ri) = Zygote.gradient(calc, r)[1]

# function rrule(::typeof(_eval_local_forces), calc::FluxPotential, Ri)
#    return _eval_local_forces(calc, Ri), 
#           dp -> (NoTangent(), NoTangent(), NoTangent())
# end


# function ChainRules.rrule(::typeof(_eval_forces), calc::FluxPotential, at::Atoms, domain_R, J)
#    function adj(dp_t)
#       #change into DState (since forces chagnes Dstate into Svectors)
#       dt = [ACE.DState(rr=dp_t[i]) for i in 1:length(dp_t)]

#       #create a "Gradient" object to store everything TODO improve this
#       dptmp = [ACE.DState(rr=zeros(3)) for _ in 1:length(domain_R[1])]
#       _ , pback = Zygote.pullback(()->Zygote.gradient(calc, domain_R[1])[1], params(calc.model))
#       dfrc = pback(dptmp)
#       #loop over atoms and neighbours adding contributions to the adjoint
#       for (i,r) in enumerate(domain_R)
#          dp = [ACE.DState(rr=zeros(3)) for _ in 1:length(r)] #temporal adjoint with length r
#          for a = 1:length(J[i])
#             dp[a] -= dt[J[i][a]]
#             dp[a] += dt[i]
#          end
#          #call Zygote to get derivative, could maybe call rrule directly? 
#          _ , pback = Zygote.pullback(()->Zygote.gradient(calc, r)[1], params(calc.model))
#          _dfrc = pback(dp)

#          #a hack to avoid the nothing that happens when we try to derivate the cutoff parameter in FluxModel
#          for dprm in params(calc.model)
#             if(_dfrc[dprm] != nothing)
#                dfrc[dprm] += _dfrc[dprm] 
#             end
#          end   
#       end
#       return NoTangent(), dfrc, NoTangent(), NoTangent(), NoTangent()
#    end
#    return _eval_forces(calc, at, domain_R, J), adj
# end





#a structure in an attempt to return the gradient when called as 
#Zygote.gradient(()->loss(FluxModel, at), p)
# mutable struct FluxForces{TC} <: SitePotential
#    calc::TC
# end

# function (y::FluxForces)(at::Atoms)
#    domain_R = Flux_neighbours_R(y.calc, at)
#    J = Flux_neighbours_J(y.calc, at)
#    return _eval_forces(y.calc, at, domain_R, J)
# end

# function _eval_forces(calc::FluxPotential, at::Atoms, domain_R, J)
#    frc = zeros(SVector{3, Float64}, length(at))
#    for (i,r) in enumerate(domain_R)
#       tmpfrc = Zygote.gradient(calc, r)[1]
#       for a = 1:length(J[i])
#          frc[J[i][a]] -= tmpfrc[a].rr
#          frc[i] += tmpfrc[a].rr
#       end
#    end
#    return frc
# end

# function ChainRules.rrule(::typeof(_eval_forces), calc::FluxPotential, at::Atoms, domain_R, J)
#    function adj(dp_t)
#       #change into DState (since forces chagnes Dstate into Svectors)
#       dt = [ACE.DState(rr=dp_t[i]) for i in 1:length(dp_t)]

#       #create a "Gradient" object to store everything TODO improve this
#       dptmp = [ACE.DState(rr=zeros(3)) for _ in 1:length(domain_R[1])]
#       _ , pback = Zygote.pullback(()->Zygote.gradient(calc, domain_R[1])[1], params(calc.model))
#       dfrc = pback(dptmp)
#       #loop over atoms and neighbours adding contributions to the adjoint
#       for (i,r) in enumerate(domain_R)
#          dp = [ACE.DState(rr=zeros(3)) for _ in 1:length(r)] #temporal adjoint with length r
#          for a = 1:length(J[i])
#             dp[a] -= dt[J[i][a]]
#             dp[a] += dt[i]
#          end
#          #call Zygote to get derivative, could maybe call rrule directly? 
#          _ , pback = Zygote.pullback(()->Zygote.gradient(calc, r)[1], params(calc.model))
#          _dfrc = pback(dp)

#          #a hack to avoid the nothing that happens when we try to derivate the cutoff parameter in FluxModel
#          for dprm in params(calc.model)
#             if(_dfrc[dprm] != nothing)
#                dfrc[dprm] += _dfrc[dprm] 
#             end
#          end   
#       end
#       return NoTangent(), dfrc, NoTangent(), NoTangent(), NoTangent()
#    end
#    return _eval_forces(calc, at, domain_R, J), adj
# end

#another attempt at getting the @train working

# #still not working
# function FluxForces(calc::FluxPotential, at::Atoms)
#    tmp = JuLIP.Potentials.alloc_temp(calc, at)
#    domain=1:length(at)
#    nlist = neighbourlist(at, cutoff(calc))
#    domain_R = []
#    J = []
#    for i in domain
#       j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
#       R = ACEConfig([State(rr = R[j]) for j in 1:length(R)])
#       push!(domain_R, R)
#       push!(J, j)
#    end
#    frc = zeros(SVector{3, Float64}, length(at))
#    for (i,r) in enumerate(domain_R)
#       tmpfrc = Zygote.gradient(calc, r)[1]
#       for a = 1:length(J[i])
#          frc[J[i][a]] -= tmpfrc[a].rr
#          frc[i] += tmpfrc[a].rr
#       end
#    end
#    return frc
# end

# function ChainRules.rrule(::typeof(FluxForces), calc::FluxPotential, at::Atoms)
#    tmp = JuLIP.Potentials.alloc_temp(calc, at)
#    domain=1:length(at)
#    nlist = neighbourlist(at, cutoff(calc))
#    domain_R = []
#    J = []
#    for i in domain
#       j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
#       R = ACEConfig([State(rr = R[j]) for j in 1:length(R)])
#       push!(domain_R, R)
#       push!(J, j)
#    end

#    function adj(dp_t)
#       dt = [ACE.DState(rr=dp_t[i]) for i in 1:length(dp_t)]
#       #create a "Gradient" object to store everything
#       dptmp = [ACE.DState(rr=zeros(3)) for _ in 1:length(domain_R[1])]
#       _ , pback = Zygote.pullback(()->Zygote.gradient(calc, domain_R[1])[1], params(calc.model))
#       dfrc = pback(dptmp)
#       for (i,r) in enumerate(domain_R)
#          dp = [ACE.DState(rr=zeros(3)) for _ in 1:length(r)]
#          for a = 1:length(J[i])
#             dp[a] -= dt[J[i][a]]
#             dp[a] += dt[i]
#          end
#          _ , pback = Zygote.pullback(()->Zygote.gradient(calc, r)[1], params(calc.model))
#          _dfrc = pback(dp)
#          for dprm in params(calc.model)
#             if(_dfrc[dprm] != nothing)
#                dfrc[dprm] += _dfrc[dprm] 
#             end
#          end   
#       end
#       return dfrc, dfrc, dfrc
#    end

#    frc = zeros(SVector{3, Float64}, length(at))
#    for (i,r) in enumerate(domain_R)
#       tmpfrc = Zygote.gradient(calc, r)[1]
#       for a = 1:length(J[i])
#          frc[J[i][a]] -= tmpfrc[a].rr
#          frc[i] += tmpfrc[a].rr
#       end
#    end
#    return frc, adj
# end