using ACEgnns, JuLIP, StaticArrays
import ChainRulesCore, ChainRules
import ChainRulesCore: rrule, NoTangent

#everything in energy!() or forces!() that we don't want to differentiate
function Flux_neighbours(calc::FluxPotential, at::Atoms)
   tmp = JuLIP.Potentials.alloc_temp(calc, at)
   domain=1:length(at)
   nlist = neighbourlist(at, cutoff(calc))
   domain_R = []
   J = []
   for i in domain
      j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
      R = ACEConfig([State(rr = R[j]) for j in 1:length(R)])
      push!(domain_R, R)
      push!(J, j)
   end
   return domain_R
end

#TODO change for a nograd
function ChainRules.rrule(::typeof(Flux_neighbours), calc::FluxPotential, at::Atoms)
   return Flux_neighbours(calc, at), dp -> dp
end

function FluxEnergy(calc::FluxPotential, at::Atoms)
   domain_R = Flux_neighbours(calc, at)
   return sum([calc(r) for r in domain_R])
end

#still not working
function FluxForces(calc::FluxPotential, at::Atoms)
   tmp = JuLIP.Potentials.alloc_temp(calc, at)
   domain=1:length(at)
   nlist = neighbourlist(at, cutoff(calc))
   domain_R = []
   J = []
   for i in domain
      j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
      R = ACEConfig([State(rr = R[j]) for j in 1:length(R)])
      push!(domain_R, R)
      push!(J, j)
   end
   frc = zeros(SVector{3, Float64}, length(at))
   for (i,r) in enumerate(domain_R)
      tmpfrc = Zygote.gradient(calc, r)[1]
      for a = 1:length(J[i])
         frc[J[i][a]] -= tmpfrc[a].rr
         frc[i] += tmpfrc[a].rr
      end
   end
   return frc
end

function ChainRules.rrule(::typeof(FluxForces), calc::FluxPotential, at::Atoms)
   tmp = JuLIP.Potentials.alloc_temp(calc, at)
   domain=1:length(at)
   nlist = neighbourlist(at, cutoff(calc))
   domain_R = []
   J = []
   for i in domain
      j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
      R = ACEConfig([State(rr = R[j]) for j in 1:length(R)])
      push!(domain_R, R)
      push!(J, j)
   end

   function adj(dp_t)
      dt = [ACE.DState(rr=dp_t[i]) for i in 1:length(dp_t)]
      @show dt
      #create a "Gradient" object to store everything
      dptmp = [ACE.DState(rr=zeros(3)) for _ in 1:length(domain_R[1])]
      _ , pback = Zygote.pullback(()->Zygote.gradient(calc, domain_R[1])[1], params(calc.model))
      dfrc = pback(dptmp)
      @show dfrc[params(calc.model)[1]]
      for (i,r) in enumerate(domain_R)
         @show i
         dp = [ACE.DState(rr=zeros(3)) for _ in 1:length(r)]
         for a = 1:length(J[i])
            dp[a] -= dt[J[i][a]]
            #dp[a] += dt[i]
         end
         _ , pback = Zygote.pullback(()->Zygote.gradient(calc, r)[1], params(calc.model))
         _dfrc = pback(dp)
         for dprm in params(calc.model)
            if(_dfrc[dprm] != nothing)
               dfrc[dprm] += _dfrc[dprm] 
            end
         end   
      end
      return dfrc, dfrc, dfrc
   end

   frc = zeros(SVector{3, Float64}, length(at))
   for (i,r) in enumerate(domain_R)
      tmpfrc = Zygote.gradient(calc, r)[1]
      for a = 1:length(J[i])
         frc[J[i][a]] -= tmpfrc[a].rr
         frc[i] += tmpfrc[a].rr
      end
   end
   return frc, adj
end