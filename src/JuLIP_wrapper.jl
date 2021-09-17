using JuLIP

struct FluxPotential{TM, TC} <: SitePotential
   model::TM
   cutoff::TC
end

(y::FluxPotential)(x) = y.model(x)

NeighbourLists.cutoff(V::FluxPotential) = V.cutoff

function JuLIP.evaluate!(tmp, V::FluxPotential, R, Z, z0)
   R=ACEConfig([State(rr = R[j]) for j in 1:length(R)]) #TODO how to get this derivated or out of eval
   tmp = V(R)
   return tmp
end

function JuLIP.evaluate_d!(dEs, tmp, V::FluxPotential, R, Z, z0)
   R=ACEConfig([State(rr = R[j]) for j in 1:length(R)])
   dEs = Zygote.gradient( x -> V(x), R )[1]
   # @show length(dEs)
   # @show length(tmp.dV)
   #TODO Fix the lengths and cutoff
   for i in 1:length(tmp.dV)
      tmp.dV[i] = dEs[i].rr
   end
   return tmp
end

JuLIP.energy(V::FluxPotential, at::AbstractAtoms) =
   JuLIP.energy!(JuLIP.Potentials.alloc_temp(V, at), V, at)

function presets(calc, at)
   TFL = JuLIP.fltype_intersect(calc, at)
   E = zero(TFL)
   nlist = neighbourlist(at, cutoff(calc))
   return(E, nlist)
end

function ChainRules.rrule(::typeof(presets), calc, at)
   return presets(calc, at), dp -> dp
end

function ChainRules.rrule(::typeof(JuLIP.Potentials.neigsz!), tmp, nlist, at, i)
   return JuLIP.Potentials.neigsz!(tmp, nlist, at, i), dp -> dp
end

function JuLIP.energy!(tmp, calc::FluxPotential, at::Atoms;
   domain=1:length(at))
   E, nlist = presets(calc, at)
   for i in domain
      j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
      E += JuLIP.evaluate!(tmp, calc, R, Z, at.Z[i])
   end
   return E
end

JuLIP.forces(V::FluxPotential, at::AbstractAtoms; kwargs...) =
   JuLIP.forces!(zeros(JVec{JuLIP.fltype_intersect(V, at)}, length(at)),
              JuLIP.Potentials.alloc_temp_d(V, at), V, at; kwargs...)

function JuLIP.forces!(frc, tmp, calc::FluxPotential, at::Atoms;
                 domain=1:length(at), reset=true)
   TFL = JuLIP.fltype_intersect(calc, at)
   if reset; fill!(frc, zero(eltype(frc))); end
   nlist = neighbourlist(at, cutoff(calc))
   for i in domain
      j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
      if length(j) > 0
         JuLIP.evaluate_d!(tmp.dV, tmp, calc, R, Z, at.Z[i])
         for a = 1:length(j)
            frc[j[a]] -= tmp.dV[a]
            frc[i]    += tmp.dV[a]
         end
      end
   end
   return frc
end
