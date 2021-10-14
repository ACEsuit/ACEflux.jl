using JuLIP

"""
A wrapper around JuLIP. It basically defines a flux potential which is a model
Flux.jl recognizes using and ACE layer. It is not intended to be differentiated.
Once you have a trained Flux ACE multi property potential you can use this functions
to work directly with the ASE.jl machinery. 
For training look at calculator.jl
"""

mutable struct FluxPotential{TM, TC} <: SitePotential
   model::TM # a function taking atoms and returning a site energy
   cutoff::TC #in Angstroms
end

(y::FluxPotential)(x) = y.model(x) 

NeighbourLists.cutoff(V::FluxPotential) = V.cutoff


function JuLIP.evaluate!(tmp, V::FluxPotential, R, Z, z0)
   R = ACEConfig([State(rr = R[j]) for j in 1:length(R)])
   tmp = V(R)
   return tmp
end

JuLIP.energy(V::FluxPotential, at::AbstractAtoms) =
   JuLIP.energy!(JuLIP.Potentials.alloc_temp(V, at), V, at)

function JuLIP.energy!(tmp, calc::FluxPotential, at::Atoms;
   domain=1:length(at))
   E = zero(JuLIP.fltype_intersect(calc, at))
   nlist = neighbourlist(at, cutoff(calc))
   for i in domain
      j, R, Z = JuLIP.Potentials.neigsz!(tmp, nlist, at, i)
      E += JuLIP.evaluate!(tmp, calc, R, Z, at.Z[i])
   end
   return E
end



#still not working
function JuLIP.evaluate_d!(dEs, tmp, V::FluxPotential, R, Z, z0)
   R=ACEConfig([State(rr = R[j]) for j in 1:length(R)])
   dEs = Zygote.gradient( x -> V(x), R )[1]
   for i in 1:length(tmp.dV)
      tmp.dV[i] = dEs[i].rr
   end
   return tmp
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



#only for testing and developlemnt

#neighboor list finder with EMT()
function neighbourfinder(at)
   Rs = []
   nlist = neighbourlist(at, cutoff(EMT()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      tmpRs=ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      append!(Rs,[tmpRs])
   end 
   return Rs
end