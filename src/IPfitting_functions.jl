using IPFitting
using Base.Threads:          SpinLock, nthreads
using IPFitting:        Dat, observations, observation, tfor_observations, 
                        DB.set_matrows!, DB.matrows, DB.flush

using StaticArrays
using JuLIP: vecs, mat, JVec

struct db_vector{Tef,Tp, Ty}
   Ψ::Tef #energies and forces
   pot::Tp #Flux potential for ACE
   Y::Ty # the targets, TODO this should not be in db... probably?
end

const ENERGY = "E"
const FORCES = "F"
const ValE = Val{:E}
const ValF = Val{:F}
eval_obs_flux(s::AbstractString, args...) = eval_obs_flux(Val(Symbol(s)), args...)
vec_obs_flux(s::AbstractString, args...) = vec_obs_flux(Val(Symbol(s)), args...)

vec_obs_flux(::ValE, E::Real) = [E]
vec_obs_flux(::ValE, E::Vector{<: Real}) = E
vec_obs_flux(v::ValF, F::AbstractVector{<:JVec}) = vec_obs_flux(v, mat(F))
vec_obs_flux(::ValF, F::AbstractMatrix) = vec(F)
function vec_obs_flux(valF::ValF, F::Vector{<: Vector})
   nbasis = length(F)
   nat = length(F[1])
   Fmat = zeros(3*nat, nbasis)
   for ib = 1:nbasis
      Fmat[:, ib] .= vec_obs_flux(valF, F[ib])
   end
   return Fmat
end

eval_obs_flux(::ValE, pot, dat::Dat) = FluxEnergy(pot, dat.at)
eval_obs_flux(::ValF, pot, dat::Dat) = FluxForces(pot, dat.at)

function safe_append!(db::db_vector, db_lock, cfg, okey)
   # defined eval_obs that take potentials
   lsqrow = eval_obs_flux(okey, db.pot, cfg)
   vec_lsqrow = vec_obs_flux(okey, lsqrow)
   irows = matrows(cfg, okey) #since the rows don't change we should be able to re-use all of this code

   lock(db_lock)
   db.Ψ[irows] = vec_lsqrow 
   db.Y[irows] = cfg.D[okey] #TODO get rid of this part
   unlock(db_lock)
   
   return nothing
end

#same allocation code, but only a vector now
function _alloc_EF_vector(configs)

   nrows = 0
   for (okey, d, _) in observations(configs)
      len = length(observation(d, okey))
      set_matrows!(d, okey, collect(nrows .+ (1:len)))
      nrows += len
   end

   return zeros(Float64, nrows)
end

function LsqDBflux(dbpath::AbstractString,
               pot::FluxPotential,
               configs::AbstractVector{Dat};
               verbose=true,
               maxnthreads=nthreads())

   #now a vector instead of a matrix [E_r1, F_r1, E_r2, ... F_rk]
   Ψ = _alloc_EF_vector(configs)
   Y = _alloc_EF_vector(configs) #TODO move this somewhere else
   #structure now contains the energy forces vector and the potential
   db = db_vector(Ψ, pot, Y)

   #the only thing changed from here on is the safe_append
   tfor_observations( configs,
      (n, okey, cfg, lck) -> safe_append!(db, lck, cfg, okey),
      msg = "Assemble LSQ blocks",
      verbose=verbose,
      maxnthreads=maxnthreads )
   # save to file
   if dbpath != ""
      verbose && @info("Writing db to disk...")
      try
         flush(db)
      catch
         @warn("""something went wrong trying to save the db to disk, but the data
               should be ok; if it is crucial to keep it, try to save manually.""")
      end
      verbose && @info("... done")
   else
      verbose && @info("db is not written to disk since `dbpath` is empty.")
   end
   return db
end

