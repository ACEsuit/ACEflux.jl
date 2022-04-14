using ACEflux
import Base.copyto!


function read_pymatgen(file)
   import StaticArrays
   using PyCall, ASE
   ase = pyimport("ase")
   pmt = pyimport("pymatgen.core")
   pmt2ase = pyimport("pymatgen.io.ase")
   json = pyimport("json")

   X = []
   Y = []
   f = open(file, "r")
   data = json.load(f)
   for d in data
      #reads the structure using pymatgen
      curr_structure = pmt.Structure.from_dict(d["structure"])
      #convert that to an ASE (python) object
      at = pmt2ase.AseAtomsAdaptor.get_atoms(curr_structure)
      at_JuLIP = Atoms(ASEAtoms(at))
      push!(X, at_JuLIP)

      #TODO check that the forces are correct
      atnum = d["num_atoms"]
      ftmp = zeros(3,atnum) 
      for i in 1:atnum
         ftmp[:,i] = d["outputs"]["forces"][i,:]
      end
      push!(Y, [ d["outputs"]["energy"], matrix2svector(ftmp)])
   end
   return (X,Y)
end


function distribute(object)
   @eval @everywhere object=$object
end

function make_fg_multi(pot, loss, X, Y, λ, procs)\
   #import packages to all procs
   using Distributed
   Distributed.addprocs(procs)
   @everywhere using Pkg
   @everywhere Pkg.activate(pwd())
   @everywhere Pkg.instantiate()
   @show nprocs()

   @everywhere using JuLIP, ACEflux, Zygote, Flux, LinearAlgebra


   #distribute the values to all processors
   @eval @everywhere pot=$pot
   @eval @everywhere loss=$loss
   @eval @everywhere X=$X
   @eval @everywhere Y=$Y

   #divide dataset into workers
   @everywhere np = nprocs() - 1 
   @everywhere yourpart(x, n) = [x[i:min(i+n-1,length(x))] for i in 1:n:length(x)]
   @everywhere i4worker = yourpart(1:length(X),Int(ceil(length(X)/np)))
   np = length(i4worker)

   #we will need the size of ACE parameters throhgout (hardcoded to 1 ace model)
   s1,s2 = size(pot.model[1].weight)


   @everywhere veclength(params::Flux.Params) = sum(length, params.params)
   @everywhere veclength(x) = length(x)
   @everywhere Base.zeros(pars::Flux.Params) = zeros(veclength(pars))

   @everywhere function copyto!(v::AbstractArray, pars::Flux.Params)
      @assert length(v) == veclength(pars)
      s = 1
      for g in pars.params
         l = length(g)
         v[s:s+l-1] .= vec(g)
         s += l
      end
      v
   end

   @everywhere function copyto!(pars::Flux.Params, v::AbstractArray)
      s = 1
      for p in pars.params
         l = length(p)
         p .= reshape(v[s:s+l-1], size(p))
         s += l
      end
      pars
   end

   function gsum2mat(gs)
      sol = gs[1].grads[gs[1].params[1]]
      for i in 2:length(gs)
         sol = sol .+ gs[i].grads[gs[i].params[1]]
      end
      return sol
   end

   @everywhere function _gl(x, y)
      l, back = Zygote.pullback(()->loss(pot,x,y), Flux.params(pot.model))
      return(l, back(1))
   end

   function gradcalc()
      futures = Vector{Future}(undef, np)
      for i in 1:np
         f = @spawnat (i+1) map((x,y) -> _gl(x,y), X[i4worker[i]], Y[i4worker[i]])
         futures[i] = f
      end
      fg = fetch.(futures)
      tl = 0.0
      grds = []
      for c1 in fg
         for c2 in c1
            l, g = c2
            tl += l
            push!(grds, g)
         end
      end
      regloss = tl / length(X) + λ*norm(pot.model[1].weight, 2)^2 #+ λ1*norm(pot.model[1].weight, 1) 
      reggrad = collect(Iterators.flatten(gsum2mat(grds))) ./ length(X) + collect(Iterators.flatten(2*λ*pot.model[1].weight)) #.+ λ1
      #the regularization term needs to be flattened when gsum2mat changes - i.e. FS
      return regloss , reggrad
   end

   fg! = function (F,G,w)
      pot.model[1].weight = reshape(w,s1,s2) #we reshape the flat array into a matrix and put it inside the local model
      deepp = deepcopy(Flux.params(pot.model)) #we deepcopy the parameters of the model
      @eval @everywhere deepp=$deepp #we send to all workers the deep copy
      @everywhere Flux.loadparams!(pot.model, deepp) #we load this new parameters to the model in each worker

      if G !== nothing
         l, grads = gradcalc()
         copyto!(G, grads)
         return l
      end
      if F !== nothing
         @show "losscalc"
         l, grads = gradcalc()
         return l
      end
   end

   return fg!
end