using ACE
using ACE: NaiveTotalDegree, State, SymmetricBasis, evaluate
using JuLIP

#to change the size of Psi
#   size of data
# a.- change the number of atoms in the structure 
# b.- change the size of each atom
# c.- change the number of atoms in the structure 
#   size of parameters
# d.- change maxdeg 


#create several atoms
structure = []
for _ in 1:100 # a.-
   at = bulk(:Si, cubic=true) * 2 # b.-
   rattle!(at,0.6)
   set_calculator!(at, StillingerWeber())
   push!(structure, at)
end

#make our basis
maxdeg = 12 # d.-
ord = 3
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = NaiveTotalDegree())
pibasis = PIBasis(B1p, ord, maxdeg; property = ACE.Invariant())
basis = SymmetricBasis(pibasis, ACE.Invariant());   

Psi = zeros(length(structure)*length(structure[1]),length(basis))
y = zeros(length(structure)*length(structure[1]))

#loop over the atoms
idx = 0
for at in structure
   #0.9 so that the structures are slightly larger than the radius used
   nlist = neighbourlist(at, 0.9 * cutoff(StillingerWeber()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      Rs = ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      global idx += 1
      Psi[idx, :] = getproperty.(evaluate(basis, Rs), :val)
      y[idx] = evaluate(StillingerWeber(), tmpRs)
   end 
end




