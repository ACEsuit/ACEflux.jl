using ACE
using ACE: NaiveTotalDegree, State, SymmetricBasis, evaluate
using JuLIP

#to change the size of Psi
#   size of data
# a.- change the number of atoms in the structure 
# b.- change the size of each atom, also affects size of structure
#   size of parameters
# c.- change maxdeg 


#create several atoms
structure = []
for _ in 1:100 # a.-
   at = bulk(:Si, cubic=true) * 3 # b.-
   r1 = rand() * 0.6
   rattle!(at, r1)
   set_calculator!(at, StillingerWeber())
   push!(structure, at)
end

#make our basis
maxdeg = 12 # c.-
ord = 3
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = NaiveTotalDegree(),  rcut = cutoff(StillingerWeber()))
pibasis = PIBasis(B1p, ord, maxdeg; property = ACE.Invariant())
basis = SymmetricBasis(pibasis, ACE.Invariant());   

Psi = zeros(length(structure)*length(structure[1]),length(basis))
y = zeros(length(structure)*length(structure[1]))

#loop over the atoms
idx = 0
for at in structure
   nlist = neighbourlist(at, cutoff(StillingerWeber()))
   for i = 1:length(at)
      Js, tmpRs, Zs = JuLIP.Potentials.neigsz(nlist, at, i); z0 = at.Z[i]
      Rs = ACEConfig([State(rr = tmpRs[j]) for j in 1:length(tmpRs)])
      global idx += 1
      Psi[idx, :] = getproperty.(evaluate(basis, Rs), :val)
      y[idx] = evaluate(StillingerWeber(), tmpRs)
   end 
end




