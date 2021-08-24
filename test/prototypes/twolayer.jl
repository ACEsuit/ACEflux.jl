

maxdeg = 5
r0 = 1.0 
rcut = 3.0 
trans = trans = PolyTransform(1, r0)

# layer 1
RnYlm1 = ACE.Utils.RnYlm_1pbasis()
ACE.init1pspec!(RnYlm1, maxdeg = maxdeg, Deg = ACE.NaiveTotalDegree())
basis1 = SymmetricBasis(RnYlm1, Invariant())
model1 = LinearACEModel(basis1)

# layer 2
RnYlm2 = ACE.Utils.RnYlm_1pbasis()  # rr
Pk2 = ACE.scal1pbasis(:x, :k, maxdeg, trans, rcut)  # xi = model1({rij})
B1P2 = RnYlm2 * Pk2
ACE.init1pspec!(B1p2, maxdeg = maxdeg, Deg = ACE.NaiveTotalDegree())
basis2 = SymmetricBasis(, Invariant())

model = model1 ∘ model2



function evaluate(model, structure)
   # i -> config1(i)
   #      State(rr = ...)
   X = [ evaluate(model.L1, config1(i)) for i = 1:length(structure) ]

   # nonlinearity? 
   map!(X, atan)

   # i -> config2(i) 
   #      State(rr = ..., x = ...)
   Es = [ evaluate(model.L2, config2(i)) for i = 1:length(structure) ]
   # Es = [ evaluate(model.L2, config1(i), [X[j]]) for i = 1:length(structure) ]
   
   # other nonlinearities? 

   return sum(Es)
end