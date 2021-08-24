using Flux
using ACE
using ACE: evaluate, SymmetricBasis, NaiveTotalDegree, PIBasis

# ------------------------------------------------------------------------
#    ACELayer
# ------------------------------------------------------------------------

#the activation funciton is inside GNL for now
struct ACELayer{T,TG}
    weights::Array{T,2}
    GNL::TG
end

function ACELayer(
    initW,
    σ,
    maxdeg,
    ord;
    totDeg = NaiveTotalDegree())

    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = totDeg)
    φ = ACE.Invariant()
    pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
    basis = SymmetricBasis(pibasis, φ);
    
    GNL = ACE.Models.NLModel(basis, initW, evaluator = :standard, F = σ) 
 
    ACELayer(initW, GNL)
end

#@functor ACELayer

#rr are distances and X is some featurized matrix that will be fed into the next layer
function (l::ACELayer)(rr)#, X)
    #for now we ignore X and only use rr
    out_mat = [ACE.EVAL(l.GNL.LM[i],rr)(l.weights[:,i]) for i=1:length(l.weights[1,:])]
    rr, out_mat
end


# ------------------------------------------------------------------------
#    Pooling layer, calculate energies
# ------------------------------------------------------------------------

#the activation funciton is inside GNL for now
struct ACEPool{T,TG}
    weights::Array{T,2}
    GNL::TG
end

function ACEPool(
    initW,
    σ,
    maxdeg,
    ord;
    totDeg = NaiveTotalDegree())

    B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = totDeg)
    φ = ACE.Invariant()
    pibasis = PIBasis(B1p, ord, maxdeg; property = φ)
    basis = SymmetricBasis(pibasis, φ);
    
    GNL = ACE.Models.NLModel(basis, initW, evaluator = :standard, F = σ) 

    ACEPool(initW, GNL)
end

#@functor ACEPool

#this layer joins an ACE layer plus a pooling layer
function (l::ACEPool)(rr)#, X)
    #for now we ignore X and only use rr
    out = ACE.Models.EVAL_NL(l.GNL,rr)(l.weights)
    rr, out
end


# ------------------------------------------------------------------------
#    Sample model of 2 FS layers
# ------------------------------------------------------------------------

maxdeg = 5;
ord = 4;
nX = 54

D = NaiveTotalDegree();
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D);
φ = ACE.Invariant();
pibasis = PIBasis(B1p, ord, maxdeg; property = φ);
basis = SymmetricBasis(pibasis, φ);

#training data (site energies for now)
train_cfgs = [ACEConfig(rand(EuclideanVectorState, B1p.bases[1], nX)) for i in 1:80]
train_Ei =  [evaluate(basis,train_cfgs[i]) for i in 1:80]
test_cfgs =  [ACEConfig(rand(EuclideanVectorState, B1p.bases[1], nX)) for i in 1:15]
test_Ei =  [evaluate(basis,test_cfgs[i]) for i in 1:15]

train_data = zip(train_cfgs, train_Ei)

σ = ϕ -> ϕ[1] + sqrt((1/10)^2 + abs(ϕ[2])) - 1/10
BB = evaluate(basis, ACEConfig(rand(EuclideanVectorState, B1p.bases[1], nX)))
initW = rand(length(BB),2).-0.5 

model = Chain(ACELayer(initW,σ,maxdeg,ord),ACEPool(initW,σ,maxdeg,ord))

loss(x, y) = Flux.Losses.mse(model(x), y)

opt = ADAM(0.003)

# and a callback to see training progress
evalcb() = @show(mean(loss.(test_cfgs, test_Ei)))
evalcb()

# train
println("Training!")
@epochs num_epochs Flux.train!(
    loss,
    params(model),
    train_data,
    opt,
    cb = Flux.throttle(evalcb, 10),
)





using Flux, ACE, StaticArrays
using ACE: acquire_B!, evaluate!, acquire!, _get_eff_coeffs!, release!, State
using ACE: Utils.RnYlm_1pbasis, Invariant, SymmetricBasis, PIBasis, NaiveTotalDegree 

#trains the non-linearity
struct NonLinACE{Tσ,TP}
    σ::Tσ
    p::TP
end

function NonLinACE() 

    σ(ϕ,p) = p[1]*ϕ[1] + p[2]*sqrt((1/10)^2 + abs(ϕ[2])) - 1/10

    NonLinACE(σ,rand(2))
end

function (l::NonLinACE)(val)
    u = l.σ(val,l.p) #for linear ACE σ=x->x
    u
end


#trains the parameters θ
struct Bbasis{TW,TP}
    c::TW
    basis::TP
end

function Bbasis(
    nP::Int, #number of properties
    pibasis,
    property = Invariant())  
    
    basis = SymmetricBasis(pibasis, property)
    initW = rand(SVector{nP,Float64}, length(basis))

    Bbasis(initW, basis)
end

#evaluation of the ACE layer / Forward pass
function (l::Bbasis)(data)
    (A,spec,_real) = data
    #use coeffs to get ctilde
    len_AA = length(l.basis.pibasis)
    @assert len_AA == size(l.basis.A2Bmap, 2)
    c̃ = acquire!(l.basis.B_pool, len_AA, SVector{length(l.c[1]),eltype(l.basis.A2Bmap)})
    _get_eff_coeffs!(c̃, l.basis, l.c)
    release!(l.basis.B_pool, c̃)

    #evaluate the values
    val = zero(eltype(c̃)) * _real(zero(eltype(A)))  
    @inbounds for iAA = 1:length(spec)
        aa = A[spec.iAA2iA[iAA, 1]]
        for t = 2:spec.orders[iAA]
            aa *= A[spec.iAA2iA[iAA, t]]
        end
        val += _real(aa) * c̃[iAA]
    end
    #release_B!(l.basis.pibasis.basis1p, cfg)
    val = getproperty.(val, :val)
    val
end


#doesn't train anything
struct pibasis{TB}
    basis::TB
end

function pibasis(
    B1p,
    ord,
    maxdeg,
    property = Invariant())  

    pibas = PIBasis(B1p, ord, maxdeg; property = property)
    
    pibasis(pibas)
end

function (l::pibasis)(A)
    spec = l.basis.spec
   _real = l.basis.real
   (A,spec,_real)
end


#we could move Rn to the forward pass to train on it
struct onePBasis{TB}
    basis::TB
end

function onePBasis(
    maxdeg::Int,
    totDeg = NaiveTotalDegree())
    B1p = RnYlm_1pbasis(; maxdeg=maxdeg, D = totDeg)
 
    onePBasis(B1p)
end

function (l::onePBasis)(cfg)
    A = acquire_B!(l.basis, cfg)
    A = evaluate!(A, l.basis, cfg) 
    A
end

#Building a model by composition of layers

PB1 = onePBasis(6)
PIB = pibasis(PB1.basis,4,6)
Bb  = Bbasis(2,PIB.basis)
NLA = NonLinACE()

model = Chain(PB1,PIB,Bb,NLA)

nX = 54
Xs() = State(rr = rand(SVector{3, Float64}), u = rand())
cfg = ACEConfig([Xs() for i in 1:nX])

model(cfg)