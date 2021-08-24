using Flux
using Statistics
using Flux: @epochs
using Flux: @functor
using ACE
using ACE: evaluate, SymmetricBasis, NaiveTotalDegree, PIBasis, set_params!
using StaticArrays


struct ACELayer{TW,TM,TS}
    weights::TW
    LM::TM
    σ::TS
end

function ACELayer(
    initW,
    σ,
    maxdeg,
    ord;
    totDeg = NaiveTotalDegree())

    #this is a naive initalization for now
    RnYlm1 = ACE.Utils.RnYlm_1pbasis()
    ACE.init1pspec!(RnYlm1, maxdeg = maxdeg, Deg = totDeg)
    pibasis = PIBasis(B1p, ord, maxdeg; property = ACE.Invariant())
    basis = SymmetricBasis(pibasis, φ);   

    LM = ACE.LinearACEModel(basis, initW, evaluator = :standard) 
 
    ACELayer(initW, LM, σ)
end

@functor ACELayer #for Flux to be able to update the weights

#evaluation of the ACE layer / Forward pass
function (l::ACELayer)(cfg)
   set_params!(l.LM, l.weights)
   u = l.σ(getproperty.(evaluate(l.LM ,cfg), :val))
   newCfg = ACEConfig([ACE.State(rr = cfg.Xs[i].rr, u = u) for i in 1:length(cfg)])
   newCfg
end


# ------------------------------------------------------------------------
#    Sample model of 2 FS layers
# ------------------------------------------------------------------------

#usual stuff
maxdeg = 5;
ord = 4;

D = NaiveTotalDegree();
B1p = ACE.Utils.RnYlm_1pbasis(; maxdeg=maxdeg, D = D);
φ = ACE.Invariant();
pibasis = PIBasis(B1p, ord, maxdeg; property = φ);
basis = SymmetricBasis(pibasis, φ);

nX = 54
Xs() = ACE.State(rr = rand(SVector{3, Float64}), u = rand())

#FS comes in as an activation function
σ = ϕ -> ϕ[1] + sqrt((1/10)^2 + abs(ϕ[2])) - 1/10

#initial weights
BB = evaluate(basis, ACEConfig([Xs() for i in 1:nX]))
initW = rand(SVector{2,Float64}, length(BB))

#2 layer model first layer just linear, second FS
model(x) = Chain(
                ACELayer(initW,x->x,maxdeg,ord),
                ACELayer(initW,σ,maxdeg,ord)
                )(x).Xs[1].u.val

loss(x, y) = Flux.Losses.mse(model(x), y)

opt = ADAM(0.3)
num_epochs = 10 

#training data (an array of configurations)
train_cfgs = [ACEConfig([Xs() for i in 1:nX]) for i in 1:80]
train_Ei =  rand(80)
test_cfgs =  [ACEConfig([Xs() for i in 1:nX]) for i in 1:15]
test_Ei =  rand(15)

train_data = zip(train_cfgs, train_Ei)

@show model(train_cfgs[1])

#A callback to see training progress
evalcb() = @show(mean(loss.(test_cfgs, test_Ei)))
evalcb()

# mod(x) = ACELayer(initW,σ,maxdeg,ord)(x).Xs[1].u.val

# W = initW

# grads = gradient(mod -> mod, loss)[1]
# grads2 = gradient(() -> loss(a,b))
# julia> struct Linear
#          W
#          b
#        end

# (l::Linear)(x) = l.W * x .+ l.b

# model = Linear(rand(2, 5), rand(2))
# Linear([0.267663 … 0.334385], [0.0386873, 0.0203294])

# dmodel = gradient(model -> sum(model(x)), model)[1]
# W = rand(2, 5); b = rand(2);

# linear(x) = W * x .+ b

# grads = gradient(() -> sum(linear(x)), Params([W, b]))










# function Flux.params(ACE::ACELayer)
#     return(ACE.weights)
# end
# params()
# ps = initW



# ps = Params(ps)
# for d in data
#     gs = gradient(ps) do loss(batchmemaybe(d)...)
#     update(opt, ps, gs)
# end
    

#   function opt_Flux(θ_S2, opt, b, iter)
#     #saving information
#     trn_loss = []
#     tst_loss = []
#     gradN = []
#     #first append
#     append!(trn_loss, L_FS(θ_S2,1:length(train)))
#     append!(tst_loss, L_test(θ_S2))
#     #our batch size, saved as a mutable array of length b
#     indx = zeros(Int64,b)
#     #n is here to double append the last grad, a small hack
#     n=0
#     for _ in 1:iter
#         #we sample our training data and get the gradient
#         StatsBase.sample!(1:length(train), indx; replace=false)
#         g = Zygote.gradient(θ -> L_FS(θ,indx),θ_S2)[1]

#         #flatten grad so it's easy to use
#         gs = [collect(Iterators.flatten(g))[i].val for i in 1:length(θ_S2)]
        
#         Flux.Optimise.update!(opt, θ_S2, gs)

#         append!(trn_loss, L_FS(θ_S2,1:length(train)))
#         n=norm(gs,1)
#         append!(gradN,n)
#         append!(tst_loss, L_test(θ_S2))

#     end
#     append!(gradN, n)
#     return (θ_S2, trn_loss, gradN, tst_loss)
# end

# update!(opt, ps, gs)

# # train
# println("Training!")
# @epochs num_epochs Flux.train!(
#     loss,
#     intiW,
#     train_data,
#     opt,
#     cb = Flux.throttle(evalcb, 10),
# )