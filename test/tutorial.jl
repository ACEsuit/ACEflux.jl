using ACE, ACEgnns, Zygote, Flux, ACE, StaticArrays

using Zygote: gradient

@everywhere begin
cfg = ACE.ACEConfig([ACE.State(rr=rand(SVector{3, Float64})) for _ = 1:10])

FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10

fs_model = Chain(Linear_ACE(3, 4, 2), GenLayer(FS), sum)

end

fs_model(cfg)

g = gradient(()->fs_model(cfg), Flux.params(fs_model))

g = gradient(fs_model, cfg)[1]


g[Flux.params(fs_model)[1]]


# I have loaded the Si data set here 
# -show table 
all_Si = IPFitting.Data.read_xyz("/zfs/users/aross88/aross88/silicon/Si.xyz", energy_key="dft_energy", force_key="dft_force", verbose=false)
data = filter(at -> configtype(at) == "dia", all_Si) #get diamon configurations

#we clean it up 

Y = []  
X = []
for i in 1:length(data)
   push!(Y, [ data[i].D["E"][1], ACEgnns.matrix2svector(reshape(data[i].D["F"], (3, Int(length(data[i].D["F"])/3)))) ])
   push!(X, data[i].at)
end

X[1]
Y[1]

#now we defina a model
#Linear_ACE is the fundamental object. It takes a configuration and 
#returns the site energy for each property. It has 

linace = Linear_ACE(3, 7, 2) #cor order, max deg, num of properties

cfg = ACE.ACEConfig([ACE.State(rr=rand(SVector{3, Float64})) for _ = 1:10])
linace(cfg)

fieldnames(typeof(linace))
linace.weight

#this is the fundamental object. For example if we wanted a FS model we simply
#pass this properties to the FS function

FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10

fs_model = x -> FS(linace(x))

#chain provides an easier way to do this

fs_model = Chain(Linear_ACE(3, 7, 2), GenLayer(FS))

#Flux provides a nice function to retrieve the parameters
params(fs_model)

#and one can even go more general

model = Chain(Linear_ACE(3,7,2), Dense(2,7), Dense(7,2), GenLayer(FS), sum)

#where now we have parameters we want to train in the Dense layers too
params(model)[1]
params(model)[2]
params(model)[3]

#now we define a potential, simply expand it with a cutoff radius

pot = FluxPotential(model, 6.0) #model, cutoff

#to get the parameters we can simply use 
p = params(pot)

#then we can compute energies and forces

FluxEnergy(pot, X[1])
FluxForces(pot, X[1])

#and derivative according to parameters 
#since these are implicit there is a special notation
g = gradient(()->energy(pot, X[1]), p)
g[p[1]]
g[p[2]]

g = gradient(()->forces(pot, X[1]), params(fs_pot))
g[p[1]]

#we can then define a loss function
sqr(x) = x.^2 #to iterate twice
loss(at, EF) =  Flux.Losses.mse(FluxEnergy(pot, at), EF[1]) + sum(sum(sqr.(FluxForces(pot, at) - EF[2])))

#and derivate it!
g = gradient(()->loss(X[1], Y[1]), p)
g[p[1]]

#Once you have the gradients you can plug into any optimizer you want
# or you can create your own

#we provide a train function that plugs into flux and allows us to use their optimizers

(1,2,3,4) = train()

#--> explain the parameters and what it returns but don't run it

#but the AD framework is way more flexible than this
#let's go back to our model again, we can evaluate a cfg with it

model(cfg)

#then differentiate it w.r.t to parameters as expected

g = gradient(()->model(cfg), p)
g[p[1]]

#or w.r.t the configuration itself to get gradients

g = gradient(model, cfg)
g[1]

#this is what's behind the forces function
#and we can even go deeper and get a mixed derivative 

g = gradient(m -> gradient(m, cfg), p)
g[p[1]]

#The pullbacks for these derivatives (forces and mixed one)
#where implemented manually so that we leverage adjoint techniques
#like the ones described in PACE, for faster performance. 

#ACEnets is simply a wrapper to these functions. ACE.jl has now implemented
#derivatives to some of it's funcitons that allows for great generality.
#all one needs is to write a wrapper and a big computer. 