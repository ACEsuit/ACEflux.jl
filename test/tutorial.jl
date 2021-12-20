using ACE, ACEgnns, Zygote, Flux, ACE, StaticArrays, IPFitting, JuLIP

using Zygote: gradient


# I have loaded the Si data set here 

all_Si = IPFitting.Data.read_xyz("G:/My Drive/documents/UBC/Julia Codes/silicon/Si.xyz", energy_key="dft_energy", force_key="dft_force");
data = filter(at -> configtype(at) == "dia", all_Si); #get diamon configurations

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

linace = Linear_ACE(3, 7, 2); #cor order, max deg, num of properties

cfg = ACE.ACEConfig([ACE.State(rr=rand(SVector{3, Float64})) for _ = 1:10]);
linace(cfg)

fieldnames(typeof(linace))
linace.weight

#this is the fundamental object. For example if we wanted a FS model we simply
#pass this properties to the FS function

FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10

fs_model = x -> FS(linace(x))

#chain provides an easier way to do this

fs_model = Chain(linace, GenLayer(FS));

#Flux provides a nice function to retrieve the parameters
params(fs_model)

#and one can even go more general

model = Chain(linace, Dense(2,7), Dense(7,2), GenLayer(FS), sum);

#where now we have parameters we want to train in the Dense layers too
params(model)[1]
params(model)[2]
params(model)[3]

p = params(model)

model(cfg)

#now we define a potential, simply expand it with a cutoff radius

pot = FluxPotential(model, 6.0); #model, cutoff

pot(cfg)

#then we can compute energies and forces

energy(pot, X[1])
forces(pot, X[1])

#and derivative according to parameters 
#since these are implicit there is a special notation
g = gradient(()->energy(pot, X[1]), p)
g[p[1]]
g[p[2]]

g = gradient(()->forces(pot, X[1]), p)
g[p[1]]

#we can then define a loss function
sqr(x) = x.^2 #to iterate twice
loss(at, EF) =  Flux.Losses.mse(energy(pot, at), EF[1]) + sum(sum(sqr.(forces(pot, at) - EF[2])))

#and derivate it!
g = gradient(()->loss(X[1], Y[1]), p)
g[p[1]]

#Once you have the gradients you can plug into any optimizer you want
# or you can create your own

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

g = gradient(()->sum(gradient(model, cfg)), p)


g = gradient(m -> gradient(m, cfg), p) #figure out how to do this call
g[p[1]]

#The pullbacks for these derivatives (forces and mixed one)
#where implemented manually so that we leverage adjoint techniques
#like the ones described in PACE, for faster performance. 

#ACEnets is simply a wrapper to these functions. ACE.jl has now implemented
#derivatives to some of it's funcitons that allows for great generality.
#all one needs is to write a wrapper and a big computer. 




using FluxOptTools














@everywhere begin
   cfg = ACE.ACEConfig([ACE.State(rr=rand(SVector{3, Float64})) for _ = 1:10])
   
   FS(ϕ) = ϕ[1] + sqrt(abs(ϕ[2]) + 1/100) - 1/10
   
   fs_model = Chain(Linear_ACE(3, 4, 2), GenLayer(FS), sum)
   
   end
   
   fs_model(cfg)
   
   g = gradient(()->fs_model(cfg), Flux.params(fs_model))
   
   g = gradient(fs_model, cfg)[1]
   
   
   g[Flux.params(fs_model)[1]]