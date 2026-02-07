using ForwardDiff, OrdinaryDiffEqVerner, OptimizationOptimisers, Optimization,
    OptimizationLBFGSB, StableRNGs, Lux, ComponentArrays, StableRNGs

rng = StableRNGs.StableRNG(1)
nn_model = @compact(
    layer1 = Dense(4, 5, Lux.relu),
    layer2 = Dense(5, 10, Lux.relu),
    layer3 = Dense(10, 1)
) do x
    embed = layer1(x)
    embed = layer2(embed)
    out = layer3(embed)
    @return out
end |> f64

function lv_reference!(du, u, p, t)
    prey, predator = u
    alpha, delta, beta, gamma = 1.3, 1.8, 0.9, 0.8
    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * prey * predator - delta * predator # predator
    return nothing
end

ps, st = Lux.setup(rng, nn_model)
ps = ComponentArray(ps) |> f64

input_data1 = [[rand(rng) * 10.0, 1.0, 2.0, 3.0] for i in 1:1000]
input_data2 = [[rand(rng) * 10.0, 3.0, 2.0, 1.0] for i in 1:1000]
output_data1 = [0.9 for x in input_data1]
output_data2 = [1.0 for x in input_data2]

function my_loss(p, ::Any)
    loss1 = sum([(nn_model(input_data1[i], p, st)[1][1] - output_data1[i])^2 for i in eachindex(input_data1)])
    loss2 = sum([(nn_model(input_data2[i], p, st)[1][1] - output_data2[i])^2 for i in eachindex(input_data2)])
    return loss1 + loss2
end

my_loss(ps, Float64[])
p0 = deepcopy(ps)
f1 = SciMLBase.OptimizationFunction(my_loss, ADTypes.AutoForwardDiff())
prob1 = SciMLBase.OptimizationProblem(f1, p0, Float64[])
sol1 = solve(prob1, Adam(1e-3), maxiters = 10000)
my_loss(sol1.u, Float64[])

PEtabSciMLTestsuite.nn_ps_to_h5(nn_model, sol1.u, nothing, :net6, joinpath(@__DIR__, "net6_OBS2_ps.hdf5"))
