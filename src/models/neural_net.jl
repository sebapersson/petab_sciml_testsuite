function get_net_models(nets_info::Dict)
    out = Dict()
    for net_id in keys(nets_info)
        if net_id == :net1
            out[net_id] = get_net1()
        end
        if net_id == :net2
            out[net_id] = get_net2()
        end
        if net_id == :net3
            out[net_id] = get_net3()
        end
    end
    return out
end

function get_net1()
    rng = StableRNGs.StableRNG(1)
    nn_model = @compact(layer1=Dense(2, 5, Lux.tanh),
                        layer2=Dense(5, 5, Lux.tanh),
                        layer3=Dense(5, 1)) do x
        embed = layer1(x)
        embed = layer2(embed)
        out = layer3(embed)
        @return out
    end
    _, st = Lux.setup(rng, nn_model)
    return st, nn_model
end

function get_net2()
    rng = StableRNGs.StableRNG(1)
    nn_model = @compact(layer1=Dense(2, 5, Lux.relu),
                        layer2=Dense(5, 10, Lux.relu),
                        layer3=Dense(10, 1)) do x
        embed = layer1(x)
        embed = layer2(embed)
        out = layer3(embed)
        @return out
    end
    _, st = Lux.setup(rng, nn_model)
    return st, nn_model
end

function get_net3()
    rng = StableRNGs.StableRNG(1)
    nn_model = @compact(layer1=Conv((5, 5), 3=>1; cross_correlation = true),
                        layer2=FlattenLayer(),
                        layer3=Dense(36=>1, Lux.relu)) do x
        embed = layer1(x)
        embed = layer2(embed)
        out = layer3(embed)
        @return out
    end
    _, st = Lux.setup(rng, nn_model)
    return st, nn_model
end
