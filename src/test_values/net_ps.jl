function get_net_parameters(nn_models, nets_info::Dict)
    out = Vector{ComponentArray}(undef, 0)
    for (net_id, net_info) in nets_info
        _ps = _get_net_parameters(nn_models, net_id, net_info[:ps_file])
        push!(out, _ps)
    end
    return out
end

function _get_net_parameters(nn_models, net_id::Symbol, net_ps_id::String)
    rng = StableRNGs.StableRNG(1)
    path_ps = joinpath(@__DIR__, "..", "..", "assets", "parameters", net_ps_id)
    pnn = Lux.initialparameters(rng, nn_models[net_id][2]) |>
        ComponentArray |>
        f64
    set_ps_net!(pnn, path_ps, nn_models[net_id][2])
    return pnn
end
