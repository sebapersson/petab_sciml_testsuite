function save_net_yaml(nets_info::Dict, dir_petab)::Nothing
    for net_id in keys(nets_info)
        src = joinpath(@__DIR__, "..", "..", "assets", "net_yaml", "$(net_id).yaml")
        dst = joinpath(dir_petab, "$(net_id).yaml")
        cp(src, dst; force = true)
    end
    return nothing
end
