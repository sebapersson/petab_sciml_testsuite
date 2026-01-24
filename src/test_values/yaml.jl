function save_hybrid_yaml(
        objective::Float64, nets_info::Dict, estimate_net_parameters::Bool,
        prior_id::Union{Symbol, Nothing}, dir_save; tol_objective = 1e-3, tol_grad = 1e-1,
        tol_simulations = 1e-3
    )
    grad_files = Dict(:mech => "grad_mech.tsv")
    if estimate_net_parameters
        for net_id in keys(nets_info)
            grad_files[net_id] = "grad_$(net_id).hdf5"
        end
    end
    yaml_dict = Dict(
        :tol_grad => tol_grad, :tol_simulations => tol_simulations,
        :grad_files => grad_files,
        :simulation_files => ["simulations.tsv"]
    )
    if isnothing(prior_id)
        yaml_dict[:llh] = objective
        yaml_dict[:tol_llh]  = tol_objective
    else
        yaml_dict[:log_posterior] = objective
        yaml_dict[:tol_log_posterior] = tol_objective

    end
    YAML.write_file(joinpath(dir_save, "solutions.yaml"), yaml_dict)
    return nothing
end

function save_initialization_yaml(initializations_info::Dict, dir_save)
    ps_files = Dict()
    for net_id in keys(initializations_info)
        ps_files[net_id] = "$(net_id)_ref.hdf5"
    end
    yaml_dict = Dict(:parameter_files => ps_files, :tol => 1e-3)
    YAML.write_file(joinpath(dir_save, "solutions.yaml"), yaml_dict)
    return nothing
end
