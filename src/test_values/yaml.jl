function save_hybrid_yaml(llh::Float64, nets_info::Dict, estimate_net_parameters::Bool,
        dir_save; tol_llh = 1e-3, tol_grad_llh = 1e-1,
        tol_simulations = 1e-3)
    grad_files = Dict(:mech => "grad_mech.tsv")
    if estimate_net_parameters
        for net_id in keys(nets_info)
            grad_files[net_id] = "grad_$(net_id).hdf5"
        end
    end
    yaml_dict = Dict(:llh => llh, :tol_llh => tol_llh, :tol_grad_llh => tol_grad_llh,
        :tol_simulations => tol_simulations, :grad_files => grad_files,
        :simulation_files => ["simulations.tsv"])
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
