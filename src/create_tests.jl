function save_test_values(dir_save, nets_info::Dict, ode_id::Symbol, llh_id::Symbol,
                          petab_parameters_ids::Vector{Symbol};
                          estimate_net_parameters::Bool = true,
                          input_ids::Union{Nothing, Vector{Symbol}} = nothing)::Nothing
    nn_models = get_net_models(nets_info)
    measurements = get_measurements(llh_id)
    ode_problem = get_odeproblem(ode_id, nn_models)
    net_parameters = get_net_parameters(nn_models, nets_info)

    x = get_x(petab_parameters_ids, net_parameters, nets_info)
    inputs = get_inputs(input_ids)
    compute_llh = get_llh(llh_id, nn_models, ode_problem, measurements, inputs)
    llh = compute_llh(x)
    save_solution_yaml(llh, nets_info, estimate_net_parameters, dir_save)
    save_grad(x, compute_llh, nn_models, estimate_net_parameters, dir_save)
    save_simulations(x, llh_id, ode_problem, nn_models, measurements, inputs, dir_save)
    return nothing
end

function create_petab_files(dir_test, nets_info::Dict, sbml_id::Symbol, llh_id::Symbol,
                            petab_parameters_ids::Vector{Symbol},
                            condition_table_id::Symbol, observable_table_id::Symbol,
                            mapping_table::DataFrame, hybridization_table::DataFrame;
                            estimate_net_parameters::Bool = true,
                            input_ids::Union{Nothing, Vector{Symbol}} = nothing)::Nothing
    dir_petab = joinpath(dir_test, "petab")
    save_sbml(sbml_id, dir_petab)
    save_parameters_table(petab_parameters_ids, nets_info, estimate_net_parameters,
                          dir_petab)
    save_measurements_table(llh_id, dir_petab)
    save_conditions_table(condition_table_id, dir_petab)
    save_observables_table(observable_table_id, dir_petab)
    save_net_parameters(nets_info, dir_petab)
    save_net_yaml(nets_info, dir_petab)
    save_petab_yaml(nets_info, dir_petab, input_ids)
    CSV.write(joinpath(dir_petab, "mapping.tsv"), mapping_table, delim = '\t')
    CSV.write(joinpath(dir_petab, "hybridization.tsv"), hybridization_table, delim = '\t')
    return nothing
end
