function save_hybrid_test_values(dir_save, nets_info::Dict, ode_id::Symbol, llh_id::Symbol,
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
    save_hybrid_yaml(llh, nets_info, estimate_net_parameters, dir_save)
    save_grad(x, compute_llh, nn_models, estimate_net_parameters, dir_save)
    save_simulations(x, llh_id, ode_problem, nn_models, measurements, inputs, dir_save)
    return nothing
end

function save_initialization_test_values(
        dir_save, nets_info::Dict, initializations_info::Dict)
    nn_models = get_net_models(nets_info)
    for (net_id, initialization_info) in initializations_info
        net_ps = _get_net_parameters(nn_models, net_id, nets_info[net_id][:ps_file])
        what_change = split(initialization_info[:what_change], '.')
        if length(what_change) == 1
            net_ps[Symbol(what_change[1])] .= initialization_info[:value]
        else
            layer_ps = @view net_ps[Symbol(what_change[1])]
            layer_ps[Symbol(what_change[2])] .= initialization_info[:value]
        end
        nn_ps_to_h5(nn_models[net_id][2], net_ps, joinpath(dir_save, "$(net_id)_ref.hdf5"))
    end

    save_initialization_yaml(initializations_info, dir_save)
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

function create_hybrid_tests()
    dir_tests = joinpath(@__DIR__, "..", "test_cases", "hybrid")
    test_cases = filter(x -> x != "README.md", readdir(dir_tests))
    for test_case in test_cases
        @info "Hybrid test-case $(test_case)"
        include(joinpath(dir_tests, test_case, "create.jl"))
    end
end

function create_initialization_tests()
    dir_tests = joinpath(@__DIR__, "..", "test_cases", "initialization")
    test_cases = filter(x -> x != "README.md", readdir(dir_tests))
    for test_case in test_cases
        @info "Initialization test-case $(test_case)"
        include(joinpath(dir_tests, test_case, "create.jl"))
    end
end

function create_net_import_tests()
    dir_tests = joinpath(@__DIR__, "..", "test_cases", "net_import")
    test_cases = filter(x -> x != "README.md", readdir(dir_tests))
    for test_case in test_cases
        @info "Net-import test-case $(test_case)"
        include(joinpath(dir_tests, test_case, "create_testdata", "net.jl"))
    end
end
