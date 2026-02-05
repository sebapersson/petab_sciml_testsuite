function save_petab_yaml(
        nets_info::Dict, dir_petab, input_file_id::Union{Nothing, Symbol},
        condition_table_id::Symbol
    )
    yaml_dict = Dict(
        :format_version => 2,
        :measurement_files => ["measurements.tsv"],
        :experiment_files => ["experiments.tsv"],
        :observable_files => ["observables.tsv"],
        :parameter_files => ["parameters.tsv"],
        :mapping_files => ["mapping.tsv"],
        :model_files => Dict(
            :lv => Dict(
                :language => "sbml",
                :location => "lv.xml"
            )
        )
    )

    if condition_table_id != :Nothing
        yaml_dict[:condition_files] = ["conditions.tsv"]
    end

    ext = Dict(
        :hybridization_files => ["hybridization.tsv"],
        :neural_nets => Dict(),
        :array_files => String[]
    )
    for (net_id, net_info) in nets_info
        ext[:neural_nets][net_id] = Dict(
            :location => "$(net_id).yaml",
            :format => "YAML",
            :pre_initialization => net_info[:pre_initialization]
        )
        push!(ext[:array_files], "$(net_id)_ps.hdf5")
    end
    if !isnothing(input_file_id)
        push!(ext[:array_files], "$(input_file_id).hdf5")
    end

    yaml_dict[:extensions] = Dict(:sciml => ext)
    YAML.write_file(joinpath(dir_petab, "problem.yaml"), yaml_dict)
    return nothing
end
