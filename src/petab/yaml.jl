function save_petab_yaml(nets_info::Dict, dir_petab,
        input_file_id::Union{Nothing, Symbol})
    yaml_dict = Dict(:format_version => 2,
        :parameter_file => "parameters.tsv",
        :problems => [
            Dict(:condition_files => ["conditions.tsv"],
            :measurement_files => ["measurements.tsv"],
            :observable_files => ["observables.tsv"],
            :mapping_files => ["mapping.tsv"],
            :model_files => Dict(:lv => Dict(:language => "sbml",
                :location => "lv.xml")))])

    ext = Dict(:hybridization_files => ["hybridization.tsv"],
        :neural_nets => Dict(),
        :array_files => String[])
    for (net_id, net_info) in nets_info
        ext[:neural_nets][net_id] = Dict(:location => "$(net_id).yaml",
            :format => "YAML",
            :static => net_info[:static])
        push!(ext[:array_files], "$(net_id)_ps.hdf5")
    end
    if !isnothing(input_file_id)
        push!(ext[:array_files], "$(input_file_id).hdf5")
    end

    yaml_dict[:extensions] = Dict(:sciml => ext)
    YAML.write_file(joinpath(dir_petab, "problem.yaml"), yaml_dict)
    return nothing
end
