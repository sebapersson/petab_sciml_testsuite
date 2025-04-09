function save_petab_yaml(nets_info::Dict, dir_petab, input_ids::Union{Nothing, Vector{Symbol}})
    yaml_dict = Dict(
        :format_version => 2,
        :parameter_file => "parameters.tsv",
        :problems => [
            Dict(
                :condition_files => ["conditions.tsv"],
                :measurement_files => ["measurements.tsv"],
                :observable_files => ["observables.tsv"],
                :mapping_files => ["mapping.tsv"],
                :model_files => Dict(
                    :lv => Dict(
                        :language => "sbml",
                        :location => "lv.xml")))])

    ext = Dict(
        :hybridization_file => "hybridization.tsv",
        :neural_nets => Dict(),
        :array_files => Dict())
    for (net_id, net_info) in nets_info
        ext[:neural_nets][net_id] = Dict(
            :location => "$(net_id).yaml",
            :format => "YAML",
            :static => net_info[:static])
        file_id = Symbol("$(net_id)_ps_file")
        ext[:array_files][file_id] = Dict(
            :location => "$(net_id)_ps.hdf5",
            :format => "HDF5")
    end
    if !isnothing(input_ids)
        for i in eachindex(input_ids)
            file_id = "input_file$(i)"
            ext[:array_files][file_id] = Dict(
            :location => "input$i.hdf5",
            :format => "HDF5")
        end
    end

    yaml_dict[:extensions] = ext
    YAML.write_file(joinpath(dir_petab, "problem.yaml"), yaml_dict)
    return nothing
end
