function save_net_inputs(input_ids::Vector{Symbol}, dir_petab)
    for (i, input_id) in pairs(input_ids)
        src = joinpath(@__DIR__, "..", "..", "assets", "inputs", "$(input_id).hdf5")
        dst = joinpath(dir_petab, "input$i.hdf5")
        cp(src, dst; force = true)
    end
    return nothing
end
