get_inputs(::Nothing) = nothing
function get_inputs(input_ids::Vector{Symbol})
    out = Array{Array{Float64}}(undef, 0)
    for input_id in input_ids
        if input_id in [:net3_input1, :net3_input2]
            order_jl, order_py = ["W", "H", "C"], ["C", "H", "W"]
        end
        # Ensure Julia indexing
        path = joinpath(@__DIR__, "..", "..", "assets", "inputs", "$(input_id).hdf5")
        __input_data = HDF5.h5read(path, "input1")
        map_input = _get_input_map(order_jl, order_py)
        _input_data = permutedims(__input_data, reverse(1:ndims(__input_data)))
        _input_data = _reshape_array(_input_data, map_input)
        push!(out, reshape(_input_data, (size(_input_data)..., 1)))
    end
    return out
end

function _get_input_map(order_jl, order_py)
    imap = zeros(Int64, length(order_jl))
    for i in eachindex(order_py)
        imap[i] = findfirst(x -> x == order_py[i], order_jl)
    end
    return collect(1:length(order_py)) .=> imap
end
