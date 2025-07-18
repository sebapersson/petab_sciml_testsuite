get_inputs(::Nothing) = nothing
function get_inputs(input_id::Symbol)
    out = Array{Array{Float64}}(undef, 0)
    if input_id in [:net3_input1, :net3_input2]
        order_jl, order_py = ["W", "H", "C"], ["C", "H", "W"]
        map_input = _get_input_map(order_jl, order_py)
    end

    path = joinpath(@__DIR__, "..", "..", "assets", "inputs", "$(input_id).hdf5")
    input_file = HDF5.h5open(path, "r")
    for i in keys(input_file["inputs"]["input0"])
        _input_data = HDF5.read_dataset(input_file["inputs"]["input0"][i], "data")
        _input_data = permutedims(_input_data, reverse(1:ndims(_input_data)))
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
