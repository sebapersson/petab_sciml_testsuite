save_net_inputs(::Nothing, ::Any) = nothing
function save_net_inputs(input_id::Symbol, dir_petab)
    src = joinpath(@__DIR__, "..", "..", "assets", "inputs", "$(input_id).hdf5")
    dst = joinpath(dir_petab, "$(input_id).hdf5")
    cp(src, dst; force = true)
    return nothing
end
