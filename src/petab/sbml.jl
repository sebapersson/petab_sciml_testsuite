function save_sbml(sbml_id::Symbol, dir_petab)
    src = joinpath(@__DIR__, "..", "..", "assets", "sbml", "$(sbml_id).xml")
    dst = joinpath(dir_petab, "lv.xml")
    cp(src, dst; force = true)
    return nothing
end
