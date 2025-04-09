using StableRNGs

function save_measurements_table(llh_id::Symbol, dir_petab)
    measurements = get_measurements(llh_id)
    CSV.write(joinpath(dir_petab, "measurements.tsv"), measurements, delim = '\t')
    return nothing
end

function get_measurements(llh_id::Symbol)::DataFrame
    rng = StableRNGs.StableRNG(1)
    oprob_reference = get_odeproblem(:reference, Dict())
    if llh_id in [:pre_ODE2, :pre_ODE8]
        p1 = (alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
        p2 = (alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 1.0)
        oprob1 = remake(oprob_reference, p = p1)
        oprob2 = remake(oprob_reference, p = p2)
        sol1 = solve(oprob1, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = 1:1:10)
        sol2 = solve(oprob2, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = 1:1:10)
        df1 = DataFrame(observableId = "prey_o", simulationConditionId = "cond1", measurement = sol1[1, :] + randn(rng, 10) .* 0.05, time = sol1.t)
        df2 = DataFrame(observableId = "predator_o", simulationConditionId = "cond1", measurement = sol1[2, :] + randn(rng, 10) .* 0.05, time = sol1.t)
        df3 = DataFrame(observableId = "prey_o", simulationConditionId = "cond2", measurement = sol2[1, :] + randn(rng, 10) .* 0.05, time = sol2.t)
        df4 = DataFrame(observableId = "predator_o", simulationConditionId = "cond2", measurement = sol2[2, :] + randn(rng, 10) .* 0.05, time = sol2.t)
        return vcat(df1, df2, df3, df4)
    end

    sol = solve(oprob_reference, Vern9(), abstol = 1e-9, reltol = 1e-9, saveat = 1:1:10)
    df1 = DataFrame(observableId = "prey_o",
                    simulationConditionId = "cond1",
                    measurement = sol[1, :] + randn(rng, 10) .* 0.05,
                    time = sol.t)
    df2 = DataFrame(observableId = "predator_o",
                    simulationConditionId = "cond1",
                    measurement = sol[2, :] + randn(rng, 10) .* 0.05,
                    time = sol.t)
    return vcat(df1, df2)
end
