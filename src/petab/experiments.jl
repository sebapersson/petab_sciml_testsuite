const EXP_TABLE1 = DataFrame(experimentId = "e1", time = 0.0, conditionId = missing)
const EXP_TABLE2 = DataFrame(
    experimentId = ["e1", "e2"],
    time = [0.0, 0.0],
    conditionId = ["cond1", "cond2"]
)

function save_experiments_table(experiment_table_id::Symbol, dir_petab)::Nothing
    if experiment_table_id == :Table1
        experiments = EXP_TABLE1
    elseif experiment_table_id == :Table2
        experiments = EXP_TABLE2
    end
    CSV.write(joinpath(dir_petab, "experiments.tsv"), experiments, delim = '\t')
    return nothing
end
