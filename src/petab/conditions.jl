const COND_TABLE1 = DataFrame(
    conditionId = ["cond1", "cond2", "cond1", "cond2"],
    targetId = ["net1_input1", "net1_input1", "net1_input2", "net1_input2"],
    targetValue = ["10.0", "net1_input_pre1", "20.0", "net1_input_pre2"]
)
const COND_TABLE2 = DataFrame(
    conditionId = ["cond1", "cond2"]
)

function save_conditions_table(condition_table_id::Symbol, dir_petab)::Nothing
    if condition_table_id == :Nothing
        return nothing
    end

    if condition_table_id == :Table1
        conditions = COND_TABLE1
    elseif condition_table_id == :Table2
        conditions = COND_TABLE2
    end
    CSV.write(joinpath(dir_petab, "conditions.tsv"), conditions, delim = '\t')
    return nothing
end
