const COND_TABLE1 = DataFrame(conditionId = "cond1")
const COND_TABLE2 = DataFrame(conditionId = ["cond1", "cond2"],
    net1_input1 = ["10.0", "net1_input_pre1"],
    net1_input2 = ["20.0", "net1_input_pre2"])
const COND_TABLE3 = DataFrame(conditionId = ["cond1", "cond2"],
    net3_input = ["input_file1", "input_file2"])

function save_conditions_table(condition_table_id::Symbol, dir_petab)::Nothing
    if condition_table_id == :Table1
        conditions = COND_TABLE1
    end
    if condition_table_id == :Table2
        conditions = COND_TABLE2
    end
    if condition_table_id == :Table3
        conditions = COND_TABLE3
    end
    CSV.write(joinpath(dir_petab, "conditions.tsv"), conditions, delim = '\t')
    return nothing
end
