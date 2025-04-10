const OBSERVABLE_TABLE1 = DataFrame(observableId = ["prey_o", "predator_o"],
                                    observableFormula = ["prey", "predator"],
                                    noiseFormula = [0.05, 0.05],
                                    observableTransformation = ["lin", "lin"],
                                    noiseDistribution = ["normal", "normal"])

const OBSERVABLE_TABLE2 = DataFrame(observableId = ["prey_o", "predator_o"],
                                    observableFormula = ["net1_output1", "predator"],
                                    noiseFormula = [0.05, 0.05],
                                    observableTransformation = ["lin", "lin"],
                                    noiseDistribution = ["normal", "normal"])

const OBSERVABLE_TABLE3 = DataFrame(observableId = ["prey_o", "predator_o"],
                                    observableFormula = ["net1_output1", "net2_output2"],
                                    noiseFormula = [0.05, 0.05],
                                    observableTransformation = ["lin", "lin"],
                                    noiseDistribution = ["normal", "normal"])

const OBSERVABLE_TABLE4 = DataFrame(observableId = ["prey_o", "predator_o"],
                                    observableFormula = ["prey", "net2_output2"],
                                    noiseFormula = [0.05, 0.05],
                                    observableTransformation = ["lin", "lin"],
                                    noiseDistribution = ["normal", "normal"])

function save_observables_table(observable_table_id::Symbol, dir_petab)::Nothing
    if observable_table_id == :Table1
        observables = OBSERVABLE_TABLE1
    end
    if observable_table_id == :Table2
        observables = OBSERVABLE_TABLE2
    end
    if observable_table_id == :Table3
        observables = OBSERVABLE_TABLE3
    end
    if observable_table_id == :Table4
        observables = OBSERVABLE_TABLE4
    end
    CSV.write(joinpath(dir_petab, "observables.tsv"), observables, delim = '\t')
    return nothing
end
