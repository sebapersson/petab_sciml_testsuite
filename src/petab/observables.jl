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
    observableFormula = ["net1_output1", "net2_output1"],
    noiseFormula = [0.05, 0.05],
    observableTransformation = ["lin", "lin"],
    noiseDistribution = ["normal", "normal"])

const OBSERVABLE_TABLE4 = DataFrame(observableId = ["prey_o", "predator_o"],
    observableFormula = ["prey", "net2_output1"],
    noiseFormula = [0.05, 0.05],
    observableTransformation = ["lin", "lin"],
    noiseDistribution = ["normal", "normal"])

const OBSERVABLE_TABLE5 = DataFrame(observableId = ["prey_o", "predator_o"],
    observableFormula = ["net4_output1", "net4_output2"],
    noiseFormula = [0.05, 0.05],
    observableTransformation = ["lin", "lin"],
    noiseDistribution = ["normal", "normal"])

const OBSERVABLE_TABLE6 = DataFrame(observableId = ["prey_o", "predator_o"],
    observableFormula = ["net4_output1", "predator"],
    noiseFormula = [0.05, 0.05],
    observableTransformation = ["lin", "lin"],
    noiseDistribution = ["normal", "normal"])

const OBSERVABLE_TABLE7 = DataFrame(observableId = ["prey_o", "predator_o"],
    observableFormula = ["net5_output1", "predator"],
    noiseFormula = [0.05, 0.05],
    observableTransformation = ["lin", "lin"],
    noiseDistribution = ["normal", "normal"])

const OBSERVABLE_TABLES = Dict(:Table1 => OBSERVABLE_TABLE1, :Table2 => OBSERVABLE_TABLE2, :Table3 => OBSERVABLE_TABLE3, :Table4 => OBSERVABLE_TABLE4, :Table5 => OBSERVABLE_TABLE5, :Table6 => OBSERVABLE_TABLE6, :Table7 => OBSERVABLE_TABLE7)

function save_observables_table(observable_table_id::Symbol, dir_petab)::Nothing
    observables = OBSERVABLE_TABLES[observable_table_id]
    CSV.write(joinpath(dir_petab, "observables.tsv"), observables, delim = '\t')
    return nothing
end
