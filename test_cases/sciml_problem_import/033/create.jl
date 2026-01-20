using CSV, DataFrames, Distributions, PEtabSciMLTestsuite

nets_info = Dict(
    :net1 => Dict(:ps_file => "net1_pre_ODE1_ps.hdf5", :static => true)
)
ode_id = :reference
llh_id = :pre_ODE1
prior_id = :prior2
experiment_table_id = :Table1
condition_table_id = :Nothing
observable_table_id = :Table1
sbml_id = :lv_reference
petab_parameters_ids = [
    :alpha, :delta, :beta, :net1_input_pre1, :net1_input_pre2, :net1_layer1_prior]
priors = Dict(:net1 => Normal(0.0, 1.0))
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(
    petabEntityId = [
        "net1_input_pre1",
        "net1_input_pre2",
        "net1_output1",
        "net1_ps",
        "net1_layer1_prior",
    ],
    modelEntityId = [
        "net1.inputs[0][0]",
        "net1.inputs[0][1]",
        "net1.outputs[0][0]",
        "net1.parameters",
        "net1.parameters[layer1]"
    ])
hybridization_table = DataFrame(targetId = ["gamma"], targetValue = ["net1_output1"])

save_hybrid_test_values(
    @__DIR__, nets_info, ode_id, llh_id, petab_parameters_ids; prior_id = prior_id
)
create_petab_files(
    @__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids, experiment_table_id,
    condition_table_id, observable_table_id, mapping_table, hybridization_table
)

# Adding the prior to the parameters table
parameters_df = CSV.read(joinpath(@__DIR__, "petab", "parameters.tsv"), DataFrame;
    stringtype = String)
parameters_df[1:3, :priorDistribution] .= "uniform"
parameters_df[1:3, :priorParameters] .= "0.0;15.0"
parameters_df[7, :priorDistribution] = "normal"
parameters_df[7, :priorParameters] = "0.0;1.0"
CSV.write(joinpath(@__DIR__, "petab", "parameters.tsv"), parameters_df; delim = '\t')
