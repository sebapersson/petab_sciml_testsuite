using DataFrames, PEtabSciMLTestsuite

# The PEtab problem
nets_info = Dict(:net1 => Dict(:ps_file => "net1_UDE1_ps.hdf5",
    :static => false))
llh_id = :UDE1
experiment_table_id = :Table1
condition_table_id = :Nothing
observable_table_id = :Table1
sbml_id = :lv_UDE1
petab_parameters_ids = [:alpha, :delta, :beta, :net1_layer1]
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(
    petabEntityId = [
        "net1_input1",
        "net1_input2",
        "net1_output1",
        "net1_ps",
        "net1_layer1"
    ],
    modelEntityId = [
        "net1.inputs[0][0]",
        "net1.inputs[0][1]",
        "net1.outputs[0][0]",
        "net1.parameters",
        "net1.parameters[layer1].bias"
    ])
hybridization_table = DataFrame(targetId = ["net1_input1", "net1_input2", "gamma"],
    targetValue = ["prey", "predator", "net1_output1"])
# Initialization info
initializations_info = Dict(:net1 => Dict(:what_change => "layer1.bias", :value => 0.0))

save_initialization_test_values(@__DIR__, nets_info, initializations_info)
create_petab_files(
    @__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids, experiment_table_id,
    condition_table_id, observable_table_id, mapping_table, hybridization_table
)
