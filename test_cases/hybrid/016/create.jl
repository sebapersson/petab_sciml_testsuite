using DataFrames, PEtabSciMLTestsuite

nets_info = Dict(
    :net1 =>
        Dict(
            :ps_file => "net1_UDE1_ps.hdf5",
            :static => false))
ode_id = :UDE1
llh_id = :UDE1
condition_table_id = :Table1
observable_table_id = :Table1
sbml_id = :lv_UDE1
estimate_net_parameters = false
petab_parameters_ids = [:alpha, :delta, :beta]
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(
    petabEntityId = ["net1_input1", "net1_input2", "net1_output1", "net1_ps_file"],
    modelEntityId = ["net1.inputs[0][0]", "net1.inputs[0][1]", "net1.outputs[0][0]", "net1.parameters"])
hybridization_table = DataFrame(
    targetId = ["net1_input1", "net1_input2", "gamma"],
    targetValue = ["prey", "predator", "net1_output1"])

save_test_values(@__DIR__, nets_info, ode_id, llh_id, petab_parameters_ids; estimate_net_parameters = false)
create_petab_files(@__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids, condition_table_id, observable_table_id, mapping_table, hybridization_table; estimate_net_parameters = false)
