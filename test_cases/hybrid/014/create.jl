using DataFrames, PEtabSciMLTestsuite

nets_info = Dict(:net3 => Dict(:ps_file => "net3_pre_ODE1_ps.hdf5",
                               :static => true))
ode_id = :reference
llh_id = :pre_ODE7
condition_table_id = :Table1
observable_table_id = :Table1
sbml_id = :lv_reference
input_ids = [:net3_input1]
petab_parameters_ids = [:alpha, :delta, :beta]
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(petabEntityId = ["input_file1", "net1_output1", "net1_ps_file"],
                          modelEntityId = [
                              "net1.inputs[0]",
                              "net1.outputs[0][0]",
                              "net1.parameters"
                          ])
hybridization_table = DataFrame(targetId = ["gamma"],
                                targetValue = ["net1_output1"])

save_test_values(@__DIR__, nets_info, ode_id, llh_id, petab_parameters_ids;
                 input_ids = input_ids)
create_petab_files(@__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids,
                   condition_table_id, observable_table_id, mapping_table,
                   hybridization_table; input_ids = input_ids)
