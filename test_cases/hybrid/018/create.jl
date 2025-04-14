using DataFrames, PEtabSciMLTestsuite

nets_info = Dict(:net1 => Dict(:ps_file => "net1_OBS1_ps.hdf5",
                               :static => false))
ode_id = :reference
llh_id = :OBS1
condition_table_id = :Table1
observable_table_id = :Table2
sbml_id = :lv_reference
estimate_net_parameters = false
petab_parameters_ids = [:alpha, :delta, :beta, :gamma]
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(petabEntityId = [
                              "net1_input1",
                              "net1_input2",
                              "net1_output1",
                              "net1_ps"
                          ],
                          modelEntityId = [
                              "net1.inputs[0][0]",
                              "net1.inputs[0][1]",
                              "net1.outputs[0][0]",
                              "net1.parameters"
                          ])
hybridization_table = DataFrame(targetId = ["net1_input1", "net1_input2"],
                                targetValue = ["prey", "predator"])

save_test_values(@__DIR__, nets_info, ode_id, llh_id, petab_parameters_ids;
                 estimate_net_parameters = false)
create_petab_files(@__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids,
                   condition_table_id, observable_table_id, mapping_table,
                   hybridization_table; estimate_net_parameters = false)
