using DataFrames, PEtabSciMLTestsuite

nets_info = Dict(:net1 => Dict(:ps_file => "net1_pre_ODE1_ps.hdf5",
    :static => true))
ode_id = :reference
llh_id = :pre_ODE1
condition_table_id = :Table1
observable_table_id = :Table1
sbml_id = :lv_reference
petab_parameters_ids = [:alpha, :delta, :beta, :net1_input_pre1, :net1_input_pre2]
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(
    petabEntityId = [
        "net1_input_pre1",
        "net1_input_pre2",
        "net1_output1",
        "net1_ps"
    ],
    modelEntityId = [
        "net1.inputs[0][0]",
        "net1.inputs[0][1]",
        "net1.outputs[0][0]",
        "net1.parameters"
    ])
hybridization_table = DataFrame(targetId = ["gamma"],
    targetValue = ["net1_output1"])

save_hybrid_test_values(@__DIR__, nets_info, ode_id, llh_id, petab_parameters_ids)
create_petab_files(@__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids,
    condition_table_id, observable_table_id, mapping_table,
    hybridization_table)
