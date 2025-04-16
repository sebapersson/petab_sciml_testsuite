using DataFrames, PEtabSciMLTestsuite

nets_info = Dict(:net4 => Dict(:ps_file => "net4_pre_ODE1_ps.hdf5", :static => true))
ode_id = :reference
llh_id = :pre_ODE9
condition_table_id = :Table1
observable_table_id = :Table1
sbml_id = :lv_reference
petab_parameters_ids = [:delta, :beta, :net1_input_pre1, :net1_input_pre2]
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(
    petabEntityId = [
        "net1_input_pre1",
        "net1_input_pre2",
        "net4_output1",
        "net4_output2",
        "net4_ps"
    ],
    modelEntityId = [
        "net4.inputs[0][0]",
        "net4.inputs[0][1]",
        "net4.outputs[0][0]",
        "net4.outputs[0][1]",
        "net4.parameters"
    ])
hybridization_table = DataFrame(
    targetId = [
        "net4_output1",
        "net4_output2"
    ],
    targetValue = [
        "alpha",
        "gamma"
    ])

save_hybrid_test_values(@__DIR__, nets_info, ode_id, llh_id, petab_parameters_ids)
create_petab_files(@__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids,
    condition_table_id, observable_table_id, mapping_table,
    hybridization_table)
