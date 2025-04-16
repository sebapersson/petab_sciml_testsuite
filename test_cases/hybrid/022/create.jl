using DataFrames, PEtabSciMLTestsuite

nets_info = Dict(:net4 => Dict(:ps_file => "net4_COMBO1_ps.hdf5", :static => false))
ode_id = :UDE1
llh_id = :COMBO4
condition_table_id = :Table1
observable_table_id = :Table4
sbml_id = :lv_UDE1
petab_parameters_ids = [:alpha, :delta, :beta]
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(
    petabEntityId = [
        "net4_input1",
        "net4_input2",
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
        "net4_input1",
        "net4_input2",
        "gamma"
    ],
    targetValue = [
        "prey",
        "predator",
        "net4_output2"
    ])

save_hybrid_test_values(@__DIR__, nets_info, ode_id, llh_id, petab_parameters_ids)
create_petab_files(@__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids,
    condition_table_id, observable_table_id, mapping_table,
    hybridization_table)
