using DataFrames, PEtabSciMLTestsuite

nets_info = Dict(:net1 => Dict(:ps_file => "net1_UDE1_ps.hdf5",
        :static => false),
    :net2 => Dict(:ps_file => "net2_OBS1_ps.hdf5",
        :static => false))
ode_id = :UDE1
llh_id = :COMBO2
condition_table_id = :Table1
observable_table_id = :Table4
sbml_id = :lv_UDE1
petab_parameters_ids = [:alpha, :delta, :beta]
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(
    petabEntityId = [
        "net1_input1",
        "net1_input2",
        "net1_output1",
        "net1_ps",
        "net2_input1",
        "net2_input2",
        "net2_output1",
        "net2_ps"
    ],
    modelEntityId = [
        "net1.inputs[0][0]",
        "net1.inputs[0][1]",
        "net1.outputs[0][0]",
        "net1.parameters",
        "net2.inputs[0][0]",
        "net2.inputs[0][1]",
        "net2.outputs[0][0]",
        "net2.parameters"
    ])
hybridization_table = DataFrame(
    targetId = [
        "net1_input1",
        "net1_input2",
        "gamma",
        "net2_input1",
        "net2_input2"
    ],
    targetValue = [
        "prey",
        "predator",
        "net1_output1",
        "alpha",
        "predator"
    ])

save_test_values(@__DIR__, nets_info, ode_id, llh_id, petab_parameters_ids)
create_petab_files(@__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids,
    condition_table_id, observable_table_id, mapping_table,
    hybridization_table)
