using DataFrames, PEtabSciMLTestsuite

nets_info = Dict(
    :net6 => Dict(
        :ps_file => "net6_UDE1_ps.hdf5",
        :pre_initialization => false
    )
)
ode_id = :UDE5
llh_id = :UDE4
experiment_table_id = :Table1
condition_table_id = :Nothing
observable_table_id = :Table1
sbml_id = :lv_reference
input_id = :net6_input1
petab_parameters_ids = [:alpha, :delta, :beta]
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(
    petabEntityId = [
        "net6_input1",
        "net6_input2",
        "net6_output1",
        "net6_ps",
    ],
    modelEntityId = [
        "net6.inputs[0][0]",
        "net6.inputs[1]",
        "net6.outputs[0][0]",
        "net6.parameters",
    ]
)
hybridization_table = DataFrame(
    targetId = ["net6_input1", "net6_input2", "gamma"],
    targetValue = ["prey", "ARRAY_FILE", "net6_output1"]
)

save_hybrid_test_values(@__DIR__, nets_info, ode_id, llh_id, petab_parameters_ids)
create_petab_files(
    @__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids, experiment_table_id,
    condition_table_id, observable_table_id, mapping_table, hybridization_table,
    input_file_id = input_id
)
