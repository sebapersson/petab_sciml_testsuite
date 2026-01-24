using DataFrames, PEtabSciMLTestsuite

nets_info = Dict(
    :net5 => Dict(
        :ps_file => "net5_UDE1_ps.hdf5",
        :static => false
    )
)
ode_id = :UDE1
llh_id = :UDE1
experiment_table_id = :Table1
condition_table_id = :Nothing
observable_table_id = :Table1
sbml_id = :lv_UDE1
petab_parameters_ids = [:alpha, :delta, :beta]
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(
    petabEntityId = [
        "net5_arg0",
        "net5_arg1",
        "net5_output1",
        "net5_ps",
    ],
    modelEntityId = [
        "net5.inputs[0][0]",
        "net5.inputs[1][0]",
        "net5.outputs[0][0]",
        "net5.parameters",
    ]
)
hybridization_table = DataFrame(
    targetId = ["net5_arg0", "net5_arg1", "gamma"],
    targetValue = ["prey", "predator", "net5_output1"]
)

save_hybrid_test_values(@__DIR__, nets_info, ode_id, llh_id, petab_parameters_ids)
create_petab_files(
    @__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids, experiment_table_id,
    condition_table_id, observable_table_id, mapping_table, hybridization_table
)
