using DataFrames, PEtabSciMLTestsuite

nets_info = Dict(:net3 => Dict(:ps_file => "net3_pre_ODE2_ps.hdf5",
    :static => true))
ode_id = :reference
llh_id = :pre_ODE8
condition_table_id = :Table3
observable_table_id = :Table1
sbml_id = :lv_reference
input_ids = [:net3_input1, :net3_input2]
petab_parameters_ids = [:alpha, :delta, :beta]
# Mapping and hybridization generally differ between tests and must thus be hand-coded
mapping_table = DataFrame(
    petabEntityId = ["net3_input", "net3_output1", "net3_ps"],
    modelEntityId = ["net3.inputs[0]", "net3.outputs[0][0]", "net3.parameters"])
hybridization_table = DataFrame(targetId = ["gamma"],
    targetValue = ["net3_output1"])

save_hybrid_test_values(@__DIR__, nets_info, ode_id, llh_id, petab_parameters_ids;
    input_ids = input_ids)
create_petab_files(@__DIR__, nets_info, sbml_id, llh_id, petab_parameters_ids,
    condition_table_id, observable_table_id, mapping_table,
    hybridization_table; input_ids = input_ids)
