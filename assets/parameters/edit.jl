using HDF5

path_ref = joinpath(@__DIR__, "net1_OBS1_ps.hdf5")
h5_ref = HDF5.h5open(path_ref, "r")

h5_new = HDF5.h5open(joinpath(@__DIR__, "net5_OBS1_ps.hdf5"), "w")
g1_meta = HDF5.create_group(h5_new, "metadata")
g1_meta["perm"] = "row"

g_ps = HDF5.create_group(h5_new, "parameters")
g_net5 = HDF5.create_group(g_ps, "net5")

g_layer1 = HDF5.create_group(g_net5, "layer1")
g_layer1["bias"] = HDF5.read_dataset(h5_ref["parameters"]["net1"]["layer1"], "bias")
g_layer1["weight"] = HDF5.read_dataset(h5_ref["parameters"]["net1"]["layer1"], "weight")

g_layer2 = HDF5.create_group(g_net5, "layer2")
g_layer2["bias"] = HDF5.read_dataset(h5_ref["parameters"]["net1"]["layer2"], "bias")
g_layer2["weight"] = HDF5.read_dataset(h5_ref["parameters"]["net1"]["layer2"], "weight")

g_layer3 = HDF5.create_group(g_net5, "layer3")
g_layer3["bias"] = HDF5.read_dataset(h5_ref["parameters"]["net1"]["layer3"], "bias")
g_layer3["weight"] = HDF5.read_dataset(h5_ref["parameters"]["net1"]["layer3"], "weight")
close(h5_new)
close(h5_ref)
