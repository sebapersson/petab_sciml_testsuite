using HDF5

path1 = joinpath(@__DIR__, "net3_input2.hdf5")
path2 = joinpath(@__DIR__, "tmp.hdf5")
mv(path1, path2)
mv(path2, path1)

data = HDF5.h5open(path2, "r")

data_new = HDF5.h5open(path1, "w")
data_cond1 = HDF5.read_dataset(data["inputs"]["input0"]["0"], "data")
data_cond2 = HDF5.read_dataset(data["inputs"]["input0"]["1"], "data")

g_meta = HDF5.create_group(data_new, "metadata")
g_meta["perm"] = "row"

g_inputs = HDF5.create_group(data_new, "inputs")
g_input0 = HDF5.create_group(g_inputs, "input0")
g_input0["0"] = data_cond1
g_input0["cond2"] = data_cond2
