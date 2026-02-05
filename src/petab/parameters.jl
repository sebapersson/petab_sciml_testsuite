const ALPHA = DataFrame(
    parameterId = "alpha",
    parameterScale = "lin",
    lowerBound = 0.0,
    upperBound = 15.0,
    nominalValue = 1.3,
    estimate = true
)

const DELTA = DataFrame(
    parameterId = "delta",
    parameterScale = "lin",
    lowerBound = 0.0,
    upperBound = 15.0,
    nominalValue = 1.8,
    estimate = true
)

const BETA = DataFrame(
    parameterId = "beta",
    parameterScale = "lin",
    lowerBound = 0.0,
    upperBound = 15.0,
    nominalValue = 0.9,
    estimate = true
)

const GAMMA = DataFrame(
    parameterId = "gamma",
    parameterScale = "lin",
    lowerBound = 0.0,
    upperBound = 15.0,
    nominalValue = 0.8,
    estimate = true
)

const NET1_INPUT_PRE1 = DataFrame(
    parameterId = "net1_input_pre1",
    parameterScale = "lin",
    lowerBound = "-inf",
    upperBound = "inf",
    nominalValue = 1.0,
    estimate = false
)

const NET1_INPUT_PRE2 = DataFrame(
    parameterId = "net1_input_pre2",
    parameterScale = "lin",
    lowerBound = "-inf",
    upperBound = "inf",
    nominalValue = 1.0,
    estimate = false
)

const NET1_INPUT_PRE2_EST = DataFrame(
    parameterId = "net1_input_pre2_est",
    parameterScale = "lin",
    lowerBound = "-inf",
    upperBound = "inf",
    nominalValue = 1.0,
    estimate = true
)

const NET2_INPUT_PRE1 = DataFrame(
    parameterId = "net2_input_pre1",
    parameterScale = "lin",
    lowerBound = "-inf",
    upperBound = "inf",
    nominalValue = 2.0,
    estimate = false
)

const NET2_INPUT_PRE2 = DataFrame(
    parameterId = "net2_input_pre2",
    parameterScale = "lin",
    lowerBound = "-inf",
    upperBound = "inf",
    nominalValue = 2.0,
    estimate = false
)

const NET1_LAYER1 = DataFrame(
    parameterId = "net1_layer1",
    parameterScale = "lin",
    lowerBound = "-inf",
    upperBound = "inf",
    nominalValue = 0.0,
    estimate = true
)

const NET1_LAYER1_FREEZE = DataFrame(
    parameterId = "net1_layer1",
    parameterScale = "lin",
    lowerBound = "-inf",
    upperBound = "inf",
    nominalValue = "array",
    estimate = false
)

const NET1_LAYER1_WEIGHT_FREEZE = DataFrame(
    parameterId = "net1_layer1_weight",
    parameterScale = "lin",
    lowerBound = "-inf",
    upperBound = "inf",
    nominalValue = "array",
    estimate = false
)

const NET1_LAYER1_PRIOR = DataFrame(
    parameterId = "net1_layer1_prior",
    parameterScale = "lin",
    lowerBound = "-inf",
    upperBound = "inf",
    nominalValue = "array",
    estimate = true,
    priorDistribution = "normal",
    priorParameters = "0.0;2.0"
)

const NET1_LAYER1_WEIGHT_PRIOR = DataFrame(
    parameterId = "net1_layer1_weight_prior",
    parameterScale = "lin",
    lowerBound = "-inf",
    upperBound = "inf",
    nominalValue = "array",
    estimate = true,
    priorDistribution = "normal",
    priorParameters = "0.0;2.0"
)

function save_parameters_table(
        petab_parameters_ids::Vector{Symbol}, nets_info::Dict, estimate_net_parameters::Bool,
        dir_petab
    )
    df_mech = DataFrame()
    for id in petab_parameters_ids
        df_mech = vcat(df_mech, _get_parameter_info(id, :DataFrame), cols = :union)
    end

    _net_ids = collect(keys(nets_info))
    net_ids = @. string(_net_ids) * "_ps"
    df_nn = DataFrame(
        parameterId = net_ids, parameterScale = :lin, lowerBound = "-inf",
        upperBound = "inf", nominalValue = "array", estimate = Int(estimate_net_parameters)
    )

    df_save = vcat(df_mech, df_nn, cols = :union)
    CSV.write(joinpath(dir_petab, "parameters.tsv"), df_save, delim = '\t')
    return nothing
end

function save_net_parameters(nets_info::Dict, dir_petab)
    for (net_id, net_info) in nets_info
        src = joinpath(@__DIR__, "..", "..", "assets", "parameters", net_info[:ps_file])
        dst = joinpath(dir_petab, "$(net_id)_ps.hdf5")
        cp(src, dst; force = true)
    end
    return nothing
end

function _get_parameter_info(id::Symbol, what_return::Symbol)
    if id == :alpha
        info = ALPHA
    end
    if id == :delta
        info = DELTA
    end
    if id == :beta
        info = BETA
    end
    if id == :gamma
        info = GAMMA
    end
    if id == :net1_input_pre1
        info = NET1_INPUT_PRE1
    end
    if id == :net1_input_pre2
        info = NET1_INPUT_PRE2
    end
    if id == :net1_input_pre2_est
        info = NET1_INPUT_PRE2_EST
    end
    if id == :net2_input_pre1
        info = NET2_INPUT_PRE1
    end
    if id == :net2_input_pre2
        info = NET2_INPUT_PRE2
    end
    if id == :net1_layer1
        info = NET1_LAYER1
    end
    if id == :net1_layer1_freeze
        info = NET1_LAYER1_FREEZE
    end
    if id == :net1_layer1_weight_freeze
        info = NET1_LAYER1_WEIGHT_FREEZE
    end
    if id == :net1_layer1_prior
        info = NET1_LAYER1_PRIOR
    end
    if id == :net1_layer1_weight_prior
        info = NET1_LAYER1_WEIGHT_PRIOR
    end

    if what_return == :DataFrame
        return info
    elseif what_return == :value
        return info.nominalValue[1]
    end
    return Bool(info.estimate[1])
end
