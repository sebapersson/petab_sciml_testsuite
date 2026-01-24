function get_llh(
        llh_id::Symbol, nn_models, oprob::ODEProblem, measurements::DataFrame, inputs
    )::Function
    if llh_id == :UDE1
        llh = let _oprob = oprob, _measurements = measurements
            (x) -> llh_UDE(x, _oprob, _measurements)
        end
    end
    if llh_id == :UDE2
        llh = let _oprob = oprob, _measurements = measurements
            (x) -> llh_UDE(x, _oprob, _measurements)
        end
    end
    if llh_id == :UDE3
        llh = let _oprob = oprob, _measurements = measurements
            (x) -> llh_UDE(x, _oprob, _measurements)
        end
    end
    if llh_id == :pre_ODE1
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_pre_ODE1(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :pre_ODE2
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_pre_ODE2(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :pre_ODE3
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_pre_ODE3(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :pre_ODE4
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_pre_ODE4(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :pre_ODE5
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_pre_ODE5(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :pre_ODE6
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_pre_ODE6(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :pre_ODE7
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_pre_ODE7(x, _oprob, _nn_models, _measurements, inputs)
        end
    end
    if llh_id == :pre_ODE8
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_pre_ODE8(x, _oprob, _nn_models, _measurements, inputs)
        end
    end
    if llh_id == :pre_ODE9
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_pre_ODE9(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :OBS1
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_OBS1(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :OBS2
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_OBS2(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :OBS3
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_OBS3(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :COMBO1
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_COMBO1(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :COMBO2
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_COMBO2(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :COMBO3
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_COMBO3(x, _oprob, _nn_models, _measurements)
        end
    end
    if llh_id == :COMBO4
        llh = let _nn_models = nn_models, _oprob = oprob, _measurements = measurements
            (x) -> llh_COMBO4(x, _oprob, _nn_models, _measurements)
        end
    end
    return llh
end

function llh_UDE(x, oprob::ODEProblem, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    _oprob = remake(oprob, p = x)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh1(sol, mprey, mpredator)
end

function llh_pre_ODE1(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    net_id = haskey(nn_models, :net1) ? :net1 : :net5
    st, nn_model = nn_models[net_id]
    nnout = nn_model([1.0, 1.0], x[net_id], st)[1]
    _p = convert.(eltype(x), ComponentArray(oprob.p))
    _p[1:3] .= x[1:3]
    _p[4] = nnout[1]
    _oprob = remake(oprob, p = _p)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh1(sol, mprey, mpredator)
end

function llh_pre_ODE2(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    experiments = ["e1", "e2"]
    inputs = [[10.0, 20.0], [1.0, 1.0]]
    st, nn_model = nn_models[:net1]
    llh = 0.0
    for (i, experiment) in pairs(experiments)
        mprey, mpredator, tsave = _get_measurement_info(
            measurements; experiment = experiment
        )
        nnout = nn_model(inputs[i], x.net1, st)[1]
        p = convert.(eltype(x), ComponentArray(oprob.p))
        p[1:3] .= x[1:3]
        p[4] = nnout[1]
        _oprob = remake(oprob, p = p)
        sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
        llh += _llh1(sol, mprey, mpredator)
    end
    return llh
end

function llh_pre_ODE3(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    st, nn_model = nn_models[:net1]
    nnout = nn_model([1.0, x.alpha], x.net1, st)[1]
    _p = convert.(eltype(x), ComponentArray(oprob.p))
    _p[1:3] .= x[1:3]
    _p[4] = nnout[1]
    _oprob = remake(oprob, p = _p)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh1(sol, mprey, mpredator)
end

function llh_pre_ODE4(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    st, nn_model = nn_models[:net1]
    nnout = nn_model([1.0, x.net1_input_pre2_est], x.net1, st)[1]
    _p = convert.(eltype(x), ComponentArray(oprob.p))
    _p[1:3] .= x[1:3]
    _p[4] = nnout[1]
    _oprob = remake(oprob, p = _p)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh1(sol, mprey, mpredator)
end

function llh_pre_ODE5(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    st, nn_model = nn_models[:net1]
    nnout = nn_model([1.0, 1.0], x.net1, st)[1]
    p = convert.(eltype(x), ComponentArray(oprob.p))
    u0 = convert.(eltype(x), ComponentArray(oprob.u0))
    p[1:4] .= x[1:4]
    u0[1] = nnout[1]
    _oprob = remake(oprob, p = p, u0 = u0)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh1(sol, mprey, mpredator)
end

function llh_pre_ODE6(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    st1, nn_model1 = nn_models[:net1]
    st2, nn_model2 = nn_models[:net2]
    nnout1 = nn_model1([1.0, 1.0], x.net1, st1)[1]
    nnout2 = nn_model2([2.0, 2.0], x.net2, st2)[1]
    p = convert.(eltype(x), ComponentArray(oprob.p))
    p[1:2] .= x[1:2]
    p[3] = nnout2[1]
    p[4] = nnout1[1]
    _oprob = remake(oprob, p = p)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh1(sol, mprey, mpredator)
end

function llh_pre_ODE6(
        x, oprob::ODEProblem, nn_models, measurements::DataFrame,
        input_arrays
    )::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    st1, nn_model1 = nn_models[:net1]
    st2, nn_model2 = nn_models[:net2]
    nnout1 = nn_model1([1.0, 1.0], x.net1, st1)[1]
    nnout2 = nn_model2([2.0, 2.0], x.net2, st2)[1]
    p = convert.(eltype(x), ComponentArray(oprob.p))
    p[1:2] .= x[1:2]
    p[3] = nnout2[1]
    p[4] = nnout1[1]
    _oprob = remake(oprob, p = p)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh1(sol, mprey, mpredator)
end

function llh_pre_ODE7(
        x, oprob::ODEProblem, nn_models, measurements::DataFrame,
        inputs
    )::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    st, nn_model = nn_models[:net3]
    nnout = nn_model(inputs[1], x.net3, st)[1]
    _p = convert.(eltype(x), ComponentArray(oprob.p))
    _p[1:3] .= x[1:3]
    _p[4] = nnout[1]
    _oprob = remake(oprob, p = _p)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh1(sol, mprey, mpredator)
end

function llh_pre_ODE8(
        x, oprob::ODEProblem, nn_models, measurements::DataFrame, inputs
    )::Real
    experiments = ["e1", "e2"]
    st, nn_model = nn_models[:net3]
    llh = 0.0
    for (i, experiment) in pairs(experiments)
        mprey, mpredator, tsave = _get_measurement_info(
            measurements; experiment = experiment
        )
        nnout = nn_model(inputs[i], x.net3, st)[1]
        p = convert.(eltype(x), ComponentArray(oprob.p))
        p[1:3] .= x[1:3]
        p[4] = nnout[1]
        _oprob = remake(oprob, p = p)
        sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
        llh += _llh1(sol, mprey, mpredator)
    end
    return llh
end

function llh_pre_ODE9(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    st, nn_model = nn_models[:net4]
    nnout = nn_model([1.0, 1.0], x.net4, st)[1]
    _p = convert.(eltype(x), ComponentArray(oprob.p))
    _p.delta = x.delta
    _p.beta = x.beta
    _p.alpha = nnout[1]
    _p.gamma = nnout[2]
    _oprob = remake(oprob, p = _p)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh1(sol, mprey, mpredator)
end

function llh_OBS1(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    _oprob = remake(oprob, p = x)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh2(sol, mprey, mpredator, x, nn_models)
end

function llh_OBS2(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    _oprob = remake(oprob, p = x)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh3(sol, mprey, mpredator, x, nn_models)
end

function llh_OBS3(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    _oprob = remake(oprob, p = x)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh6(sol, mprey, mpredator, x, nn_models)
end

function llh_COMBO1(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    st2, nn_model2 = nn_models[:net2]
    p = convert.(eltype(x), oprob.p)
    p.alpha = x.alpha
    p.delta = x.delta
    p.beta = nn_model2([2.0, 2.0], x.net2, st2)[1][1]
    p.net1 .= x.net1
    _oprob = remake(oprob, p = p)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh1(sol, mprey, mpredator)
end

function llh_COMBO2(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    p = convert.(eltype(x), oprob.p)
    p.alpha = x.alpha
    p.delta = x.delta
    p.beta = x.beta
    p.net1 .= x.net1
    _oprob = remake(oprob, p = p)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh4(sol, mprey, mpredator, x, nn_models)
end

function llh_COMBO3(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    st1, nn_model1 = nn_models[:net1]
    nnout = nn_model1([1.0, 1.0], x.net1, st1)[1]
    _p = convert.(eltype(x), ComponentArray(oprob.p))
    _p[1:3] .= x[1:3]
    _p[4] = nnout[1]
    _oprob = remake(oprob, p = _p)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh4(sol, mprey, mpredator, x, nn_models)
end

function llh_COMBO4(x, oprob::ODEProblem, nn_models, measurements::DataFrame)::Real
    mprey, mpredator, tsave = _get_measurement_info(measurements)
    _p = convert.(eltype(x), ComponentArray(oprob.p))
    _p[1:3] .= x[1:3]
    _p.net4 .= x.net4
    _oprob = remake(oprob, p = _p)
    sol = solve(_oprob, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = tsave)
    return _llh5(sol, mprey, mpredator, x, nn_models)
end

function _llh1(sol::ODESolution, mprey, mpredator)
    prey, predator = sol[1, :], sol[2, :]
    nllh, σ = 0.0, 0.05
    for i in eachindex(mprey)
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mprey[i] - prey[i])^2 / σ^2
    end
    for i in eachindex(mpredator)
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mpredator[i] - predator[i])^2 / σ^2
    end
    return nllh * -1
end

function _llh2(sol::ODESolution, mprey, mpredator, x, nn_models)
    prey, predator = sol[1, :], sol[2, :]
    nllh, σ = 0.0, 0.05
    for i in eachindex(mprey)
        net_id = haskey(nn_models, :net1) ? :net1 : :net5
        st, nn_model = nn_models[net_id]
        model_output = nn_model([prey[i], predator[i]], x[net_id], st)[1][1]
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mprey[i] - model_output)^2 / σ^2
    end
    for i in eachindex(mpredator)
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mpredator[i] - predator[i])^2 / σ^2
    end
    return nllh * -1
end

function _llh3(sol::ODESolution, mprey, mpredator, x, nn_models)
    prey, predator = sol[1, :], sol[2, :]
    st1, nn_model1 = nn_models[:net1]
    st2, nn_model2 = nn_models[:net2]
    nllh, σ = 0.0, 0.05
    for i in eachindex(mprey)
        model_output = nn_model1([prey[i], predator[i]], x.net1, st1)[1][1]
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mprey[i] - model_output)^2 / σ^2
    end
    for i in eachindex(mpredator)
        model_output = nn_model2([x.alpha, predator[i]], x.net2, st2)[1][1]
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mpredator[i] - model_output)^2 / σ^2
    end
    return nllh * -1
end

function _llh4(sol::ODESolution, mprey, mpredator, x, nn_models)
    prey, predator = sol[1, :], sol[2, :]
    st2, nn_model2 = nn_models[:net2]
    nllh, σ = 0.0, 0.05
    for i in eachindex(mprey)
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mprey[i] - prey[i])^2 / σ^2
    end
    for i in eachindex(mpredator)
        model_output = nn_model2([x.alpha, predator[i]], x.net2, st2)[1][1]
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mpredator[i] - model_output)^2 / σ^2
    end
    return nllh * -1
end

function _llh5(sol::ODESolution, mprey, mpredator, x, nn_models)
    prey, predator = sol[1, :], sol[2, :]
    st4, nn_model4 = nn_models[:net4]
    nllh, σ = 0.0, 0.05
    for i in eachindex(mprey)
        model_output = nn_model4([prey[i], predator[i]], x.net4, st4)[1][1]
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mprey[i] - model_output)^2 / σ^2
    end
    for i in eachindex(mpredator)
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mpredator[i] - predator[i])^2 / σ^2
    end
    return nllh * -1
end

function _llh6(sol::ODESolution, mprey, mpredator, x, nn_models)
    prey, predator = sol[1, :], sol[2, :]
    st4, nn_model4 = nn_models[:net4]
    nllh, σ = 0.0, 0.05
    for i in eachindex(mprey)
        model_output = nn_model4([prey[i], predator[i]], x.net4, st4)[1][1]
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mprey[i] - model_output)^2 / σ^2
    end
    for i in eachindex(mpredator)
        model_output = nn_model4([prey[i], predator[i]], x.net4, st4)[1][2]
        nllh += log(σ) + 0.5 * log(2π) + 0.5 * (mpredator[i] - model_output)^2 / σ^2
    end
    return nllh * -1
end

function _get_measurement_info(measurements::DataFrame; experiment = "")
    if !isempty(experiment)
        measurements = measurements[measurements[:, :experimentId] .== experiment, :]
    end
    mprey = measurements[measurements[!, :observableId] .== "prey_o", :measurement]
    mpredator = measurements[measurements[!, :observableId] .== "predator_o", :measurement]
    tsave = unique(measurements.time)
    return mprey, mpredator, tsave
end

function save_grad(
        x, llh::Function, nn_models::Dict, estimate_net_parameters::Bool,
        freeze_info::Union{Nothing, Dict}, dir_save
    )::Nothing
    grad = get_grad_objective(x, llh)

    # Non-neural net parameters
    i_mechanistic = findall(x -> x ∉ keys(nn_models), keys(x))
    grad_mechanistic = keys(grad)[i_mechanistic] .=> grad[i_mechanistic]
    df_mech = DataFrame(
        parameterId = first.(grad_mechanistic),
        value = last.(grad_mechanistic)
    )
    CSV.write(joinpath(dir_save, "grad_mech.tsv"), df_mech, delim = '\t')

    # Neural net parameters
    if estimate_net_parameters
        for net_id in keys(nn_models)
            nn_ps_to_h5(
                nn_models[net_id][2], grad[net_id], freeze_info, net_id,
                joinpath(dir_save, "grad_$(net_id).hdf5")
            )
        end
    end
    return nothing
end

function get_grad_objective(x::T, objective::Function)::T where {T}
    return FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), objective, x)[1]
end

function get_x(petab_parameters_ids, x_nn, nets_info::Dict)
    # Mechanistic (non-neural net) parameters
    x_mechanistic = Vector{Pair{Symbol, Float64}}(undef, 0)
    for id in petab_parameters_ids
        _get_parameter_info(id, :estimate) == false && continue
        ismissing(_get_parameter_info(id, :value)) && continue
        push!(x_mechanistic, id => _get_parameter_info(id, :value))
    end
    x_mechanistic = NamedTuple(x_mechanistic)

    # Neural net parameters
    net_ids = collect(keys(nets_info))
    _x_nn = Vector{Pair{Symbol, ComponentArray}}(undef, length(net_ids))
    for (i, net_id) in pairs(net_ids)
        _x_nn[i] = net_id => x_nn[i]
    end

    # Return in order mechanistic + neural net
    x = merge(x_mechanistic, NamedTuple(_x_nn)) |>
        ComponentVector
    return x[tuple(keys(x_mechanistic)..., net_ids...)]
end
