function get_odeproblem(ode_id::Symbol, nn_models::Dict)
    rng = StableRNGs.StableRNG(1)
    if ode_id == :reference
        u0 = [0.44249296, 4.6280594]
        p_mechanistic = (alpha = 1.3, delta = 1.8, beta = 0.9, gamma = 0.8)
        return ODEProblem(lv_reference!, u0, (0.0, 10.0), p_mechanistic)
    end

    if ode_id == :UDE1
        lv! = let nn = nn_models
            (du, u, p, t) -> lv_ude1(du, u, p, t, nn)
        end
        p_mechanistic = (alpha = 1.3, delta = 1.8, beta = 0.9)
        if haskey(nn_models, :net1)
            pnn = Lux.initialparameters(rng, nn_models[:net1][2])
            p_ode = ComponentArray(merge(p_mechanistic, (net1 = pnn,)))
        else
            pnn = Lux.initialparameters(rng, nn_models[:net4][2])
            p_ode = ComponentArray(merge(p_mechanistic, (net4 = pnn,)))
        end
        u0 = [0.44249296, 4.6280594]
        return ODEProblem(lv!, u0, (0.0, 10.0), p_ode)
    end

    if ode_id == :UDE2
        lv! = let nn = nn_models
            (du, u, p, t) -> lv_ude2!(du, u, p, t, nn)
        end
        p_mechanistic = (delta = 1.8, beta = 0.9)
        pnn1 = Lux.initialparameters(rng, nn_models[:net1][2])
        pnn2 = Lux.initialparameters(rng, nn_models[:net2][2])
        p_ode = ComponentArray(merge(p_mechanistic, (net1 = pnn1, net2 = pnn2)))
        u0 = [0.44249296, 4.6280594]
        return ODEProblem(lv!, u0, (0.0, 10.0), p_ode)
    end

    if ode_id == :UDE3
        lv! = let nn = nn_models
            (du, u, p, t) -> lv_ude3!(du, u, p, t, nn)
        end
        p_mechanistic = (delta = 1.8, beta = 0.9)
        pnn4 = Lux.initialparameters(rng, nn_models[:net4][2])
        p_ode = ComponentArray(merge(p_mechanistic, (net4 = pnn4, )))
        u0 = [0.44249296, 4.6280594]
        return ODEProblem(lv!, u0, (0.0, 10.0), p_ode)
    end
end

function lv_reference!(du, u, p, t)
    prey, predator = u
    @unpack alpha, delta, beta, gamma = p

    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = gamma * prey * predator - delta * predator # predator
    return nothing
end

function lv_ude1(du, u, p, t, nn_models)
    prey, predator = u
    @unpack alpha, delta, beta = p

    if haskey(nn_models, :net1)
        st1, nn1 = nn_models[:net1]
        du_nn = nn1([prey, predator], p.net1, st1)[1][1]
    else
        st4, nn4 = nn_models[:net4]
        du_nn = nn4([prey, predator], p.net4, st4)[1][2]
    end

    du[1] = alpha * prey - beta * prey * predator # prey
    du[2] = du_nn - delta * predator # predator
    return nothing
end

function lv_ude2!(du, u, p, t, nn_models)
    prey, predator = u
    @unpack delta, beta = p

    st1, nn1 = nn_models[:net1]
    du_nn1 = nn1([prey, predator], p.net1, st1)[1]
    st2, nn2 = nn_models[:net2]
    du_nn2 = nn2([prey, predator], p.net2, st2)[1]

    du[1] = du_nn2[1] - beta * prey * predator # prey
    du[2] = du_nn1[1] - delta * predator # predator
    return nothing
end

function lv_ude3!(du, u, p, t, nn_models)
    prey, predator = u
    @unpack delta, beta = p

    st4, nn4 = nn_models[:net4]
    du_nn = nn4([prey, predator], p.net4, st4)[1]

    du[1] = du_nn[1] - beta * prey * predator # prey
    du[2] = du_nn[2] - delta * predator # predator
    return nothing
end
