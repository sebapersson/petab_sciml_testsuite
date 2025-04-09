function save_simulations(x, llh_id::Symbol, ode_problem::ODEProblem, nn_models, measurements::DataFrame, inputs, dir_save)
    simulations = get_simulations(x, llh_id, ode_problem, nn_models, measurements, inputs)
    CSV.write(joinpath(dir_save, "simulations.tsv"), simulations, delim = '\t')
    return nothing
end

function get_simulations(x, llh_id::Symbol, ode_problem::ODEProblem, nn_models, measurements::DataFrame, inputs)
    simulations = deepcopy(measurements)

    if llh_id in [:UDE1, :UDE2]
        _ode_problem = remake(ode_problem, p = x)
        sol = solve(ode_problem, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        simulated_values = vcat(sol[1, :], sol[2, :])
    end

    if llh_id in [:pre_ODE1, :pre_ODE3, :pre_ODE4]
        if llh_id in [:pre_ODE1, :pre_ODE4]
            input = [1.0, 1.0]
        elseif llh_id == :pre_ODE3
            input = [1.0, x.alpha]
        end
        st, nn_model = nn_models[:net1]
        p = (alpha = 1.3, delta = 1.8, beta = 0.9, gamma = nn_model(input, x.net1, st)[1][1])
        _ode_problem = remake(ode_problem, p = p)
        sol = solve(_ode_problem, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        simulated_values = vcat(sol[1, :], sol[2, :])
    end

    if llh_id == :pre_ODE2
        st, nn_model = nn_models[:net1]
        p1 = (alpha = 1.3, delta = 1.8, beta = 0.9, gamma = nn_model([10.0, 20.0], x.net1, st)[1][1])
        p2 = (alpha = 1.3, delta = 1.8, beta = 0.9, gamma = nn_model([1.0, 1.0], x.net1, st)[1][1])
        ode_problem1 = remake(ode_problem, p = p1)
        ode_problem2 = remake(ode_problem, p = p2)
        sol1 = solve(ode_problem1, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        sol2 = solve(ode_problem2, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        simulated_values = vcat(sol1[1, :], sol1[2, :], sol2[1, :], sol2[2, :])
    end

    if llh_id == :pre_ODE5
        st, nn_model = nn_models[:net1]
        u0 = [nn_model([1.0, 1.0], x.net1, st)[1][1], 4.6280594]
        _ode_problem = remake(ode_problem, u0 = u0)
        sol = solve(_ode_problem, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        simulated_values = vcat(sol[1, :], sol[2, :])
    end

    if llh_id == :pre_ODE6
        st1, nn_model1 = nn_models[:net1]
        st2, nn_model2 = nn_models[:net1]
        γ = nn_model1([1.0, 1.0], x.net1, st1)[1][1]
        β = nn_model2([2.0, 2.0], x.net2, st2)[1][1]
        p = (alpha = 1.3, delta = 1.8, beta = β, gamma = γ)
        _ode_problem = remake(ode_problem, p = p)
        sol = solve(_ode_problem, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        simulated_values = vcat(sol[1, :], sol[2, :])
    end

    if llh_id == :pre_ODE7
        st, nn_model = nn_models[:net3]
        γ = nn_model(inputs[1], x.net3, st)[1][1]
        p = (alpha = 1.3, delta = 1.8, beta = 0.9, gamma = γ)
        _ode_problem = remake(ode_problem, p = p)
        sol = solve(_ode_problem, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        simulated_values = vcat(sol[1, :], sol[2, :])
    end

    if llh_id == :pre_ODE8
        st, nn_model = nn_models[:net3]
        p1 = (alpha = 1.3, delta = 1.8, beta = 0.9, gamma = nn_model(inputs[1], x.net3, st)[1][1])
        p2 = (alpha = 1.3, delta = 1.8, beta = 0.9, gamma = nn_model(inputs[2], x.net3, st)[1][1])
        ode_problem1 = remake(ode_problem, p = p1)
        ode_problem2 = remake(ode_problem, p = p2)
        sol1 = solve(ode_problem1, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        sol2 = solve(ode_problem2, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        simulated_values = vcat(sol1[1, :], sol1[2, :], sol2[1, :], sol2[2, :])
    end

    if llh_id == :OBS1
        st, nn_model = nn_models[:net1]
        sol = solve(ode_problem, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        prey = [nn_model(sol[:, i], x.net1, st)[1][1] for i in eachindex(sol.t)]
        simulated_values = vcat(prey, sol[2, :])
    end

    if llh_id == :OBS2
        st1, nn_model1 = nn_models[:net1]
        st2, nn_model2 = nn_models[:net2]
        sol = solve(ode_problem, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        prey = [nn_model1(sol[:, i], x.net1, st1)[1][1] for i in eachindex(sol.t)]
        predator = [nn_model2(sol[:, i], x.net2, st2)[1][1] for i in eachindex(sol.t)]
        simulated_values = vcat(prey, sol[2, :])
    end

    if llh_id == :COMBO1
        st2, nn_model2 = nn_models[:net2]
        ode_problem.p.beta = nn_model2([2.0, 2.0], x.net2, st2)[1][1]
        ode_problem.p.net1 .= x.net1
        sol = solve(ode_problem, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        simulated_values = vcat(sol[1, :], sol[2, :])
    end

    if llh_id == :COMBO2
        st2, nn_model2 = nn_models[:net2]
        ode_problem.p.beta = nn_model2([2.0, 2.0], x.net2, st2)[1][1]
        ode_problem.p.net1 .= x.net1
        sol = solve(ode_problem, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        predator = [nn_model2(sol[:, i], x.net2, st2)[1][1] for i in eachindex(sol.t)]
        simulated_values = vcat(sol[1, :], predator)
    end

    if llh_id == :COMBO3
        st1, nn_model1 = nn_models[:net1]
        st2, nn_model2 = nn_models[:net2]
        p = (alpha = 1.3, delta = 1.8, beta = 0.9, gamma = nn_model1([1.0, 1.0], x.net1, st1)[1][1])
        sol = solve(ode_problem, Vern9(), abstol = 1e-12, reltol = 1e-12, saveat = unique(measurements.time))
        predator = [nn_model2(sol[:, i], x.net2, st2)[1][1] for i in eachindex(sol.t)]
        simulated_values = vcat(sol[1, :], predator)
    end

    rename!(simulations, "measurement" => "simulation")
    simulations.simulation .= simulated_values
    return simulations
end
