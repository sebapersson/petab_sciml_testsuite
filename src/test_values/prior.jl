function get_log_prior(prior_id::Union{Nothing, Symbol})::Function
    if isnothing(prior_id)
        return _no_prior
    elseif prior_id == :prior1
        return _prior1
    elseif prior_id == :prior2
        return _prior2
    elseif prior_id == :prior3
        return _prior3
    end
end

function _no_prior(x)
    return 0.0
end

function _prior1(x)
    mech = sum(logpdf.(Uniform(0.0, 15.0), [x.alpha, x.delta, x.beta]))
    ml = sum(logpdf.(Normal(0.0, 1.0), collect(x.net1)))
    return mech + ml
end

function _prior2(x)
    mech = sum(logpdf.(Uniform(0.0, 15.0), [x.alpha, x.delta, x.beta]))
    layer1 = sum(logpdf.(Normal(0.0, 2.0), collect(x.net1.layer1)))
    layer2 = sum(logpdf.(Normal(0.0, 1.0), collect(x.net1.layer2)))
    layer3 = sum(logpdf.(Normal(0.0, 1.0), collect(x.net1.layer3)))
    return mech + layer1 + layer2 + layer3
end

function _prior3(x)
    mech = sum(logpdf.(Uniform(0.0, 15.0), [x.alpha, x.delta, x.beta]))
    layer1_weight = sum(logpdf.(Normal(0.0, 2.0), collect(x.net1.layer1.weight)))
    layer1_bias = sum(logpdf.(Normal(0.0, 1.0), collect(x.net1.layer1.bias)))
    layer2 = sum(logpdf.(Normal(0.0, 1.0), collect(x.net1.layer2)))
    layer3 = sum(logpdf.(Normal(0.0, 1.0), collect(x.net1.layer3)))
    return mech + layer1_weight + layer1_bias + layer2 + layer3
end
