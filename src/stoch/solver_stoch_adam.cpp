#include "loop.h"
#include "lrate.h"
#include "tensor/momentum.h"
#include "solver_stoch_adam.h"

using namespace nano;

stoch_adam_t::stoch_adam_t(const string_t& configuration) :
        stoch_solver_t(to_params(configuration,
        "alpha0", 1.0, "decay", 0.5, "epsilon", 1e-6, "beta1", 0.900, "beta2", 0.999))
{
}

function_state_t stoch_adam_t::tune(const stoch_params_t& param, const function_t& function, const vector_t& x0)
{
        const auto beta1s = make_finite_space(scalar_t(0.90));
        const auto beta2s = make_finite_space(scalar_t(0.90), scalar_t(0.95), scalar_t(0.99));
        const auto tuned = stoch_tune(this, param, function, x0, make_alpha0s(), make_decays(), make_epsilons(), beta1s, beta2s);
        config(to_params(
                "alpha0", std::get<0>(tuned.params()),
                "decay", std::get<1>(tuned.params()),
                "epsilon", std::get<2>(tuned.params()),
                "beta1", std::get<3>(tuned.params()),
                "beta2", std::get<4>(tuned.params())));
        return tuned.optimum();
}

function_state_t stoch_adam_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return  minimize(param.tuned(), function, x0,
                from_params<scalar_t>(config(), "alpha0"),
                from_params<scalar_t>(config(), "decay"),
                from_params<scalar_t>(config(), "epsilon"),
                from_params<scalar_t>(config(), "beta1"),
                from_params<scalar_t>(config(), "beta2"));
}

function_state_t stoch_adam_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t alpha0, const scalar_t decay,
        const scalar_t epsilon, const scalar_t beta1, const scalar_t beta2)
{
        // learning rate schedule
        lrate_t lrate(alpha0, decay);

        // first-order momentum of the gradient
        momentum_t<vector_t> m(beta1, x0.size());

        // second-order momentum of the gradient
        momentum_t<vector_t> v(beta2, x0.size());

        // assembly the optimizer
        const auto optimizer = [&] (function_state_t& cstate, const function_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                m.update(cstate.g);
                v.update(cstate.g.array().square());

                cstate.d = -m.value().array() / (epsilon + v.value().array().sqrt());

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const function_state_t& cstate, function_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return  stoch_loop(param, function, x0, optimizer, snapshot,
                to_params("alpha0", alpha0, "decay", decay, "epsilon", epsilon, "beta1", beta1, "beta2", beta2));
}
