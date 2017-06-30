#include "loop.h"
#include "solver_stoch_adagrad.h"

using namespace nano;

stoch_adagrad_t::stoch_adagrad_t(const string_t& configuration) :
        stoch_solver_t(to_params(configuration, "alpha0", 1.0, "epsilon", 1e-6))
{
}

function_state_t stoch_adagrad_t::tune(const stoch_params_t& param, const function_t& function, const vector_t& x0)
{
        const auto tuned = stoch_tune(this, param, function, x0, make_alpha0s(), make_epsilons());
        config(to_params(
                "alpha0", std::get<0>(tuned.params()),
                "epsilon", std::get<1>(tuned.params())));
        return tuned.optimum();
}

function_state_t stoch_adagrad_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return  minimize(param.tuned(), function, x0,
                from_params<scalar_t>(config(), "alpha0"),
                from_params<scalar_t>(config(), "epsilon"));
}

function_state_t stoch_adagrad_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t alpha0, const scalar_t epsilon)
{
        // second-order gradient momentum
        vector_t gsum2 = vector_t::Zero(x0.size());

        // assembly the optimizer
        const auto optimizer = [&] (function_state_t& cstate, const function_state_t&)
        {
                // learning rate
                const scalar_t alpha = alpha0;

                // descent direction
                gsum2.array() += cstate.g.array().square();

                cstate.d = -cstate.g.array() / (epsilon + gsum2.array().sqrt());

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const function_state_t& cstate, function_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return  stoch_loop(param, function, x0, optimizer, snapshot,
                to_params("alpha0", alpha0, "epsilon", epsilon));
}
