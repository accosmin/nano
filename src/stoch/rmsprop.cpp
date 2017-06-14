#include "loop.h"
#include "lrate.h"
#include "rmsprop.h"
#include "tensor/momentum.h"

using namespace nano;

stoch_rmsprop_t::stoch_rmsprop_t(const string_t& configuration) :
        stoch_solver_t(configuration)
{
}

function_state_t stoch_rmsprop_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return stoch_tune(this, param, function, x0, make_alpha0s(), make_decays(), make_momenta(), make_epsilons());
}

function_state_t stoch_rmsprop_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t alpha0, const scalar_t decay, const scalar_t momentum, const scalar_t epsilon) const
{
        // learning rate schedule
        lrate_t lrate(alpha0, decay);

        // second-order momentum of the gradient
        momentum_t<vector_t> gsum2(momentum, x0.size());

        // assembly the optimizer
        const auto optimizer = [&] (function_state_t& cstate, const function_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                gsum2.update(cstate.g.array().square());

                cstate.d = -cstate.g.array() / (epsilon + gsum2.value().array().sqrt());

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const function_state_t& cstate, function_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return  stoch_loop(param, function, x0, optimizer, snapshot,
                to_params("alpha0", alpha0, "decay", decay, "momentum", momentum, "epsilon", epsilon));
}
