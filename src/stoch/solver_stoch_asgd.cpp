#include "loop.h"
#include "lrate.h"
#include "tensor/momentum.h"
#include "solver_stoch_asgd.h"

using namespace nano;

stoch_asgd_t::stoch_asgd_t(const string_t& configuration) :
        stoch_solver_t(configuration)
{
}

function_state_t stoch_asgd_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return stoch_tune(this, param, function, x0, make_alpha0s(), make_decays(), make_momenta());
}

function_state_t stoch_asgd_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t alpha0, const scalar_t decay, const scalar_t momentum) const
{
        // learning rate schedule
        lrate_t lrate(alpha0, decay);

        // average state
        momentum_t<vector_t> xavg(momentum, x0.size());

        // assembly the optimizer
        const auto optimizer = [&] (function_state_t& cstate, const function_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                xavg.update(cstate.x);
                cstate.d = -cstate.g;

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const function_state_t&, function_state_t& sstate)
        {
                sstate.update(function, xavg.value());
        };

        return  stoch_loop(param, function, x0, optimizer, snapshot,
                to_params("alpha0", alpha0, "decay", decay, "momentum", momentum));
}
