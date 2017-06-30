#include "loop.h"
#include "lrate.h"
#include "solver_stoch_sg.h"

using namespace nano;

stoch_sg_t::stoch_sg_t(const string_t& configuration) :
        stoch_solver_t(to_params(configuration, "alpha0", 1.0, "decay", 0.5))
{
}

function_state_t stoch_sg_t::tune(const stoch_params_t& param, const function_t& function, const vector_t& x0)
{
        const auto tuned = stoch_tune(this, param, function, x0, make_alpha0s(), make_decays());
        config(to_params(
                "alpha0", std::get<0>(tuned.params()),
                "decay", std::get<1>(tuned.params())));
        return tuned.optimum();
}

function_state_t stoch_sg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return  minimize(param.tuned(), function, x0,
                from_params<scalar_t>(config(), "alpha0"),
                from_params<scalar_t>(config(), "decay"));
}

function_state_t stoch_sg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t alpha0, const scalar_t decay)
{
        // learning rate schedule
        lrate_t lrate(alpha0, decay);

        // assembly the optimizer
        const auto optimizer = [&] (function_state_t& cstate, const function_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                cstate.d = -cstate.g;

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const function_state_t& cstate, function_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return  stoch_loop(param, function, x0, optimizer, snapshot,
                to_params("alpha0", alpha0, "decay", decay));
}
