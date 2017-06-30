#include "loop.h"
#include "lrate.h"
#include "tensor/momentum.h"
#include "solver_stoch_sgm.h"

using namespace nano;

stoch_sgm_t::stoch_sgm_t(const string_t& params) :
        stoch_solver_t(to_params(params, "alpha0", 1.0, "decay", 0.5, "momentum", 0.9))
{
}

function_state_t stoch_sgm_t::tune(const stoch_params_t& param, const function_t& function, const vector_t& x0)
{
        const auto tuned = stoch_tune(this, param, function, x0, make_alpha0s(), make_decays(), make_momenta());
        config(to_params(
                "alpha0", std::get<0>(tuned.params()),
                "decay", std::get<1>(tuned.params()),
                "momentum", std::get<2>(tuned.params())));
        return tuned.optimum();
}

function_state_t stoch_sgm_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
{
        return  minimize(param.tuned(), function, x0,
                from_params<scalar_t>(config(), "alpha0"),
                from_params<scalar_t>(config(), "decay"),
                from_params<scalar_t>(config(), "momentum"));
}

function_state_t stoch_sgm_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
        const scalar_t alpha0, const scalar_t decay, const scalar_t momentum)
{
        // learning rate schedule
        lrate_t lrate(alpha0, decay);

        // first-order momentum of the gradient
        momentum_t<vector_t> gsum1(momentum, x0.size());

        // assembly the optimizer
        const auto optimizer = [&] (function_state_t& cstate, const function_state_t&)
        {
                // learning rate
                const scalar_t alpha = lrate.get();

                // descent direction
                gsum1.update(cstate.g);

                cstate.d = -gsum1.value();

                // update solution
                function.stoch_next();
                cstate.stoch_update(function, alpha);
        };

        const auto snapshot = [&] (const function_state_t& cstate, function_state_t& sstate)
        {
                sstate.update(function, cstate.x);
        };

        return  stoch_loop(param, function, x0, optimizer, snapshot,
                to_params("alpha0", alpha0, "decay", decay, "momentum", momentum));
}
