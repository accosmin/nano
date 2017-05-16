#include "svrg.h"
#include "loop.h"
#include "lrate.h"

namespace nano
{
        stoch_svrg_t::stoch_svrg_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_svrg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
        {
                return stoch_tune(this, param, function, x0, make_alpha0s(), make_decays());
        }

        state_t stoch_svrg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay) const
        {
                // learning rate schedule
                lrate_t lrate(alpha0, decay);

                // assembly the optimizer
                const auto optimizer = [&] (state_t& cstate, const state_t& sstate)
                {
                        // learning rate
                        const scalar_t alpha = lrate.get();

                        // descent direction
                        function.stoch_eval(sstate.x, &cstate.d);// NB: reuse descent direction to store snapshot gradient!
                        cstate.d.noalias() = - cstate.g + cstate.d - sstate.g;

                        // update solution
                        function.stoch_next();
                        cstate.stoch_update(function, alpha);
                };

                const auto snapshot = [&] (const state_t& cstate, state_t& sstate)
                {
                        sstate.update(function, cstate.x);
                };

                return  stoch_loop(param, function, x0, optimizer, snapshot,
                        to_params("alpha0", alpha0, "decay", decay));
        }
}
