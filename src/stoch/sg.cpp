#include "sg.h"
#include "loop.h"
#include "lrate.h"

namespace nano
{
        stoch_sg_t::stoch_sg_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        function_state_t stoch_sg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
        {
                return stoch_tune(this, param, function, x0, make_alpha0s(), make_decays());
        }

        function_state_t stoch_sg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay) const
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
}
