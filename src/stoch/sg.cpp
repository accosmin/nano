#include "sg.h"
#include "loop.h"
#include "lrate.h"
#include "text/to_params.h"

namespace nano
{
        stoch_sg_t::stoch_sg_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_sg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
        {
                return stoch_tune(this, param, function, x0, make_alpha0s(), make_decays());
        }

        state_t stoch_sg_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay) const
        {
                assert(function.size() == x0.size());

                // learning rate schedule
                lrate_t lrate(alpha0, decay, param.m_epoch_size);

                // optimizer
                const auto optimizer = [&] (state_t& cstate, const state_t&)
                {
                        // learning rate
                        const scalar_t alpha = lrate.get();

                        // descent direction
                        cstate.d = -cstate.g;

                        // update solution
                        function.stoch_next();
                        cstate.stoch_update(function, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(param, function, x0, optimizer,
                        to_params("alpha0", alpha0, "decay", decay));
        }
}

