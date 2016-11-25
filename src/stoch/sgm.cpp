#include "sgm.h"
#include "loop.h"
#include "lrate.h"
#include "math/momentum.h"
#include "text/to_params.h"

namespace nano
{
        stoch_sgm_t::stoch_sgm_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_sgm_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
        {
                return stoch_tune(this, param, function, x0, make_alpha0s(), make_decays(), make_momenta());
        }

        state_t stoch_sgm_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay, const scalar_t momentum) const
        {
                assert(function.size() == x0.size());

                // learning rate schedule
                lrate_t lrate(alpha0, decay, param.m_epoch_size);

                // first-order momentum of the update
                momentum_vector_t<vector_t> davg(momentum, x0.size());

                // optimizer
                const auto optimizer = [&] (state_t& cstate, const state_t&)
                {
                        // learning rate
                        const scalar_t alpha = lrate.get();

                        // descent direction
                        davg.update(-alpha * cstate.g);
                        cstate.d = davg.value();

                        // update solution
                        function.stoch_next();
                        cstate.stoch_update(function, 1);
                };

                // OK, assembly the optimizer
                return  stoch_loop(param, function, x0, optimizer,
                        to_params("alpha0", alpha0, "decay", decay, "momentum", momentum));
        }
}

