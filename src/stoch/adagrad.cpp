#include "loop.h"
#include "lrate.h"
#include "adagrad.h"
#include "text/to_params.h"

namespace nano
{
        stoch_adagrad_t::stoch_adagrad_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_adagrad_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
        {
                return stoch_tune(this, param, function, x0, make_alpha0s(), make_decays(), make_epsilons());
        }

        state_t stoch_adagrad_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay, const scalar_t epsilon) const
        {
                // learning rate schedule
                lrate_t lrate(alpha0, decay);

                // second-order gradient momentum
                vector_t gsum2 = vector_t::Zero(x0.size());

                // assembly the optimizer
                const auto optimizer = [&] (state_t& cstate, const state_t&)
                {
                        // learning rate
                        const scalar_t alpha = lrate.get();

                        // descent direction
                        gsum2.array() += cstate.g.array().square();

                        cstate.d = -cstate.g.array() / (epsilon + gsum2.array().sqrt());

                        // update solution
                        function.stoch_next();
                        cstate.stoch_update(function, alpha);
                };

                const auto snapshot = [&] (const state_t& cstate, state_t& sstate)
                {
                        sstate.update(function, cstate.x);
                };

                return  stoch_loop(param, function, x0, optimizer, snapshot,
                        to_params("alpha0", alpha0, "decay", decay, "epsilon", epsilon));
        }
}
