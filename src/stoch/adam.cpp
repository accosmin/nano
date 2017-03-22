#include "adam.h"
#include "loop.h"
#include "lrate.h"
#include "text/to_params.h"
#include "tensor/momentum.h"

namespace nano
{
        stoch_adam_t::stoch_adam_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_adam_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
        {
                const auto beta1s = make_finite_space(scalar_t(0.900));
                const auto beta2s = make_finite_space(scalar_t(0.99), scalar_t(0.999), scalar_t(0.9999), scalar_t(0.99999));
                return stoch_tune(this, param, function, x0, make_alpha0s(), make_decays(), make_epsilons(), beta1s, beta2s);
        }

        state_t stoch_adam_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay, const scalar_t epsilon, const scalar_t beta1, const scalar_t beta2) const
        {
                // learning rate schedule
                lrate_t lrate(alpha0, decay, param.m_epoch_size);

                // first-order momentum of the gradient
                nano::momentum_t<vector_t> m(beta1, x0.size());

                // second-order momentum of the gradient
                nano::momentum_t<vector_t> v(beta2, x0.size());

                // assembly the optimizer
                const auto optimizer = [&] (state_t& cstate, const state_t&)
                {
                        // learning rate
                        const scalar_t alpha = lrate.get();

                        // descent direction
                        m.update(cstate.g);
                        v.update(cstate.g.array().square());

                        cstate.d = -m.value().array() / (epsilon + v.value().array().sqrt());

                        // update solution
                        function.stoch_next();
                        cstate.stoch_update(function, alpha);
                };

                const auto snapshot = [&] (const state_t& cstate, state_t& sstate)
                {
                        sstate.update(function, cstate.x);
                };

                return  stoch_loop(param, function, x0, optimizer, snapshot,
                        to_params("alpha0", alpha0, "decay", decay, "epsilon", epsilon, "beta1", beta1, "beta2", beta2));
        }
}

