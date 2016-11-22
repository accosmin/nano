#include "adam.h"
#include "loop.h"
#include "math/momentum.h"

namespace nano
{
        stoch_adam_t::stoch_adam_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_adam_t::minimize(const stoch_params_t& param, const function_t& problem, const vector_t& x0) const
        {
                return stoch_tune(this, param, problem, x0, make_alpha0s(), make_epsilons());
        }

        state_t stoch_adam_t::minimize(const stoch_params_t& param, const function_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t epsilon) const
        {
                assert(problem.size() == x0.size());

                const scalar_t beta1 = scalar_t(0.900);
                const scalar_t beta2 = scalar_t(0.999);

                // first-order momentum of the gradient
                momentum_vector_t<vector_t> m(beta1, x0.size());

                // second-order momentum of the gradient
                momentum_vector_t<vector_t> v(beta2, x0.size());

                // optimizer
                const auto optimizer = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = alpha0;

                        // descent direction
                        m.update(cstate.g);
                        v.update(cstate.g.array().square());

                        cstate.d = -m.value().array() / (epsilon + v.value().array().sqrt());

                        // update solution
                        cstate.stoch_update(problem, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(param, problem, x0, optimizer,
                        {{"alpha0", alpha0}, {"epsilon", epsilon}, {"beta1", beta1}, {"beta2", beta2}});
        }
}

