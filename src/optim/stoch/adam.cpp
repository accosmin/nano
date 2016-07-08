#include "adam.h"
#include "loop.hpp"

namespace nano
{
        state_t stoch_adam_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                return stoch_tune(this, param, problem, x0, make_alpha0s(), make_epsilons());
        }

        state_t stoch_adam_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t epsilon) const
        {
                assert(problem.size() == x0.size());

                const scalar_t beta1 = scalar_t(0.900);
                const scalar_t beta2 = scalar_t(0.999);

                // first-order momentum of the gradient
                momentum_vector_t<vector_t> m(beta1, x0.size());

                // second-order momentum of the gradient
                momentum_vector_t<vector_t> v(beta2, x0.size());

                const auto op_iter = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = alpha0;

                        // descent direction
                        m.update(cstate.g);
                        v.update(cstate.g.array().square());

                        cstate.d = -m.value().array() / (epsilon + v.value().array().sqrt());

                        // update solution
                        cstate.update(problem, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                        {{"alpha0", alpha0}, {"epsilon", epsilon}, {"beta1", beta1}, {"beta2", beta2}});
        }
}

