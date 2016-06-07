#include "adam.h"
#include "loop.hpp"

namespace nano
{
        state_t stoch_adam_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                const auto op = [&] (const auto... params)
                {
                        return this->operator()(param.tunable(), problem, x0, params...);
                };

                const auto param0 = make_finite_space(scalar_t(1e-4), scalar_t(1e-3));
                const auto param1 = make_epsilons();
                const auto param2 = make_log10_space(std::log10(scalar_t(0.1)), std::log10(scalar_t(0.9000)), scalar_t(0.2));
                const auto param3 = make_log10_space(std::log10(scalar_t(0.9)), std::log10(scalar_t(0.9999)), scalar_t(0.2));
                const auto config = nano::tune(op, param0, param1, param2, param3);
                return operator()(param, problem, x0, config.param0(), config.param1(), config.param2(), config.param3());
        }

        state_t stoch_adam_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t epsilon,
                const scalar_t beta1, const scalar_t beta2) const
        {
                assert(problem.size() == x0.size());

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

