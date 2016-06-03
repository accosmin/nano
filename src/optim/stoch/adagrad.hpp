#pragma once

#include "loop.hpp"
#include "math/average.hpp"

namespace nano
{
        ///
        /// \brief stochastic AdaGrad
        ///     see "Adaptive subgradient methods for online learning and stochastic optimization"
        ///     by J. C. Duchi, E. Hazan, and Y. Singer
        ///
        ///     see http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
        ///
        struct stoch_adagrad_t
        {
                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
                {
                        const auto op = [&] (const auto... params)
                        {
                                return this->operator()(param.tunable(), problem, x0, params...);
                        };

                        const auto param0 = make_alpha0s();
                        const auto param1 = make_epsilons();
                        const auto config = nano::tune(op, param0, param1);
                        return operator()(param, problem, x0, config.param0(), config.param1());
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t epsilon) const
                {
                        assert(problem.size() == x0.size());

                        // second-order gradient momentum
                        average_vector_t<vector_t> gavg(x0.size());

                        const auto op_iter = [&] (state_t& cstate)
                        {
                                // learning rate
                                const scalar_t alpha = alpha0;

                                // descent direction
                                gavg.update(cstate.g.array().square());

                                cstate.d = -cstate.g.array() /
                                           (epsilon + gavg.value().array()).sqrt();

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                                {{"alpha0", alpha0}, {"epsilon", epsilon}});
                }
        };
}

