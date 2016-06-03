#pragma once

#include "loop.hpp"
#include "math/momentum.hpp"

namespace nano
{
        ///
        /// \brief stochastic AdaDelta,
        ///     see "ADADELTA: An Adaptive Learning Rate Method", by Matthew D. Zeiler
        ///
        struct stoch_adadelta_t
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

                        const auto param0 = make_momenta();
                        const auto param1 = make_epsilons();
                        const auto config = nano::tune(op, param0, param1);
                        return operator()(param, problem, x0, config.param0(), config.param1());
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                        const scalar_t momentum, const scalar_t epsilon) const
                {
                        assert(problem.size() == x0.size());

                        // second-order momentum of the gradient
                        momentum_vector_t<vector_t> gavg(momentum, x0.size());

                        // second-order momentum of the step updates
                        momentum_vector_t<vector_t> davg(momentum, x0.size());

                        const auto op_iter = [&] (state_t& cstate)
                        {
                                // learning rate
                                const scalar_t alpha = 1;

                                // descent direction
                                gavg.update(cstate.g.array().square());

                                cstate.d = -cstate.g.array() *
                                           (epsilon + davg.value().array()).sqrt() /
                                           (epsilon + gavg.value().array()).sqrt();

                                davg.update(cstate.d.array().square());

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                                {{"momentum", momentum}, {"epsilon", epsilon}});
                }
        };
}

