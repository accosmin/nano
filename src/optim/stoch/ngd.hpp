#pragma once

#include "loop.hpp"

namespace nano
{
        ///
        /// \brief (stochastic) normalized gradient descent
        ///     see "Beyond Convexity: Stochastic Quasi-Convex Optimization",
        ///     by Elan Hazan, Kfir Y. Levi, Shai Shalev-Shwartz
        ///
        struct stoch_ngd_t
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
                        const auto config = nano::tune(op, param0);
                        return operator()(param, problem, x0, config.param0());
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                        const scalar_t alpha0) const
                {
                        assert(problem.size() == x0.size());

                        const auto op_iter = [&] (state_t& cstate)
                        {
                                // learning rate
                                const scalar_t alpha = alpha0;

                                // descent direction
                                const scalar_t norm = 1 / cstate.g.template lpNorm<2>();
                                cstate.d = -cstate.g * norm;

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                                {{"alpha0", alpha0}});
                }
        };
}

