#pragma once

#include "loop.hpp"
#include "ls_init.hpp"
#include "ls_strategy.hpp"

namespace nano
{
        ///
        /// \brief gradient descent
        ///
        struct batch_gd_t
        {
                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const batch_params_t& param, const problem_t& problem, const vector_t& x0) const
                {
                        assert(problem.size() == x0.size());

                        // line-search initial step length
                        ls_init_t ls_init(param.m_ls_initializer);

                        // line-search step
                        ls_strategy_t ls_step(param.m_ls_strategy, scalar_t(1e-4), scalar_t(0.1));

                        const auto op = [&] (state_t& cstate, const std::size_t)
                        {
                                // descent direction
                                cstate.d = -cstate.g;

                                // line-search
                                const scalar_t t0 = ls_init(cstate);
                                return ls_step(problem, t0, cstate);
                        };

                        // OK, assembly the optimizer
                        return batch_loop(param, state_t(problem, x0), op);
                }
        };
}

