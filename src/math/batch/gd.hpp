#pragma once

#include "ls_init.hpp"
#include "batch_loop.hpp"
#include "ls_strategy.hpp"

namespace math
{
        ///
        /// \brief gradient descent
        ///
        template
        <
                typename tproblem                       ///< optimization problem
        >
        struct batch_gd_t
        {
                using tparam = batch_params_t<tproblem>;
                using tstate = typename tparam::tstate;
                using tscalar = typename tparam::tscalar;
                using tvector = typename tparam::tvector;

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const tparam& param, const tproblem& problem, const tvector& x0) const
                {
                        assert(problem.size() == x0.size());

                        // line-search initial step length
                        ls_init_t<tstate> ls_init(param.m_ls_initializer);

                        // line-search step
                        ls_strategy_t<tproblem> ls_step(param.m_ls_strategy, 1e-4, 0.1);

                        const auto op = [&] (tstate& cstate, const std::size_t)
                        {
                                // descent direction
                                cstate.d = -cstate.g;

                                // line-search
                                const tscalar t0 = ls_init(cstate);
                                return ls_step(problem, t0, cstate);
                        };

                        // OK, assembly the optimizer
                        return batch_loop(param, tstate(problem, x0), op);
                }
        };
}

