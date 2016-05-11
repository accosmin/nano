#pragma once

#include "params.hpp"

namespace nano
{
        ///
        /// \brief batch optimization loop running until:
        ///     - convergence is achieved (critical point, possiblly a local/global minima) or
        ///     - the maximum number of iterations is reached or
        ///     - the user canceled the optimization (using the logging function) or
        ///     - the optimizer failed (e.g. line-search failed)
        ///
        template
        <
                typename tproblem,      ///< optimization problem
                typename toptimizer     ///< optimization algorithm
        >
        auto batch_loop(
                const batch_params_t<tproblem>& params,
                const typename batch_params_t<tproblem>::tstate& istate,
                const toptimizer& optimizer)
        {
                // current state
                auto cstate = istate;

                // for each iteration ...
                for (std::size_t i = 0; i < params.m_max_iterations; i ++)
                {
                        // check convergence
                        if (cstate.converged(params.m_epsilon))
                        {
                                break;
                        }

                        if (!optimizer(cstate, i))
                        {
                                break;
                        }

                        // log the current state & check the stopping criteria
                        if (!params.ulog(cstate))
                        {
                                break;
                        }
                }

                // OK
                return cstate;
        }
}

